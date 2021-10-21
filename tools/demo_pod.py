# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import threading
import torch
import numpy as np
from xml.dom.minidom import Document

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.polygon_processing_modified import py_cpu_oriented_nms_modified, bbox_adjust_polygon, validate_clockwise_points, py_cpu_polygon_nms_modified, bbox_adjust_polygon_v2
from detectron2.utils.polyfit_text_line import polyfit_curve_text_line
from detectron2.utils.curve_text_classification import curve_text_lines_classification
from detectron2.structures import Boxes, QuadBoxes, Instances
from detectron2.engine.defaults import DefaultPredictor

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.OUTPUT_DIR = args.output
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--im_or_folder',
        default="",
        help='directory to load images for demo')
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        '--no_demo', action='store_true'
    )
    parser.add_argument(
        '--do_eval', action='store_true'
    )
    parser.add_argument(
        '--num_loader', type=int, default=0,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_text_lines(rois, rois_scores, poly_rois, rois_group_id, rois_quad_labels, roi_classes, frcn_conf_thresh, deg, textline_points):

    group_num = int(np.max(rois_group_id)) + 1
    poly_text_lines = np.zeros((0, 2 * textline_points + 3), dtype=np.float32)
    for group_id in range(group_num):
        group = np.where(rois_group_id == group_id)[0]
        group_rois = rois[group, :]
        group_rois_scores = rois_scores[group]
        group_poly_rois = poly_rois[group, :]
        group_rois_quad_labels = rois_quad_labels[group]
        group_rois_classes = roi_classes[group]
        assert len(set(list(group_rois_classes.reshape(-1)))) == 1
        group_rois_quad_label = group_rois_quad_labels.mean()
        group_rois_class = group_rois_classes.mean()
        poly_text_line = polyfit_curve_text_line(group_poly_rois, group_rois_scores, group_rois_quad_label, deg, int(textline_points // 2))
        poly_text_lines = np.vstack((poly_text_lines, np.hstack((poly_text_line, np.array([group_rois_class])))))

    return poly_text_lines

def text_detection_generation(cfg, output_blob):
    # setting params:
    FRCN_CONF_THRESH = 0.8 # segment wise text/non-text classifications
    DEG = 4 #2 # Degree of the fitting polynomial
    TEXTLINE_POINTS = 32 #16 # using n_points to represent arbitrary curved text (upper_bound and under_bound)

    TEXTLINE_CONF_THRESH = 0.8
    TEXTLINE_NMS_THRESH = 0.2
    MASK_AREA_THRESH = 0.8 # for curve/non-curve text-lines classification

    k_min = int(np.log2(cfg.MODEL.RPN.MSDENSEBOX_LAYER_PARAMS[0]["feat_stride"]))
    k_max = int(np.log2(cfg.MODEL.RPN.MSDENSEBOX_LAYER_PARAMS[-1]["feat_stride"]))

    poly_text_lines = np.zeros((0, 2 * TEXTLINE_POINTS + 2), dtype=np.float32)
    fpn_level_flags = np.zeros((0, ), dtype=np.float32)
    text_line_classes = np.zeros((0, ), dtype=np.float32)

    for lvl in range(k_min, k_max + 1):
        slvl = str(lvl)
        rpn_rois = output_blob['rpn_rois_fpn' + slvl]
        rpn_poly_rois = output_blob['rpn_poly_rois_fpn' + slvl]
        rpn_roi_scores = output_blob['rpn_roi_probs_fpn' + slvl]
        rpn_roi_group_id = output_blob['rpn_roi_group_id_fpn' + slvl]
        rpn_roi_quad_labels = output_blob['rpn_roi_quad_labels_fpn' + slvl]
        rpn_roi_classes = output_blob['rpn_roi_classes_fpn' + slvl]
        poly_text_lines_fpn_lvl = get_text_lines(rpn_rois, rpn_roi_scores, rpn_poly_rois, rpn_roi_group_id, rpn_roi_quad_labels, rpn_roi_classes, FRCN_CONF_THRESH, DEG, TEXTLINE_POINTS)
        if len(poly_text_lines_fpn_lvl) >= 1:
            poly_text_lines = np.vstack((poly_text_lines, poly_text_lines_fpn_lvl[:, :-1]))
            text_line_classes = np.hstack((text_line_classes, poly_text_lines_fpn_lvl[:, -1].reshape(-1)))
            fpn_lvl_flags = np.ones((poly_text_lines_fpn_lvl.shape[0],), dtype=np.int32) * lvl
            fpn_level_flags = np.hstack((fpn_level_flags, fpn_lvl_flags))

    if len(poly_text_lines) >= 1:
        # curve_text_labels, text_rotated_rect_bboxes for following text recognition
        curve_text_labels, text_rotated_rect_bboxes = curve_text_lines_classification(poly_text_lines, mask_area_thresh = MASK_AREA_THRESH)

        conf_keep = np.where((poly_text_lines[:, -2]) >= TEXTLINE_CONF_THRESH)[0]
        poly_text_lines = poly_text_lines[conf_keep, :]
        text_line_classes = text_line_classes[conf_keep]
        curve_text_labels = curve_text_labels[conf_keep]
        text_rotated_rect_bboxes = text_rotated_rect_bboxes[conf_keep, :]
        fpn_level_flags = fpn_level_flags[conf_keep]

        nms_keep = py_cpu_polygon_nms_modified(text_rotated_rect_bboxes, poly_text_lines[:, :-1], TEXTLINE_NMS_THRESH, 'IoU')
        poly_text_lines = poly_text_lines[nms_keep, :]
        text_line_classes = text_line_classes[nms_keep]
        curve_text_labels = curve_text_labels[nms_keep]
        text_rotated_rect_bboxes = text_rotated_rect_bboxes[nms_keep, :]
        fpn_level_flags = fpn_level_flags[nms_keep]

        combine_idx = bbox_adjust_polygon_v2(text_rotated_rect_bboxes, poly_text_lines[:, :-1], score_margin=0.01, overlap_threshold=0.3)
        poly_text_lines = poly_text_lines[combine_idx, :]
        text_line_classes = text_line_classes[combine_idx]
        curve_text_labels = curve_text_labels[combine_idx]
        text_rotated_rect_bboxes = text_rotated_rect_bboxes[combine_idx, :]
        fpn_level_flags = fpn_level_flags[combine_idx]

    if len(poly_text_lines) == 0:
        text_rotated_rect_bboxes = np.zeros((0, 8))
    return poly_text_lines, text_line_classes, text_rotated_rect_bboxes, fpn_level_flags, TEXTLINE_POINTS

def postprocess_text_formu(cfg, predictions):
    poly_text_lines, text_line_classes, text_rotated_rect_bboxes, fpn_level_flags, TEXTLINE_POINTS = text_detection_generation(cfg, predictions)
    new_predictions = Instances(predictions["tab_fig_instances"].image_size)
    new_predictions.scores = torch.from_numpy(poly_text_lines[:, -2])
    new_predictions.quad_labels = torch.from_numpy(poly_text_lines[:, -1])
    new_predictions.poly_text_lines = torch.from_numpy(poly_text_lines[:, :-2])
    new_predictions.text_line_classes = torch.from_numpy(text_line_classes)
    new_predictions.text_rotated_rect_bboxes = torch.from_numpy(text_rotated_rect_bboxes)
    new_predictions.fpn_level_flags = torch.from_numpy(fpn_level_flags)
    return {"text_formu_instances": new_predictions}

def postprocess_tab_fig(cfg, predictions):
    '''
    extra post-propocess, including Poly-NMS, bbox_adjust_polygon, validate_clockwise_points
    '''
    if cfg.MODEL.RPN_ONLY == True:
        proposal_boxes = predictions["tab_fig_instances"].proposal_boxes
        objectness_logits = predictions["tab_fig_instances"].objectness_logits
        classes = predictions["tab_fig_instances"].classes + 1 # make the class_id start from `1` for fg
        boxes = proposal_boxes
        scores = objectness_logits # the objectness_logits have already been performed sigmoid
        keep = torch.nonzero(scores>cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, as_tuple=True)
        boxes = boxes[keep].tensor.cpu().numpy()
        scores = scores[keep].cpu().numpy()
        classes = classes[keep].cpu().numpy()

        polygon_dets = np.hstack((boxes, scores.reshape((-1, 1))))
        keep = py_cpu_oriented_nms_modified(polygon_dets, cfg.MODEL.ROI_HEADS.POLY_NMS_THRESH_TEST, 'IoU')
        polygon_dets = polygon_dets[keep, :]
        classes = classes[keep]
        keep = bbox_adjust_polygon(polygon_dets)
        polygon_dets = polygon_dets[keep, :]
        classes = classes[keep]
        keep = []
        for idx, box in enumerate(polygon_dets):
            if validate_clockwise_points(box[:8]) == True:
                keep.append(idx)
        polygon_dets = polygon_dets[keep, :]
        classes = classes[keep]
        boxes = polygon_dets[:, :8]
        scores = polygon_dets[:, 8]
        new_predictions = Instances(predictions["tab_fig_instances"].image_size)
        new_predictions.pred_boxes = QuadBoxes(torch.from_numpy(boxes))
        new_predictions.scores = torch.from_numpy(scores)
        new_predictions.pred_classes = torch.from_numpy(classes)
        return {"tab_fig_instances": new_predictions}
    else:
        boxes = predictions["tab_fig_instances"].pred_boxes
        scores = predictions["tab_fig_instances"].scores
        classes = predictions["tab_fig_instances"].pred_classes + 1 # make the class_id start from `1` for fg
        boxes = boxes.tensor.cpu().numpy()
        scores = scores.cpu().numpy()
        classes = classes.cpu().numpy().astype(np.int32)

        polygon_dets = np.hstack((boxes, scores.reshape((-1, 1))))
        keep = py_cpu_oriented_nms_modified(polygon_dets, cfg.MODEL.ROI_HEADS.POLY_NMS_THRESH_TEST, 'IoU')
        polygon_dets = polygon_dets[keep, :]
        classes = classes[keep]
        # keep = bbox_adjust_polygon(polygon_dets)
        # polygon_dets = polygon_dets[keep, :]
        # classes = classes[keep]
        keep = []
        for idx, box in enumerate(polygon_dets):
            if validate_clockwise_points(box[:8]) == True:
                keep.append(idx)
        polygon_dets = polygon_dets[keep, :]
        classes = classes[keep]
        boxes = polygon_dets[:, :8]
        scores = polygon_dets[:, 8]
        new_predictions = Instances(predictions["tab_fig_instances"].image_size)
        new_predictions.pred_boxes = QuadBoxes(torch.from_numpy(boxes))
        new_predictions.scores = torch.from_numpy(scores)
        new_predictions.pred_classes = torch.from_numpy(classes)
        return {"tab_fig_instances": new_predictions}

def draw_box_on_img(box, draw_surface, color, width=3):
    box = box.reshape(-1).astype(np.int32)
    num_points = len(box) // 2
    circle_color = [(0,0,255),(0,255,255),(255,0,255),(0,255,0)] * (num_points // 4)
    for i in range(num_points-1):
        i = i * 2
        cv2.circle(draw_surface,(box[i], box[i+1]),25,circle_color[i//2], -1)
        cv2.line(draw_surface, (box[i], box[i+1]), (box[i+2], box[i+3]), color, width)
    cv2.circle(draw_surface,(box[-2], box[-1]),25,circle_color[-1], -1)
    cv2.line(draw_surface, (box[-2], box[-1]), (box[0], box[1]), color, width)

def img_reader(img_list, img_queue, idxes, package):
    n = len(idxes)
    for i, idx in enumerate(idxes):
        img_name = img_list[idx]
        img = read_image(img_name, format="BGR", package=package)
        img_queue[idx] = img

def random_rotate(image, angle_list=[0], jitter_degree=5):
    # generate a random ratation angle
    angle = np.random.choice(np.array(angle_list)) + (np.random.rand() * 2 * jitter_degree - jitter_degree)
    if angle > 180:
        angle = angle - 360

    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))
    return image

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    if os.path.isfile(args.im_or_folder):
        name_list = [args.im_or_folder]
    elif os.path.isdir(args.im_or_folder):
        name_list = sorted([os.path.join(args.im_or_folder, file_name) for file_name in os.listdir(args.im_or_folder) if file_name != "Thumbs.db"])[:]
    else:
        # To perform evaluation only
        assert args.im_or_folder == ""
        name_list = []

    package = cfg.INPUT.IMAGE_PROC_PACKAGE
    if args.num_loader > 0:
        # Use multi-threads to read images
        start_time = time.time()
        num_threads = args.num_loader
        threads = []
        num_images = len(name_list)
        img_queue = [0 for i in range(num_images)]
        for i in range(num_threads):
            threads.append(threading.Thread(
                target=img_reader,
                args=(name_list, img_queue, range(i, num_images, num_threads), package))
            )
        for thread in threads:
            thread.start()
        print("All loader start!")
        for thread in threads:
            thread.join()
        print("Take {}s to load all images.".format(time.time() - start_time))

    assert args.output
    out_dirs = [args.output, os.path.join(args.output, "txt"), os.path.join(args.output, "image")]
    for _ in out_dirs:
        if not os.path.isdir(_):
            os.makedirs(_)

    predictor = DefaultPredictor(cfg)

    count = 0
    for path in tqdm.tqdm(name_list, disable=False):
        logger.info("Processing: {}".format(path))
        # use PIL, to be consistent with evaluation
        if args.num_loader > 0:
            img = img_queue[count]
        else:
            img = read_image(path, format="BGR", package=package)

        # # for debug
        # img = random_rotate(img, angle_list=[5], jitter_degree=0)

        count += 1
        start_time = time.time()
        predictions = predictor(img)
        # extra post-propocess, including Poly-NMS, bbox_adjust_polygon, validate_clockwise_points
        head_names = cfg.MODEL.RPN.HEAD_NAME.split("_")
        if head_names[0] != "None": # head_names[0] is a placeholder for cornernet_head family, which is for table and figure detection
            predictions.update(postprocess_tab_fig(cfg, predictions))
        if head_names[1] != "None": # head_names[1] is a placeholder for seglink_head family, which is for text and formula detection
            predictions.update(postprocess_text_formu(cfg, predictions))

        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["tab_fig_instances"]) + len(predictions["text_formu_instances"]))
                if "tab_fig_instances" in predictions and "text_formu_instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        imname = os.path.basename(path).split('.')[0]
        out_txt_name = os.path.join(args.output, "txt", imname + '.txt')
        if not args.no_demo:
            im_vis = np.array(img.copy()).astype(np.float32)
        with open(out_txt_name, 'w') as f:
            if head_names[0] != "None":
                boxes = predictions["tab_fig_instances"].pred_boxes.tensor.numpy()
                scores = predictions["tab_fig_instances"].scores.numpy()
                classes = predictions["tab_fig_instances"].pred_classes.numpy()
                for idx, box in enumerate(boxes):
                    box = box.reshape(-1)
                    score = scores[idx]
                    cls = classes[idx]
                    if cls == 1:
                        color = (255, 0, 0) # blue for figure
                    if cls == cfg.MODEL.ROI_HEADS.NUM_CLASSES:
                        color = (0, 165, 255) # orange for table
                    # if cls == cfg.MODEL.ROI_HEADS.NUM_CLASSES: # record table coords for evaluation
                    if True:
                        f.write(str(int(box[0])) + ',')
                        f.write(str(int(box[1])) + ',')
                        f.write(str(int(box[2])) + ',')
                        f.write(str(int(box[3])) + ',')
                        f.write(str(int(box[4])) + ',')
                        f.write(str(int(box[5])) + ',')
                        f.write(str(int(box[6])) + ',')
                        f.write(str(int(box[7])) + ',')
                        f.write(str(score) + '\n')
                        # f.write(str(score) + ',{}'.format("Figure" if cls==1 else "Table") + '\n')
                    if not args.no_demo:
                        if score > 0.3:
                            draw_box_on_img(box, im_vis, color, width=5)
                            cv2.putText(im_vis, '{:.3f}'.format(score), (int(box[0]), int(np.maximum(int(box[1]) - 5, 0))), 0, 3.0,  (0, 255, 255), 10)
            if head_names[1] != "None":
                boxes = predictions["text_formu_instances"].poly_text_lines.numpy()
                scores = predictions["text_formu_instances"].scores.numpy()
                classes = predictions["text_formu_instances"].text_line_classes.numpy()
                quad_labels = predictions["text_formu_instances"].quad_labels.numpy()
                for idx, box in enumerate(boxes):
                    box = box.reshape(-1).astype(np.int32)
                    score = scores[idx]
                    cls = classes[idx]
                    is_horizontal = int(quad_labels[idx])
                    if cls == 1:
                        color = (0, 0, 255) # red for text
                    else:
                        assert cls == 2
                        color = (0, 255, 0) # green for formula
                    coord_str = [str(_) for _ in box]
                    coord_str = ",".join(coord_str)
                    # f.write(coord_str)
                    # f.write(',')
                    # # f.write(str(score) + '\n')
                    # f.write(str(score) + ',{},{}'.format(is_horizontal, "Text" if cls==1 else "Formula") + '\n')
                    if not args.no_demo:
                        draw_box_on_img(box, im_vis, color, width=10)
                        # cv2.putText(im_vis, '{:.3f}'.format(score), (int(box[0]), int(np.maximum(int(box[1]) - 5, 0))), 0, 3.0,  (0, 255, 255), 10)
            f.close()
        if not args.no_demo and head_names[2] != "None": # head_names[2] is a placeholder for slp_head family
            im_vis_slp = np.array(img.copy()).astype(np.float32)
            for prefix, color in zip(['row', 'col'], [np.array([[[255, 255, 0]]]), np.array([[[255, 0, 255]]])]):
                pred_solid_line_probs = predictions["pred_{}_solid_line_probs".format(prefix)].cpu().numpy()[0, 0, :, :]
                if cfg.MODEL.RPN.SLP_CLS_FEAT_STRIDE > 1:
                    stride = cfg.MODEL.RPN.SLP_CLS_FEAT_STRIDE
                    pred_solid_line_probs = cv2.resize(pred_solid_line_probs, None, None, fx=stride, fy=stride, interpolation=cv2.INTER_LINEAR)
                pred_solid_line_probs = pred_solid_line_probs[:predictions["input_height"], :predictions["input_width"]]
                pred_solid_line_probs = cv2.resize(pred_solid_line_probs, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = (pred_solid_line_probs > cfg.MODEL.RPN.SLP_SCORE_THRESH).astype(np.uint8)
                # cv2.imwrite(os.path.join(args.output, "image", imname.split('.')[0]+"_{}_mask.jpg".format(prefix)), mask * 255)
                idx = np.nonzero(mask)
                im_vis_slp[idx[0], idx[1], :] = im_vis_slp[idx[0], idx[1], :] * 0.2 + color * 0.8
            # vis cells from physical lines
            for cell in predictions["cell_polygons"]:
                cell = cell.astype(np.int32)
                cell_point_num = len(cell) // 2
                for idx in range(cell_point_num):
                    next_idx = (idx + 1) % cell_point_num
                    cv2.line(im_vis_slp, (cell[idx * 2], cell[idx * 2 + 1]), (cell[next_idx * 2], cell[next_idx * 2 + 1]), (0, 0, 255), 10)
                    cv2.circle(im_vis_slp, (cell[idx * 2], cell[idx * 2 + 1]), 8, (0, 255, 255), -1)
                    cv2.circle(im_vis_slp, (cell[next_idx * 2], cell[next_idx * 2 + 1]), 8, (0, 255, 255), -1)
            out_image_name = os.path.join(args.output, "image", imname + '_slp.jpg')
            # im_vis_slp = cv2.resize(im_vis_slp, None, None, fx=0.4, fy=0.4)
            cv2.imwrite(out_image_name, im_vis_slp)

        if not args.no_demo:
            out_image_name = os.path.join(args.output, "image", imname + '.jpg')
            # im_vis = cv2.resize(im_vis, None, None, fx=0.4, fy=0.4)
            cv2.imwrite(out_image_name, im_vis)

    if args.do_eval:
        if cfg.DATASETS.TEST[0] == 'POD2017':
            # generate a xml file for evaluation tool
            with open(os.path.join(args.output, 'submission.xml'), 'w') as f_out:
                f_out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                filenames = sorted(os.listdir(os.path.join(args.output, 'txt')))
                for filename in filenames:
                    f_out.write('<document filename="{}.bmp">\n'.format(filename.split('.')[0]))
                    det_txt = open(os.path.join(args.output, 'txt', filename), 'r')
                    lines = det_txt.readlines()
                    for line in lines:
                        line = line.strip().split(',')
                        coords = [int(_) for _ in line[:8]]
                        score = float(line[8])
                        f_out.write('\t<tableRegion prob="{}">\n'.format(score))
                        f_out.write('\t\t<Coords points="{},{} {},{} {},{} {},{}"/>\n'.format(coords[0], coords[1], coords[2], coords[3], coords[6], coords[7], coords[4], coords[5]))
                        f_out.write('\t</tableRegion>\n')
                    f_out.write('</document>\n')
            cmd  = "conda activate py27 \n"
            cmd += "cd ./tools/POD2017_eval_tool/ \n"
            cmd += "python evaluate.py {} \n".format(os.path.abspath(args.output))
            cmd += "cd ../../ \n"
            cmd += "conda activate detectron2 \n"
            print("Note: Please manually copy the following scripts into cmd line to perform evaluation.")
            print(cmd)
            # Here, we need to manually copy these scripts into cmd line to perform evaluation, because the evaluation tool need a python2 env.
            # os.system(cmd)
        elif cfg.DATASETS.TEST[0] in ['cTDaR2019_TRACKA', 'publaynet_val']:
            if not os.path.isdir(os.path.join(args.output, "xml")):
                os.makedirs(os.path.join(args.output, "xml"))
            for th in range(500, 1000, 25):
                th = th / 1000.0
                print("\nscore thresh @ {:.3f}:".format(th))
                filenames = sorted(os.listdir(os.path.join(args.output, "txt")))
                for filename in filenames:
                    doc = Document()
                    document = doc.createElement("document")
                    document.setAttribute("filename", filename.split('.')[0])
                    doc.appendChild(document)
                    with open(os.path.join(args.output, "txt", filename), 'r') as f_in:
                        for line in f_in.readlines():
                            line = line.strip().split(',')
                            box = [int(_) for _ in line[:8]]
                            score = float(line[8])
                            if score > th:
                                table = doc.createElement("table")
                                document.appendChild(table)
                                Coords = doc.createElement("Coords")
                                points_str = "{},{} {},{} {},{} {},{}".format(box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7])
                                Coords.setAttribute("points", points_str)
                                table.appendChild(Coords)
                    xml_result_file = os.path.join(args.output, "xml", filename.split('.')[0] + '.xml')
                    with open(xml_result_file, "w") as f:
                        f.write(doc.toprettyxml(indent="  "))
                cmd = "cd ./tools/ctdar_measurement_tool/ \n"
                if cfg.DATASETS.TEST[0] == 'cTDaR2019_TRACKA':
                    cmd += "python evaluate.py -cTDaR2019_trackA {} \n".format(os.path.join(os.path.abspath(args.output), "xml"))
                elif cfg.DATASETS.TEST[0] == 'publaynet_val':
                    cmd += "python evaluate.py -publaynet_val_trackA {} \n".format(os.path.join(os.path.abspath(args.output), "xml"))
                cmd += "cd ../../ \n"
                os.system(cmd)
            print(cmd)
            if cfg.DATASETS.TEST[0] == 'publaynet_val':
                cmd = "cd ./tools/publaynet_eval_tool/ \n" #i.e., COCO evaluation tool
                cmd += "python ./evaluate.py {} \n".format(os.path.join(os.path.abspath(args.output), "txt"))
                cmd += "cd ../../ \n"
                os.system(cmd)
                print(cmd)
        else: # for MSRA_POD
            score_th_list = [str(th / 1000.0) for th in range(100, 1000, 25)]
            score_th_list = " ".join(score_th_list)
            cmd = "cd ./tools/msra_pod_measurement_tool/ \n"
            cmd += "python sortdetection.py {} {} \n".format(os.path.join(os.path.abspath(args.output), "txt"), score_th_list)
            cmd += "python test_ctw1500_eval.py {} \n".format(score_th_list)
            cmd += "cd ../../ \n"
            print(cmd)
            os.system(cmd)
