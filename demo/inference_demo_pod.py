from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import argparse
import cv2
import numpy as np
import os
import time
import threading
import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description="MMDetection demo for builtin models")
    parser.add_argument(
        "--config",
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
        "--num_loader", type=int, default=0,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--checkpoint', default=os.path.join('./latest.pth'),help='checkpoint file')
    return parser

def read_image(file_name, format=None, package="PIL"):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"

    Returns:
        image (np.ndarray): an HWC image in the given format.
    """
    from PIL import Image, ImageOps
    if package == "PIL":
        with open(file_name, "rb") as f:
            image = Image.open(f)

            # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass

            if format is not None:
                # PIL only supports RGB, so convert to RGB and flip channels over below
                conversion_format = format
                if format == "BGR":
                    conversion_format = "RGB"
                image = image.convert(conversion_format)
            image = np.asarray(image)
            if format == "BGR":
                # flip channels if needed
                image = image[:, :, ::-1]
            # PIL squeezes out the channel dimension for "L", so make it HWC
            if format == "L":
                image = np.expand_dims(image, -1)
            return image
    else:
        assert package == "cv2"
        image = cv2.imread(file_name)
        return image

def img_reader(img_list, img_queue, idxes, package):
    n = len(idxes)
    for i, idx in enumerate(idxes):
        img_name = img_list[idx]
        img = read_image(img_name, format="BGR", package=package)
        img_queue[idx] = img

if __name__ == "__main__":
    config_file = '../configs/DARDet/exp1.py'
    # checkpoint_file = '../latest.pth'
    args = get_parser().parse_args()
    if args.config:
        config_file = args.config
    checkpoint_file = args.checkpoint

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    if os.path.isfile(args.im_or_folder):
        name_list = [args.im_or_folder]
    elif os.path.isdir(args.im_or_folder):
        name_list = sorted([os.path.join(args.im_or_folder, file_name) for file_name in os.listdir(args.im_or_folder) if file_name != "Thumbs.db"])[:]
    else:
        # To perform evaluation only
        assert args.im_or_folder == ""
        name_list = []
    
    package = 'PIL'
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

    count = 0
    for path in tqdm.tqdm(name_list, disable=False):
        print("Processing: {}".format(path))
        if args.num_loader > 0:
            img = img_queue[count]
        else:
            img = read_image(path, format="BGR", package=package)
        count += 1
        # start_time = time.time()
        result = inference_detector(model, img)
        # time.time() - start_time
        if hasattr(model, 'module'):
            model_ = model.module
        else:
            model_ = model
        imname = os.path.basename(path).split('.')[0]
        out_txt_name = os.path.join(args.output, "txt", imname + '.txt')
        out_image_name = os.path.join(args.output, "image", imname + '.jpg')
        model_.show_result(
            img,
            result,
            score_thr=0.5,
            show=True,
            wait_time=0,
            win_name=result,
            bbox_color=(72, 101, 241),
            text_color=(72, 101, 241),out_file=out_image_name,thickness=8,font_scale=8)

        bboxes = result[0]
        with open(out_txt_name, 'w') as f:
            boxes = bboxes[...,10:]
            scores = bboxes[...,4:5]
            for idx, box in enumerate(boxes):
                box = box.reshape(-1)
                score = scores[idx]
                if True:
                    f.write(str(int(box[0])) + ',')
                    f.write(str(int(box[1])) + ',')
                    f.write(str(int(box[2])) + ',')
                    f.write(str(int(box[3])) + ',')
                    f.write(str(int(box[4])) + ',')
                    f.write(str(int(box[5])) + ',')
                    f.write(str(int(box[6])) + ',')
                    f.write(str(int(box[7])) + ',')
                    f.write(str(score[0]) + '\n')
    
    score_th_list = [str(th / 1000.0) for th in range(100, 1000, 25)]
    score_th_list = " ".join(score_th_list)
    cmd = "cd ./tools/msra_pod_measurement_tool_45/ \n"
    cmd += "python sortdetection.py {} {} \n".format(os.path.join(os.path.abspath(args.output), "txt"), score_th_list)
    cmd += "python test_ctw1500_eval.py {} \n".format(score_th_list)
    cmd += "cd ../../ \n"
    print(cmd)
    os.system(cmd)
        
        
