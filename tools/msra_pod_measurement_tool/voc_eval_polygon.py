# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# Modified by yl
# --------------------------------------------------------

import os
# import cPickle
import pickle
import numpy as np
import cv2
import math
from six.moves import xrange

from shapely.geometry import *
import xml.etree.cElementTree as ET

def parse_rec_txt(filename):
    with open(filename.strip(),'r') as f:
        gts = f.readlines()
        objects = []
        for obj in gts:
            cors = obj.strip().split(',')
            obj_struct = {}
            obj_struct['name'] = 'text'
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [int(cors[0]),
                                  int(cors[1]),
                                  int(cors[2]),
                                  int(cors[3])]
            objects.append(obj_struct)
    return objects

def rotate_box(point1, point2, point3, point4, mid_x, mid_y, theta):
    theta = -theta * math.pi / 180
    sin = math.sin(theta)
    cos = math.cos(theta)
    point1 = point1 - [mid_x, mid_y]
    point2 = point2 - [mid_x, mid_y]
    point3 = point3 - [mid_x, mid_y]
    point4 = point4 - [mid_x, mid_y]
    x1 = point1[0] * cos - point1[1] * sin + mid_x
    y1 = point1[0] * sin + point1[1] * cos + mid_y
    x2 = point2[0] * cos - point2[1] * sin + mid_x
    y2 = point2[0] * sin + point2[1] * cos + mid_y
    x3 = point3[0] * cos - point3[1] * sin + mid_x
    y3 = point3[0] * sin + point3[1] * cos + mid_y
    x4 = point4[0] * cos - point4[1] * sin + mid_x
    y4 = point4[0] * sin + point4[1] * cos + mid_y
    return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

def quadrangle2minAreaRect(quad_coord_boxes):
    quad_coord = np.array(quad_coord_boxes).reshape((4,2))
    min_area_rect = cv2.minAreaRect(quad_coord)
    mid_x, mid_y = min_area_rect[0]
    theta = min_area_rect[2]
    box = cv2.boxPoints(min_area_rect)
    # determine the minAreaRect direction
    # reference: http://blog.csdn.net/sunflower_boy/article/details/51170232
    x0 = box[0][0]
    count = np.sum(box[:,0].reshape(-1)>x0)

    if count >= 2:
        theta = theta
        hori_box = rotate_box(box[1], box[2], box[3], box[0], mid_x, mid_y, theta)
    else:
        theta = 90 + theta
        hori_box = rotate_box(box[2], box[3], box[0], box[1], mid_x, mid_y, theta)

    min_x = np.min(hori_box[:,0])
    min_y = np.min(hori_box[:,1])
    max_x = np.max(hori_box[:,0])
    max_y = np.max(hori_box[:,1])
    mid_x = (min_x+max_x)/2.0
    mid_y = (min_y+max_y)/2.0

    # normalize the rotate angle in -45 to 45
    items = [min_x, min_y, max_x, max_y]
    if theta > 90:
        theta = theta - 180
    if theta < -90:
        theta = theta + 180
    if theta > 45:
        theta = theta - 90
        width = items[3] - items[1]
        height = items[2] - items[0]
    elif theta < -45:
        theta = theta + 90
        width = items[3] - items[1]
        height = items[2] - items[0]
    else:
        width = items[2] - items[0]
        height = items[3] - items[1]
    return [mid_x,mid_y,width,height,-theta]# positive degree for the gt box rotated counter-clockwisely to the horizontal rectangle

def curve_parse_rec_txt(filename):
    with open(filename.strip(),'r') as f:
        gts = f.readlines()
        objects = []
        if len(gts) == 0:
            obj_struct = {}
            obj_struct['name'] = 'text'
            obj_struct['difficult'] = 1
            obj_struct['bbox'] = []
            # obj_struct['minAreaRect'] = []
            objects.append(obj_struct)
        else:
            for obj in gts:
                cors = obj.strip().split(',')
                obj_struct = {}
                obj_struct['name'] = 'text'
                # if cors[-1] == "-1":
                #     obj_struct['difficult'] = 1
                #     print('difficult')
                # else:
                #     obj_struct['difficult'] = 0
                obj_struct['difficult'] = 0
                # obj_struct['bbox'] = [int(cors[0]), int(cors[1]),int(cors[2]),int(cors[3]),
                #                       int(cors[4]), int(cors[5]),int(cors[6]),int(cors[7])]
                obj_struct['bbox'] = [int(coor) for coor in cors]
                # obj_struct['minAreaRect'] = quadrangle2minAreaRect(obj_struct['bbox'])
                objects.append(obj_struct)
    return objects

def is_valid_tag(tags):
    all_tags = tags.split('|')
    valid = True
    count_tag = 0
    for cls in ['Text', 'Formula', 'FormulaSN', 'Figure', 'Table', 'Table_Form', 'ItemList', 'Table_keyvalue_vertical', 'Table_keyvalue_horizontal']:
        if cls in all_tags:
            count_tag += 1
    if count_tag == 0:
        tags += "|Text"
    elif count_tag != 1:
        valid = False
        # print(valid)
    return valid

def curve_parse_rec_xml(filename):
    tree = ET.parse(filename.strip())
    root = tree.getroot()
    objects = []
    for elem in root.iter('Line'):
        poly = elem.find('Polygon')
        tags = elem.find('Tag')

        tag_notsure = 0  # 0 for text, 1 for ambiguous
        if tags is None:
            continue
        else:
            tags = tags.text
            if tags is None:
                continue

        valid = is_valid_tag(tags)
        if valid == False:
            tag_notsure = 1
        if 'NotSure' in tags.split('|'):
            tag_notsure = 1
        # if not ('Table' in tags.split('|')):
        if not ('Table' in tags.split('|') or 'Table_Form' in tags.split('|') or 'ItemList' in tags.split('|') or 'Table_keyvalue_vertical' in tags.split('|') or 'Table_keyvalue_horizontal' in tags.split('|')):
            if tag_notsure == 0:
                continue

        # if not (('Table' in tags.split('|')) and ('Text' not in tags.split('|'))):
        #     continue

        if poly is None:
            continue
        else:
            poly = poly.text
            if poly is None:
                continue
        items = poly.split(' ')
        obj_struct = {}
        obj_struct['name'] = 'text'
        obj_struct['difficult'] = tag_notsure
        obj_struct['bbox'] = [int(coor) for coor in items]
        objects.append(obj_struct)
    if len(objects) == 0:
        obj_struct = {}
        obj_struct['name'] = 'text'
        obj_struct['difficult'] = 1
        obj_struct['bbox'] = []
        objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval_polygon(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    #detpath = "split_result/technical-publications-scan.txt"
    # detpath = "detections_text0.8.txt"
    # first load gt
    cachefile = 'annots.pkl'
    # read list of images
    # with open(imagesetfile, 'r') as f, open(annopath, 'r') as fa:
    #     lines = f.readlines()
    #     anno_lines = fa.readlines()
    # imagenames = [x.strip() for x in lines]
    imagenames = [x[:-4] for x in sorted(os.listdir(r'./test_dataset/images/'))]
    # anno_names = [y.strip() for y in anno_lines]
    anno_names = [os.path.join(r'./test_dataset/xml/', y) for y in sorted(os.listdir(r'./test_dataset/xml/'))]
    assert(len(imagenames) == len(anno_names)), 'each image should correspond to one label file'

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            assert imagename == os.path.basename(anno_names[i])[:-4]
            print(anno_names[i].strip())
            # recs[imagename] = curve_parse_rec_txt(anno_names[i])
            recs[imagename] = curve_parse_rec_xml(anno_names[i])
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            # cPickle.dump(recs, f)
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            # recs = cPickle.load(f)
            recs = pickle.load(f)

    class_recs = {}
    npos = 0
    for ix, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # text
        # assert(R), 'Can not find any object in '+ classname+' class.'
        if not R:
            continue
        # bbox = np.array([x['bbox'] for x in R])
        bbox = [np.array(x['bbox']) for x in R]
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    # BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    BB = [np.array([float(z) for z in x[2:]]) for x in splitlines]
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    # BB = BB[sorted_ind, :]
    # image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d] # mask rcnn
        det_bbox = bb[:]
        pts = [(det_bbox[j], det_bbox[j+1]) for j in xrange(0,len(bb),2)]
        try:
            pdet = Polygon(pts)
        except Exception as e:
            print('exception pos1:', e)
            continue
        if not pdet.is_valid:
            print('exception pos2: predicted polygon has intersecting sides.')
            print(pts, '{}.jpg'.format(image_ids[d]))
            continue

        ovmax = -np.inf
        BBGT = R['bbox']#.astype(float)
        info_bbox_gt = BBGT#[:, 0:8]
        ls_pgt = []
        overlaps = np.zeros(len(BBGT))
        for iix in xrange(len(BBGT)):
            if len(info_bbox_gt[iix]) % 2!=0:
                print(info_bbox_gt[iix])
                info_bbox_gt[iix] = info_bbox_gt[iix][:-1]
            try:
                pts = [(info_bbox_gt[iix][j], info_bbox_gt[iix][j+1]) for j in xrange(0,len(info_bbox_gt[iix]),2)]
            except:
                print(len(info_bbox_gt[iix]))
                print(len(BBGT))
                print(info_bbox_gt[iix])
                print(BBGT)
            pgt = Polygon(pts)
            if not pgt.is_valid:
                # print('exception pos3: GT polygon has intersecting sides.')
                continue
            try:
                sec = pdet.intersection(pgt)
            except Exception as e:
                print('exception pos4: intersect invalid',e)
                continue
            try:
                assert(sec.is_valid), 'exception pos4: polygon has intersection sides.' # for mask rcnn
            except Exception as e:
                print(e)
                continue
            inters = sec.area
            uni = pgt.area + pdet.area - inters
            if uni <= 0.00001:
                uni = 0.00001
            overlaps[iix] = inters*1.0 / uni

        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap

def calcIOU(rect1, rect2):
    x1, y1, w1, h1 = rect1# (x,y): center points
    x2, y2, w2, h2 = rect2# (x,y): center points
    if (abs(x1 - x2) < (w1 + w2)/2.) and (abs(y1-y2) < (h1 + h2)/2.):
        left = max(x1 - w1 / 2., x2 - w2 / 2.)
        upper = max(y1 - h1 / 2., y2 - h2 / 2.)

        right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
        bottom = min(y1 + h1 / 2.0, y2 + h2 / 2.0)

        inter_w = abs(left - right)
        inter_h = abs(upper - bottom)
        inter_square = inter_w * inter_h
        union_square = (w1 * h1)+(w2 * h2)-inter_square

        calcIOU = inter_square* 1.0/union_square
        return calcIOU
    else:
        # print("No intersection!")
        return 0.

def voc_eval_rotated_rectangle(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    DEBUG = False
    VIS = False
    if DEBUG:
        # first load gt
        cachefile = 'annots.pkl'
        imagenames = [x.split('.')[0] for x in sorted(os.listdir('/mnt/v-chima/data/DAST1500_testset/image/')) if 'IMG_2099' in x]
        anno_names = [os.path.join('./txt_annotations', y) for y in sorted(os.listdir('./txt_annotations')) if 'IMG_2099' in y]
        assert(len(imagenames) == len(anno_names)), 'each image should correspond to one label file'
        im = cv2.imread(os.path.join('/mnt/v-chima/data/DAST1500_testset/image/', imagenames[0]+'.JPG'))
    else:
        # first load gt
        cachefile = 'annots.pkl'
        imagenames = [x.split('.')[0] for x in sorted(os.listdir('/mnt/v-chima/data/DAST1500_testset/image/')) if x.endswith('.JPG')]
        anno_names = [os.path.join('./txt_annotations', y) for y in sorted(os.listdir('./txt_annotations'))]
        assert(len(imagenames) == len(anno_names)), 'each image should correspond to one label file'

    if DEBUG:
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            assert imagename == anno_names[i].split(os.sep)[-1].split('.')[0]
            print(anno_names[i].strip())
            recs[imagename] = curve_parse_rec_txt(anno_names[i])
            boxes = recs[imagename][0]['bbox']
            minAreaRect = recs[imagename][0]['minAreaRect']
            im = cv2.line(im, (boxes[0],boxes[1]),(boxes[2],boxes[3]), (0,255,0),2)
            im = cv2.line(im, (boxes[2],boxes[3]),(boxes[4],boxes[5]), (0,255,0),2)
            im = cv2.line(im, (boxes[4],boxes[5]),(boxes[6],boxes[7]), (0,255,0),2)
            im = cv2.line(im, (boxes[6],boxes[7]),(boxes[0],boxes[1]), (0,255,0),2)
            im = cv2.line(im, (int(minAreaRect[0]-minAreaRect[2]/2.),int(minAreaRect[1]-minAreaRect[3]/2.)),(int(minAreaRect[0]+minAreaRect[2]/2.),int(minAreaRect[1]-minAreaRect[3]/2.)), (0,0,255),2)
            im = cv2.line(im, (int(minAreaRect[0]+minAreaRect[2]/2.),int(minAreaRect[1]-minAreaRect[3]/2.)),(int(minAreaRect[0]+minAreaRect[2]/2.),int(minAreaRect[1]+minAreaRect[3]/2.)), (0,0,255),2)
            im = cv2.line(im, (int(minAreaRect[0]+minAreaRect[2]/2.),int(minAreaRect[1]+minAreaRect[3]/2.)),(int(minAreaRect[0]-minAreaRect[2]/2.),int(minAreaRect[1]+minAreaRect[3]/2.)), (0,0,255),2)
            im = cv2.line(im, (int(minAreaRect[0]-minAreaRect[2]/2.),int(minAreaRect[1]+minAreaRect[3]/2.)),(int(minAreaRect[0]-minAreaRect[2]/2.),int(minAreaRect[1]-minAreaRect[3]/2.)), (0,0,255),2)
            cv2.putText(im, '{:.3f}'.format(minAreaRect[4]), (int(minAreaRect[0]-minAreaRect[2]/2.), int(np.maximum(int(minAreaRect[1]-minAreaRect[3]/2.)-3, 0))), 0, 0.3, (255,255,255), 1)
            # cv2.imwrite(os.path.join('./debug/', imagename+'.jpg'), im)
            # raise
    else:
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                assert imagename == anno_names[i].split(os.sep)[-1].split('.')[0]
                print(anno_names[i].strip())
                recs[imagename] = curve_parse_rec_txt(anno_names[i])
            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                # cPickle.dump(recs, f)
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                # recs = cPickle.load(f)
                recs = pickle.load(f)

    class_recs = {}
    npos = 0
    for ix, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # text
        # assert(R), 'Can not find any object in '+ classname+' class.'
        if not R:
            continue
        bbox = np.array([x['bbox'] for x in R])
        minAreaRect = np.array([x['minAreaRect'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                'minAreaRect': minAreaRect,
                                'difficult': difficult,
                                'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    if DEBUG:
        splitlines = [x.strip().split(' ') for x in lines if 'IMG_2099' in x]
    else:
        splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    # print(BB)
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    # BB = BB[sorted_ind, :]
    # image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd+1):
        if VIS == True:
            if d == 0:
                img = cv2.imread(os.path.join('/mnt/v-chima/data/DAST1500_testset/image/', image_ids[0]+'.JPG'))
            elif d == nd:
                cv2.imwrite(os.path.join('./debug/', image_ids[d-1]+'.jpg'), img)
                break
            elif image_ids[d] == image_ids[d-1]:
                pass
            else:
                cv2.imwrite(os.path.join('./debug/', image_ids[d-1]+'.jpg'), img)
                img = cv2.imread(os.path.join('/mnt/v-chima/data/DAST1500_testset/image/', image_ids[d]+'.JPG'))
        else:
            if d == nd:
                break

        R = class_recs[image_ids[d]] # all gt boxes in a single image
        bb = BB[d] # mask rcnn #one of the detected boxes in one image
        det_bbox = bb[:]#one of the detected boxes in one image
        # print(det_bbox)
        if DEBUG:
            im = cv2.line(im, (int(det_bbox[0]),int(det_bbox[1])),(int(det_bbox[2]),int(det_bbox[3])), (255,0,0),2)
            im = cv2.line(im, (int(det_bbox[2]),int(det_bbox[3])),(int(det_bbox[4]),int(det_bbox[5])), (255,0,0),2)
            im = cv2.line(im, (int(det_bbox[4]),int(det_bbox[5])),(int(det_bbox[6]),int(det_bbox[7])), (255,0,0),2)
            im = cv2.line(im, (int(det_bbox[6]),int(det_bbox[7])),(int(det_bbox[0]),int(det_bbox[1])), (255,0,0),2)
            # cv2.imwrite(os.path.join('./debug/', imagename+'.jpg'), im)
        if VIS:
            img = cv2.line(img, (int(det_bbox[0]),int(det_bbox[1])),(int(det_bbox[2]),int(det_bbox[3])), (0,0,0),2)
            img = cv2.line(img, (int(det_bbox[2]),int(det_bbox[3])),(int(det_bbox[4]),int(det_bbox[5])), (0,0,0),2)
            img = cv2.line(img, (int(det_bbox[4]),int(det_bbox[5])),(int(det_bbox[6]),int(det_bbox[7])), (0,0,0),2)
            img = cv2.line(img, (int(det_bbox[6]),int(det_bbox[7])),(int(det_bbox[0]),int(det_bbox[1])), (0,0,0),2)

        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        minAreaRectGT = R['minAreaRect'].astype(float)
        info_bbox_gt = BBGT[:, 0:8]
        info_minAreaRect_gt = minAreaRectGT[:, 0:5]
        overlaps = np.zeros(BBGT.shape[0])
        diff_angles = 45*np.ones(BBGT.shape[0])
        # print(len(BBGT[0]))
        if len(BBGT[0]) == 0:
            print('len(BBGT[0]) == 0')
            fp[d] = 1.
            color_line = (0,255,255) # false alarm
            if VIS:
                img = cv2.line(img, (int(det_bbox[0]),int(det_bbox[1])),(int(det_bbox[2]),int(det_bbox[3])), color_line,2)
                img = cv2.line(img, (int(det_bbox[2]),int(det_bbox[3])),(int(det_bbox[4]),int(det_bbox[5])), color_line,2)
                img = cv2.line(img, (int(det_bbox[4]),int(det_bbox[5])),(int(det_bbox[6]),int(det_bbox[7])), color_line,2)
                img = cv2.line(img, (int(det_bbox[6]),int(det_bbox[7])),(int(det_bbox[0]),int(det_bbox[1])), color_line,2)
            continue
        for iix in xrange(BBGT.shape[0]):
            gt_bbox = info_bbox_gt[iix, :]
            gt_minAreaRect = info_minAreaRect_gt[iix, :]
            # print(gt_minAreaRect)
            if VIS:
                if R['difficult'][iix] == 0: #gt
                    color_line = (0,255,0)
                else:
                    color_line = (0,0,255)
                img = cv2.line(img, (int(gt_bbox[0]),int(gt_bbox[1])),(int(gt_bbox[2]),int(gt_bbox[3])), color_line,2)
                img = cv2.line(img, (int(gt_bbox[2]),int(gt_bbox[3])),(int(gt_bbox[4]),int(gt_bbox[5])), color_line,2)
                img = cv2.line(img, (int(gt_bbox[4]),int(gt_bbox[5])),(int(gt_bbox[6]),int(gt_bbox[7])), color_line,2)
                img = cv2.line(img, (int(gt_bbox[6]),int(gt_bbox[7])),(int(gt_bbox[0]),int(gt_bbox[1])), color_line,2)

            theta = gt_minAreaRect[-1]
            sin = np.sin(theta*math.pi/180)
            cos = np.cos(theta*math.pi/180)

            # centered coord
            centered_det_coords = np.array(det_bbox).reshape(1,8).copy()
            centered_det_coords[:,0::2] = centered_det_coords[:,0::2]-gt_minAreaRect[0].reshape((-1,1))
            centered_det_coords[:,1::2] = centered_det_coords[:,1::2]-gt_minAreaRect[1].reshape((-1,1))
            #clockwisely rotate theta
            rotated_centered_det_coords = centered_det_coords.copy()
            rotated_centered_det_coords[:,0::2] = centered_det_coords[:,0::2] * cos - centered_det_coords[:,1::2] * sin
            rotated_centered_det_coords[:,1::2] = centered_det_coords[:,0::2] * sin + centered_det_coords[:,1::2] * cos
            #de-centered
            rotated_det_coords = rotated_centered_det_coords.copy()
            rotated_det_coords[:,0::2] = rotated_det_coords[:,0::2]+gt_minAreaRect[0].reshape((-1,1))
            rotated_det_coords[:,1::2] = rotated_det_coords[:,1::2]+gt_minAreaRect[1].reshape((-1,1))
            if DEBUG:
                im = cv2.line(im, (int(rotated_det_coords[0,0]),int(rotated_det_coords[0,1])),(int(rotated_det_coords[0,2]),int(rotated_det_coords[0,3])), (255,255,0),2)
                im = cv2.line(im, (int(rotated_det_coords[0,2]),int(rotated_det_coords[0,3])),(int(rotated_det_coords[0,4]),int(rotated_det_coords[0,5])), (255,255,0),2)
                im = cv2.line(im, (int(rotated_det_coords[0,4]),int(rotated_det_coords[0,5])),(int(rotated_det_coords[0,6]),int(rotated_det_coords[0,7])), (255,255,0),2)
                im = cv2.line(im, (int(rotated_det_coords[0,6]),int(rotated_det_coords[0,7])),(int(rotated_det_coords[0,0]),int(rotated_det_coords[0,1])), (255,255,0),2)

            det_minAreaRect = quadrangle2minAreaRect(rotated_det_coords.astype(np.int32).reshape(-1))
            assert abs(det_minAreaRect[-1]) <= 45
            if abs(det_minAreaRect[-1]) <= 22.5:
                diff_angles[iix] = abs(det_minAreaRect[-1])
            overlaps[iix] = calcIOU(det_minAreaRect[:4], gt_minAreaRect[:4])
            # print(overlaps[iix])

            if DEBUG:
                im = cv2.line(im, (int(det_minAreaRect[0]-det_minAreaRect[2]/2.),int(det_minAreaRect[1]-det_minAreaRect[3]/2.)),(int(det_minAreaRect[0]+det_minAreaRect[2]/2.),int(det_minAreaRect[1]-det_minAreaRect[3]/2.)), (0,255,255),2)
                im = cv2.line(im, (int(det_minAreaRect[0]+det_minAreaRect[2]/2.),int(det_minAreaRect[1]-det_minAreaRect[3]/2.)),(int(det_minAreaRect[0]+det_minAreaRect[2]/2.),int(det_minAreaRect[1]+det_minAreaRect[3]/2.)), (0,255,255),2)
                im = cv2.line(im, (int(det_minAreaRect[0]+det_minAreaRect[2]/2.),int(det_minAreaRect[1]+det_minAreaRect[3]/2.)),(int(det_minAreaRect[0]-det_minAreaRect[2]/2.),int(det_minAreaRect[1]+det_minAreaRect[3]/2.)), (0,255,255),2)
                im = cv2.line(im, (int(det_minAreaRect[0]-det_minAreaRect[2]/2.),int(det_minAreaRect[1]+det_minAreaRect[3]/2.)),(int(det_minAreaRect[0]-det_minAreaRect[2]/2.),int(det_minAreaRect[1]-det_minAreaRect[3]/2.)), (0,255,255),2)
                cv2.putText(im, '{:.3f}'.format(det_minAreaRect[4]), (int(det_minAreaRect[0]-det_minAreaRect[2]/2.), int(np.maximum(int(det_minAreaRect[1]-det_minAreaRect[3]/2.)-3, 0))), 0, 0.3, (255,255,255), 1)
                cv2.imwrite(os.path.join('./debug/', imagename+'.jpg'), im)

        angle_invalid_idx = np.where(diff_angles>22.5)[0]
        overlaps[angle_invalid_idx] = 0.
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                    color_line = (255,0,0) # det hit gt
                else:
                    fp[d] = 1.
                    color_line = (0,255,255) # false alarm
        else:
            fp[d] = 1.
            color_line = (0,255,255) # false alarm

        if VIS:
            img = cv2.line(img, (int(det_bbox[0]),int(det_bbox[1])),(int(det_bbox[2]),int(det_bbox[3])), color_line,2)
            img = cv2.line(img, (int(det_bbox[2]),int(det_bbox[3])),(int(det_bbox[4]),int(det_bbox[5])), color_line,2)
            img = cv2.line(img, (int(det_bbox[4]),int(det_bbox[5])),(int(det_bbox[6]),int(det_bbox[7])), color_line,2)
            img = cv2.line(img, (int(det_bbox[6]),int(det_bbox[7])),(int(det_bbox[0]),int(det_bbox[1])), color_line,2)

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap
