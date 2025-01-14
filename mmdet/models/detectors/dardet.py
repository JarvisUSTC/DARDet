from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
import numpy as np
import mmcv 
import os
from mmdet.core import rotated_box_to_poly, multiclass_nms_rotated_bbox
import cv2
from shapely.geometry import *
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from mmcv.image import imread, imwrite

def write_rotate_xml(output_floder,img_name,size,gsd,imagesource,gtbox_label,CLASSES):#size,gsd,imagesource#将检测结果表示为中科星图比赛格式的程序,这里用folder字段记录gsd
    ##添加写出为xml函数
    voc_headstr = """\
    <annotation>
        <folder>{}</folder>
        <filename>{}</filename>
        <path>{}</path>
        <source>
            <database>{}</database>
        </source>
        <size>
            <width>{}</width>
            <height>{}</height>
            <depth>{}</depth>
        </size>
        <segmented>0</segmented>
        """
    voc_rotate_objstr = """\
    <object>
        <name>{}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>{}</difficult>
        <robndbox>
            <cx>{}</cx>
            <cy>{}</cy>
            <w>{}</w>
            <h>{}</h>
            <angle>{}</angle>
        </robndbox>
        <extra>{:.2f}</extra>
    </object>
    """
    voc_tailstr = '''\
        </annotation>
        '''
    [floder,name]=os.path.split(img_name)
    # filename=name.replace('.jpg','.xml')
    filename=os.path.join(floder,os.path.splitext(name)[0]+'.xml')
    foldername=os.path.split(img_name)[0]
    head=voc_headstr.format(gsd,name,foldername,imagesource,size[1],size[0],size[2])
    rotate_xml_name=os.path.join(output_floder,os.path.split(filename)[1])
    f = open(rotate_xml_name, "w",encoding='utf-8')
    f.write(head)
    for i,box in enumerate (gtbox_label):
        obj=voc_rotate_objstr.format(CLASSES[int(box[6])],0,box[0],box[1],box[2],box[3],box[4],box[5])
        f.write(obj)
    f.write(voc_tailstr)
    f.close()

@DETECTORS.register_module()
class DARDet(SingleStageDetector):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DARDet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)

    def rbox2result(self, bboxes, labels, num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor | np.ndarray): shape (n, 5)
            labels (torch.Tensor | np.ndarray): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 9), dtype=np.float32) for i in range(num_classes)]#TODOinsert
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()#dets,rboxes[keep],scores_k[keep]
                labels = labels.detach().cpu().numpy()

            return [bboxes[labels == i, :] for i in range(num_classes)]
        
    def imshow_gpu_tensor(self, tensor):#调试中显示表标签图
        from PIL import Image
        from torchvision import transforms
        device=tensor[0].device
        mean= torch.tensor([123.675, 116.28, 103.53])
        std= torch.tensor([58.395, 57.12, 57.375])
        mean=mean.to(device)
        std=std.to(device)
        tensor = (tensor[0].squeeze() * std[:,None,None]) + mean[:,None,None]
        tensor=tensor[0:1]
        if len(tensor.shape)==4:
            image = tensor.permute(0,2, 3,1).cpu().clone().numpy()
        else:
            image = tensor.permute(1, 2,0).cpu().clone().numpy()
        image = image.astype(np.uint8).squeeze()
        image = transforms.ToPILImage()(image)
        image = image.resize((256, 256),Image.ANTIALIAS)
        # image.show(image)
        image.save('./img.jpg')
    
    def load_semantic_map_from_mask(self, gt_boxes, gt_masks, gt_labels,feature_shape):
        pad_shape=feature_shape[-2:]
        gt_areas=gt_masks.areas
        # heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        gt_sem_map = gt_boxes.new_zeros((self.bbox_head.num_classes, int(pad_shape[0] ), int(pad_shape[1] )))
        gt_sem_weights = gt_boxes.new_zeros((self.bbox_head.num_classes, int(pad_shape[0] ), int(pad_shape[1] )))
        box_masks=gt_masks.rescale(1/8).masks
        indexs = torch.sort(gt_areas)
        for ind in indexs[::-1]:
            box_mask=box_masks[ind]
            gt_sem_map[gt_labels[ind]][box_mask > 0] = 1
            gt_sem_weights[gt_labels[ind]][box_mask > 0] = np.min([1 / (gt_areas[ind]+0.000001),1])
        return gt_sem_map, gt_sem_weights
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None
                      ):
        self.debug = True
        if self.debug:
            img_name = 'debug_for_gt.jpg'
            img_debug = np.int8(np.copy(img[0].cpu().numpy()).transpose(1,2,0)*255).copy()
            gt_masks_debug = gt_masks[0].masks
            for mask_debug in gt_masks_debug:
                mask_points = mask_debug[0].reshape(-1,1,2).astype(np.int32)
                cv2.polylines(img=img_debug, pts=[mask_points], isClosed=True, color=(0,0,255), thickness=3)
            cv2.imwrite(img_name, img_debug)
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        # self.imshow_gpu_tensor(img)
    
        gt_mask_arrays=[]
        for i in gt_masks:
            gt_mask_array=np.array(i.masks).squeeze()
            gt_mask_array = gt_mask_array.astype(float)  # numpy强制类型转换
            gt_mask_array=gt_bboxes[0].new_tensor(gt_mask_array)
            gt_mask_arrays.append(gt_mask_array)
        gt_masks=gt_mask_arrays
        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore,gt_masks)
        return losses


    def rot_img(self,x, theta):
        # theta = torch.tensor(theta)theta
        device=x.device
        theta=x.new_tensor(theta* np.pi / (-180))#.to(device)#表示这里是反的顺时针旋转 是负号
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]]).to(x.device)
        rot_mat=rot_mat[None, ...]#.repeat(x.shape[0],1,1)[None, ...]
        out_size = torch.Size((1, x.size()[1],  x.size()[2],  x.size()[3]))
        grid = F.affine_grid(rot_mat, out_size)
        rotate = F.grid_sample(x, grid)
        # rotate = F.grid_sample(x.unsqueeze(0).unsqueeze(0), grid)
        return rotate#.squeeze()
    
    def rot_boxes(self,bbox_list, theta,center):
        
        theta=bbox_list[0][0].new_tensor(theta* np.pi / (-180))
        bboxes = bbox_list[0][0].detach()
        labels = bbox_list[0][1].detach()
        scores=bboxes[:,4:5]
        rboxes=bboxes[:,5:10]
        N = rboxes.shape[0]
        
        angle=scores.clone()
        angle[:]=theta
        x_ctr, y_ctr,  = rboxes.select(1, 0), rboxes.select( 1, 1)
        new_x_c,new_y_c=x_ctr - center[0], y_ctr - center[1]

        rects = torch.stack([new_x_c, new_y_c], dim=0).reshape(2, 1, N).permute(2, 0, 1)
        sin, cos = torch.sin(angle), torch.cos(angle)
        # M.shape=[N,2,2]
        M = torch.stack([cos, -sin, sin, cos],
                        dim=0).reshape(2, 2, N).permute(2, 0, 1)
        # polys:[N,8]
        polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
        rboxes[:,0] = polys[:,0]+center[0]
        rboxes[:,1] = polys[:,1]+center[1]
        rboxes[:,4]+=theta
        
        points = rotated_box_to_poly(rboxes)
    
        t4bx, t4by = points[:,0:8:2],points[:,1:8:2]
        t2xmin, _= torch.min(t4bx,1,keepdim=True)
        t2ymin, _= torch.min(t4by,1,keepdim=True)
        t2xmax, _= torch.max(t4bx,1,keepdim=True)
        t2ymax,_= torch.max(t4by,1,keepdim=True)
        r2bboxes=torch.cat((t2xmin,t2ymin,t2xmax,t2ymax),1)       
        return  torch.cat([r2bboxes, scores, rboxes, points], 1), labels
    
    def merge_boxes(self,bbox_list):
        cfg = self.test_cfg 
        labels_list,scores_list, rboxes_list = [],[],[]
        for i ,box in  enumerate (bbox_list):
            bboxes = box[0][0].detach()
            labels = box[0][1].detach()

            scores=bboxes[:,4]
            rboxes=bboxes[:,5:10]
            labels_list.append(labels)
            scores_list.append(scores)
            rboxes_list.append(rboxes)
        labels=torch.cat(labels_list)
        scores=torch.cat(scores_list)
        rboxes=torch.cat(rboxes_list)
        
        scores_pad=scores.new_zeros(scores.shape[0],  self.bbox_head.num_classes+1)
        scores_pad[torch.arange(0,labels.shape[0]),labels.long()]=scores
        if rboxes.shape[0]==0:
            return bbox_list[0][0][0],bbox_list[0][0][1]
        det_bboxes,  det_labels = multiclass_nms_rotated_bbox(rboxes,
                                                        scores_pad,  cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes,det_labels
    
    def simple_test(self, img, img_metas, rescale=False):
        cfg = self.test_cfg 
        rotate_test=cfg.get('rotate_test', False)
        if 'nms' not in cfg:
            with_nms = False
        else:
            with_nms = True
        if not rotate_test:
            x = self.extract_feat(img)
            outs = self.bbox_head(x)
            self.debug_vis_score_map(outs[0],img)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale, with_nms=with_nms)
        else:
            #旋转测试
            angles=cfg.get('rotate_test_angles', [0,90,180,270])
            bbox_list_all=[]
            for angle in angles:
                # self.imshow_gpu_tensor(img[0])
                rotated_img=self.rot_img(img, angle) #顺时针旋转90度
                # self.imshow_gpu_tensor(rotated_img[0])

                x = self.extract_feat(rotated_img)
                outs = self.bbox_head(x)
                bbox_list = self.bbox_head.get_bboxes(
                    *outs, img_metas, rescale=rescale)
                assert(len(bbox_list)==1)
                
                h, w =img.shape[-2],img.shape[-1]
                center = ((w - 1) * 0.5, (h - 1) * 0.5)
                if bbox_list[0][0].shape[0]>0:
                    bbox_list=[self.rot_boxes(bbox_list,angle,center)] #结果逆时针旋转90度
                bbox_list_all.append(bbox_list)
            bbox_list = [self.merge_boxes(bbox_list_all)]
        
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list
        bbox_results = [
            self.rbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes,  det_labels  in bbox_list     ]
        return bbox_results
    
    def debug_vis_score_map(self, score_map, img):
        images_cpu = img.cpu().numpy().transpose((0, 2, 3, 1))[0]
        target_vis = os.path.join("/datadisk/v-jiaweiwang/DARDet/", "debug")
        if not os.path.isdir(target_vis):
            os.makedirs(target_vis)
        with torch.no_grad():
            im_names = ["{}.jpg".format(np.random.rand()) for _ in range(len(score_map))]
            for im_idx in range(len(score_map)):
                im_name = im_names[im_idx]
                im_vis = images_cpu*np.array([[self.cfg.img_norm_cfg['std']]]) + np.array([[self.cfg.img_norm_cfg['mean']]])
                cv2.imwrite(os.path.join(target_vis, im_name), im_vis)
                im_vis = cv2.imread(os.path.join(target_vis, im_name))
                # score_heat = torch.sigmoid(score_map[im_idx][0,0:1,:,:])
                score_heat = score_map[im_idx][0,0:1,:,:]
                score_heat = F.interpolate(score_heat[None], scale_factor=self.bbox_head.strides[im_idx], mode="bilinear", align_corners=True)[0, 0, :im_vis.shape[0], :im_vis.shape[1]]
                score_heat = score_heat.cpu().numpy().squeeze()
                idx = np.where(score_heat > 0.5)
                im_vis[idx[0], idx[1], :] = im_vis[idx[0], idx[1], :] * 0.5 + np.array([[[0, 255, 0]]]) * 0.5
                cv2.imwrite(os.path.join(target_vis, im_name + ".heatmaps.jpg"), im_vis)


    # def simple_test(self, img, img_metas, rescale=False):
    #     """Test function without test time augmentation.

    #     Args:
    #         imgs (list[torch.Tensor]): List of multiple images
    #         img_metas (list[dict]): List of image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.

    #     Returns:
    #         list[list[np.ndarray]]: BBox results of each image and classes.
    #             The outer list corresponds to each image. The inner list
    #             corresponds to each class.
    #     """
    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #     bbox_list = self.bbox_head.get_bboxes(
    #         *outs, img_metas, rescale=rescale)
    #     # skip post-processing when exporting to ONNX
    #     if torch.onnx.is_in_onnx_export():
    #         return bbox_list
    #     bbox_results = [
    #         self.rbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes,  det_labels  in bbox_list
    #     ]
    #     return bbox_results

    def result2rotatexml(self,bboxes,labels,thr):
        if bboxes.shape[0]>0:
            index=bboxes[:,5]>thr
            bboxes=bboxes[index]
            labels=labels[index]

        rotateboxes=[]
        for i in range(bboxes.shape[0]):
            if(bboxes.size != 0):
                [cx, cy, w, h,angle, score]=bboxes[i,:]
                # angle -= np.pi/2
                rotateboxes.append([cx, cy, w,h,angle,score,labels[i],score])
                    # rotateboxes=np.vstack((cx, cy, w,h,Angle,result[:,7],result[:,4])).T
                    # rotateboxes.append([bbox[0],bbox[1],bbox[3],bbox[2],bbox[4]+np.pi/2,bbox[5],class_id])
        return np.array(rotateboxes)

    def box2rotatexml(self,bboxes,labels):
        index=bboxes[:,4]>0.3
        bboxes=bboxes[index]
        labels=labels[index]
        rotateboxes=[]
        for i in range(bboxes.shape[0]):
            if(bboxes.size != 0):
                [xmin, ymin, xmax, ymax, score, x1, y1, x2, y2,x3,y3,x4,y4]=bboxes[i,:]
                cx,cy = (x1 + x2+x3+x4)/4,(y1+y2+y3+y4)/4
                det=[x3-x1, y3-y1]
                h=np.linalg.norm(det)
                w=np.linalg.norm([x4-x2, y4-y2])
                if det[0]==0:
                    if det[1]>0:
                        Angle = np.pi/2
                    else:
                        Angle = -np.pi/2
                elif det[0]<0:
                    Angle = np.arctan(det[1]/det[0])+np.pi/2
                else:
                    Angle = np.arctan(det[1]/det[0])-np.pi/2
                rotateboxes.append([cx, cy, w,h,Angle,score,labels[i],score])
                    # rotateboxes=np.vstack((cx, cy, w,h,Angle,result[:,7],result[:,4])).T
                    # rotateboxes.append([bbox[0],bbox[1],bbox[3],bbox[2],bbox[4]+np.pi/2,bbox[5],class_id])
        return np.array(rotateboxes)

    def drow_points(self,img,points,labels,class_names=None,
                    score_thr=0.3,show=False,win_name='',thickness=1.0,font_scale=1.0,
                    wait_time=0,
                    out_file=None):
        if points.shape[0]>0:
            index=points[:,0]>score_thr
            points=points[index]
            for i in range(points.shape[0]):
                rbox=points[i][1:]    
                p_rotate=np.int32(rbox.reshape(-1,2))
                # cv2.circle(img, (int(rbox[0]), int(rbox[1])), 5, (0,255,0), -1)
                # cv2.circle(img, (int(rbox[2]), int(rbox[3])), 4, (0,0,255), -1)
                # cv2.circle(img, (int(rbox[4]), int(rbox[5])), 3, (255,0,0), -1)
                # cv2.circle(img, (int(rbox[6]), int(rbox[7])), 2, (255,0,255), -1)
                # p_rotate=np.int32(np.vstack((rbox[0:2],rbox[2:4],rbox[4:8],rbox[8:10])))  
                cv2.polylines(img,[np.array(p_rotate)],True,self.color_list[int(labels[i])+1],thickness)
                label_text = class_names[labels[i]] if class_names is not None else f'cls {labels[i]}'
                
                label_text += f'|{points[i][0] :.02f}'
                index=np.argmin(p_rotate[:,0])
                bbox_int=p_rotate[index]
                cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                            cv2.FONT_HERSHEY_COMPLEX, font_scale, self.color_list[int(labels[i])+1])

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)
        return img

    
    def imshow_det_rboxes(img,
                        bboxes,
                        labels,
                        class_names=None,
                        score_thr=0,
                        bbox_color='green',
                        text_color='green',
                        thickness=1,
                        font_scale=0.5,
                        show=True,
                        win_name='',
                        wait_time=0,
                        out_file=None):
        """Draw bboxes and class labels (with scores) on an image.

        Args:
            img (str or ndarray): The image to be displayed.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
                (n, 5).
            labels (ndarray): Labels of bboxes.
            class_names (list[str]): Names of each classes.
            score_thr (float): Minimum score of bboxes to be shown.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            show (bool): Whether to show the image.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
            out_file (str or None): The filename to write the image.

        Returns:
            ndarray: The image with bboxes drawn on it.
        """
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        # assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = imread(img)
        img = np.ascontiguousarray(img)

        if score_thr > 0:
            # assert bboxes.shape[1] == 6
            scores = bboxes[:, 0]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
 

        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            rbox=bboxes[i][1:]   
            p_rotate=np.int32(np.vstack(rbox[0:2],rbox[2:4],rbox[4:8],rbox[8:10],))                                                    #BRG
            cv2.polylines(img,[np.array(p_rotate)],True,color_val[int(label)],2)

            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, bbox_color, thickness=thickness)
            label_text = class_names[
                label] if class_names is not None else f'cls {label}'
            if len(bbox) > 4:
                label_text += f'|{bbox[-1]:.02f}'
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

        if show:
            imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)
        return img

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=2,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
            #这里开始让他一开始就显示rotatexml
        if bbox_result[0].shape[1]>5:
            # rotateboxes=self.box2rotatexml(bboxes,labels)
            rbox=np.hstack((bboxes[...,5:10],bboxes[...,4:5]))
            rotateboxes=self.result2rotatexml(rbox,labels,score_thr)
            # write_rotate_xml(os.path.dirname(out_file),out_file,[1024 ,1024,3],0.5,'0.5',rotateboxes.reshape((-1,8)),self.CLASSES)

        showboxs=np.hstack((bboxes[...,4:5],bboxes[...,10:]))
                    
        if out_file:
            file_dir=os.path.dirname(out_file)
            print(file_dir)
            if not os.path.exists(file_dir):
                os.mkdir(file_dir)
        img=self.drow_points(img,showboxs,labels,class_names=self.CLASSES,score_thr=score_thr,thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        #draw bounding boxes
        # showboxs=np.hstack((bboxes[...,4:5],bboxes[...,10:]))
        # self.imshow_det_rboxes(
        #     img,
        #     showboxs,
        #     labels,
        #     class_names=self.CLASSES,
        #     score_thr=score_thr,
        #     bbox_color=bbox_color,
        #     text_color=text_color,
        #     thickness=thickness,
        #     font_scale=font_scale,
        #     win_name=win_name,
        #     show=show,
        #     wait_time=wait_time,
        #     out_file=out_file)
        # imshow_det_bboxes(
        #     img,
        #     bboxes[:,0:5],
        #     labels,
        #     class_names=self.CLASSES,
        #     score_thr=score_thr,
        #     bbox_color=bbox_color,
        #     text_color=text_color,
        #     thickness=thickness,
        #     font_scale=font_scale,
        #     win_name=win_name,
        #     show=show,
        #     wait_time=wait_time,
        #     out_file=out_file)

        # if not (show or out_file):
        #     warnings.warn('show==False and out_file is not specified, only '
        #                     'result image will be returned')
        #     return img
    
    color_list= np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
    color_list = np.int32(color_list.reshape((-1, 3)) * 255)
    color_list=color_list.tolist()
    # colors = [(color_list[_]).astype(np.uint8) \
    #         for _ in range(len(color_list))]