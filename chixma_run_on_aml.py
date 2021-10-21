import argparse
import os
import json
import numpy as np
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--expName",
        dest="expName",
        default="",
        help="name of Exp",
        type=str,
    )
    parser.add_argument(
        "--expCode",
        dest="expCode",
        default="",
        help="name of Exp",
        type=str,
    )
    parser.add_argument(
        "--expVersion",
        dest="expVersion",
        default="",
        help="name of Exp",
        type=str,
    )
    parser.add_argument(
        "--node_nums",
        help="num of used nodes",
        type=int,
        required=True
    )
    parser.add_argument(
        "--gpus_per_node",
        help="num of gpus per node is used for multi nodes or gpus used if use single node",
        type=int,
        required=True
    )
    args, unparsed = parser.parse_known_args()
    extra_args = " ".join(unparsed)
    return args, extra_args


def main():
    args, extra_args = parse_args()
    init_on_aml()
    root_path = "/blob/workstation/mmdetection"
    #<========================Change Here==========================================
    output_dir = "{}/{}/{}/{}/output/".format(root_path, args.expName, args.expCode, args.expVersion)
    tmp_outdir = "./output/"
    #<========================Change Here==========================================
    config_file = "./configs/{}/{}.py".format(args.expName, args.expCode) #DARDet/exp1.py
    copy_data_from_blob(config_file)
    cfg = Config.fromfile(args.config)
    if args.node_nums > 1:
        assert args.node_nums == 1, "for now, only support single node"
    else:
        cmd = "local=$(pwd) \n export PYTHONPATH=${local}/mmdet \n"
        cmd += "export MKL_THREADING_LAYER=GNU \n"
        cmd += "mkdir -p {} \n".format(tmp_outdir)
        cmd += "cp {}/* {} \n".format(output_dir, tmp_outdir)
        cmd += "python -m torch.distributed.launch --nproc_per_node={} tools/train.py {}  --launcher pytorch --work-dir {}".format(args.gpus_per_node,config_file,tmp_outdir)
        print(cmd)
        os.system(cmd)
    # As it is too slow when writing logs into Azure Blob in realtime, we first log it in the dorcker image and copy it into Azure Blob finally.
    cmd = "mkdir -p {} \n".format(output_dir)
    cmd += "cp {}/* {} \n".format(tmp_outdir, output_dir)
    print(cmd)
    os.system(cmd)

    # # Make result file for MLT2017 RPN_ONLY test
    # cmd = "sudo mkdir /origin_results \n"
    # cmd += "sudo chmod -R 777 /origin_results \n"
    # cmd += "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    # cmd += "python /detectron/tools/demo_af_rpn.py --config-file {} --im_or_folder /detectron/datasets/icdar2015_mlt_test/JPEGImages/ --output /origin_results/ --confidence-threshold 0.5 --no_demo --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/rpn/model_final.pth MODEL.RPN_ONLY True DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/txt \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.txt' | sudo zip /res_txt.zip -@ \n"
    # #<========================Change Here===========================================
    # cmd += "sudo cp /res_txt.zip {}/{}/{}/{}/rpn/res_txt_50.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # # Change threshold
    # cmd += "sudo mkdir /result \n"
    # cmd += "sudo cp /blob/workstation/scripts/change_threshold.py /change_threshold.py \n"
    # cmd += "cd /result \n"
    # for th in range(500, 975, 25):
    #     th = th / 1000.0
    #     cmd += "python /change_threshold.py /origin_results/txt {} \n".format(th)
    #     cmd += "sudo zip -r /test_scratch_{:.3f}.zip ./ \n".format(th)
    #     cmd += "sudo cp /test_scratch_{:.3f}.zip {}/{}/{}/{}/rpn/test_scratch_{:.3f}.zip \n".format(th, root_path, args.expName, args.expCode, args.expVersion, th)
    # print(cmd)
    # os.system(cmd)

    # # Make result file for MLT2017 FRCN test
    # cmd = "sudo mkdir /origin_results \n"
    # cmd += "sudo chmod -R 777 /origin_results \n"
    # cmd += "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    # cmd += "python /detectron/tools/demo_af_rpn.py --config-file {} --im_or_folder /detectron/datasets/icdar2015_mlt_test/JPEGImages/ --output /origin_results/ --confidence-threshold 0.5 --no_demo --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/txt \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.txt' | sudo zip /res_txt.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_txt.zip {}/{}/{}/{}/frcn/res_txt_50.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # # Change threshold
    # cmd += "sudo mkdir /result \n"
    # cmd += "sudo cp /blob/workstation/scripts/change_threshold.py /change_threshold.py \n"
    # cmd += "cd /result \n"
    # for th in range(500, 975, 25):
    #     th = th / 1000.0
    #     cmd += "python /change_threshold.py /origin_results/txt {} \n".format(th)
    #     cmd += "sudo zip -r /test_scratch_{:.3f}.zip ./ \n".format(th)
    #     cmd += "sudo cp /test_scratch_{:.3f}.zip {}/{}/{}/{}/frcn/test_scratch_{:.3f}.zip \n".format(th, root_path, args.expName, args.expCode, args.expVersion, th)
    # print(cmd)
    # os.system(cmd)

    # Make result file for MSRA_POD FRCN test
    cmd = "sudo mkdir /origin_results \n"
    cmd += "sudo chmod -R 777 /origin_results \n"
    cmd += "local=$(pwd) \n export PYTHONPATH=${local}/mmdet \n"
    cmd += "rm -r /mmdetection/tools/msra_pod_measurement_tool/test_dataset \n"
    cmd += "mkdir -p /mmdetection/tools/msra_pod_measurement_tool/test_dataset \n"
    cmd += "ln -s /mmdetection/datasets/POD_RevB_combined/raw_images_horizontal /mmdetection/tools/msra_pod_measurement_tool/test_dataset/images \n"
    cmd += "ln -s /mmdetection/datasets/POD_RevB_combined/xml /mmdetection/tools/msra_pod_measurement_tool/test_dataset/xml \n"
    cmd += "python /mmdetection/tools/demo_pod.py --config-file {} --im_or_folder /mmdetection/datasets/POD_RevB_combined/raw_images_horizontal/ --output /origin_results/ --confidence-threshold 0.1 --do_eval --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/box_refinement/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    cmd += "cd /origin_results/txt \n"    
    cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.txt' | sudo zip -q /res_txt.zip -@ \n"
    #<========================Change Here============================================
    cmd += "sudo cp /res_txt.zip {}/{}/{}/{}/box_refinement/res_txt_th0.5_horizontal.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    cmd += "cd /origin_results/image \n"
    cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.jpg' | sudo zip -q /res_ims.zip -@ \n"
    #<========================Change Here============================================
    cmd += "sudo cp /res_ims.zip {}/{}/{}/{}/box_refinement/res_ims_th0.5_horizontal.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /detectron \n"
    # cmd += "python /detectron/tools/demo_pod.py --config-file {} --im_or_folder /blob/data/kownlege_lake_testset/test_images/ --output /origin_results2/ --confidence-threshold 0.1 --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    #<========================Change Here============================================
    # cmd += "cd /origin_results2/image \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.jpg' | sudo zip -q /res_ims_kl.zip -@ \n"
    # cmd += "sudo cp /res_ims_kl.zip {}/{}/{}/{}/frcn/res_ims_kl_th0.5.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    print(cmd)
    os.system(cmd)

    # Make result file for Rotated_360_MSRA_POD FRCN test
    cmd = "sudo mkdir /origin_results \n"
    cmd += "sudo chmod -R 777 /origin_results \n"
    cmd += "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    cmd += "rm -r /detectron/tools/msra_pod_measurement_tool/test_dataset \n"
    cmd += "rm -rf /detectron/tools/msra_pod_measurement_tool/annots.pkl \n"
    cmd += "mkdir -p /detectron/tools/msra_pod_measurement_tool/test_dataset \n"
    cmd += "ln -s /detectron/datasets/rotated_360_POD_RevB_combined/raw_images_horizontal /detectron/tools/msra_pod_measurement_tool/test_dataset/images \n"
    cmd += "ln -s /detectron/datasets/rotated_360_POD_RevB_combined/xml /detectron/tools/msra_pod_measurement_tool/test_dataset/xml \n"
    cmd += "python /detectron/tools/demo_pod.py --config-file {} --im_or_folder /detectron/datasets/rotated_360_POD_RevB_combined/raw_images_horizontal/ --output /origin_results/ --confidence-threshold 0.1 --do_eval --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/box_refinement/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    cmd += "cd /origin_results/txt \n"    
    cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.txt' | sudo zip -q /res_txt.zip -@ \n"
    #<========================Change Here============================================
    cmd += "sudo cp /res_txt.zip {}/{}/{}/{}/box_refinement/res_txt_th0.5_360.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    cmd += "cd /origin_results/image \n"
    cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.jpg' | sudo zip -q /res_ims.zip -@ \n"
    #<========================Change Here============================================
    cmd += "sudo cp /res_ims.zip {}/{}/{}/{}/box_refinement/res_ims_th0.5_360.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /detectron \n"
    # cmd += "python /detectron/tools/demo_pod.py --config-file {} --im_or_folder /blob/data/kownlege_lake_testset/test_images/ --output /origin_results2/ --confidence-threshold 0.1 --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # #<========================Change Here============================================
    # cmd += "cd /origin_results2/image \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.jpg' | sudo zip -q /res_ims_kl.zip -@ \n"
    # cmd += "sudo cp /res_ims_kl.zip {}/{}/{}/{}/frcn/res_ims_kl_th0.5.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    print(cmd)
    os.system(cmd)

    # Make result file for Rotated_45_MSRA_POD FRCN test
    cmd = "sudo mkdir /origin_results \n"
    cmd += "sudo chmod -R 777 /origin_results \n"
    cmd += "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    cmd += "rm -r /detectron/tools/msra_pod_measurement_tool/test_dataset \n"
    cmd += "rm -rf /detectron/tools/msra_pod_measurement_tool/annots.pkl \n"
    cmd += "mkdir -p /detectron/tools/msra_pod_measurement_tool/test_dataset \n"
    cmd += "ln -s /detectron/datasets/rotated_45_POD_RevB_combined/raw_images_horizontal /detectron/tools/msra_pod_measurement_tool/test_dataset/images \n"
    cmd += "ln -s /detectron/datasets/rotated_45_POD_RevB_combined/xml /detectron/tools/msra_pod_measurement_tool/test_dataset/xml \n"
    cmd += "python /detectron/tools/demo_pod.py --config-file {} --im_or_folder /detectron/datasets/rotated_45_POD_RevB_combined/raw_images_horizontal/ --output /origin_results/ --confidence-threshold 0.1 --do_eval --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/box_refinement/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    cmd += "cd /origin_results/txt \n"    
    cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.txt' | sudo zip -q /res_txt.zip -@ \n"
    #<========================Change Here============================================
    cmd += "sudo cp /res_txt.zip {}/{}/{}/{}/box_refinement/res_txt_th0.5_45.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    cmd += "cd /origin_results/image \n"
    cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.jpg' | sudo zip -q /res_ims.zip -@ \n"
    #<========================Change Here============================================
    cmd += "sudo cp /res_ims.zip {}/{}/{}/{}/box_refinement/res_ims_th0.5_45.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /detectron \n"
    # cmd += "python /detectron/tools/demo_pod.py --config-file {} --im_or_folder /blob/data/kownlege_lake_testset/test_images/ --output /origin_results2/ --confidence-threshold 0.1 --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # #<========================Change Here============================================
    # cmd += "cd /origin_results2/image \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.jpg' | sudo zip -q /res_ims_kl.zip -@ \n"
    # cmd += "sudo cp /res_ims_kl.zip {}/{}/{}/{}/frcn/res_ims_kl_th0.5.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    print(cmd)
    os.system(cmd)

    # # Make result file for cTDaR2019_TRACKA FRCN test
    # cmd = "sudo mkdir /origin_results \n"
    # cmd += "sudo chmod -R 777 /origin_results \n"
    # cmd += "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    # cmd += "rm -r /detectron/tools/ctdar_measurement_tool/annotations/cTDaR2019/trackA \n"
    # cmd += "ln -s /detectron/datasets/cTDaR2019_TRACKA/testing_data/raw_xml /detectron/tools/ctdar_measurement_tool/annotations/cTDaR2019/trackA \n"
    # cmd += "python /detectron/tools/demo_table_detection.py --config-file {} --im_or_folder /detectron/datasets/cTDaR2019_TRACKA/testing_data/converted_jpg_ims/ --output /origin_results/ --confidence-threshold 0.5 --no_demo --do_eval --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/txt \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.txt' | sudo zip -q /res_txt.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_txt.zip {}/{}/{}/{}/frcn/res_txt_th0.5.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # print(cmd)
    # os.system(cmd)

    # # Make result file for publaynet_val FRCN test
    # cmd = "sudo mkdir /origin_results \n"
    # cmd += "sudo chmod -R 777 /origin_results \n"
    # cmd += "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    # cmd += "rm -r /detectron/tools/ctdar_measurement_tool/annotations/publaynet_val/trackA \n"
    # cmd += "ln -s /detectron/datasets/publaynet_val/val_gt_xml_for_table_only /detectron/tools/ctdar_measurement_tool/annotations/publaynet_val/trackA \n"
    # cmd += "rm -r /detectron/tools/coco_eval_tool/publaynet_val.json \n"
    # cmd += "ln -s /detectron/datasets/publaynet_val/val.json /detectron/tools/coco_eval_tool/publaynet_val.json \n"
    # cmd += "python /detectron/tools/demo_table_detection.py --config-file {} --im_or_folder /detectron/datasets/publaynet_val/val/ --output /origin_results/ --confidence-threshold 0.5 --no_demo --do_eval --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/txt \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.txt' | sudo zip -q /res_txt.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_txt.zip {}/{}/{}/{}/frcn/res_txt_th0.5.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # print(cmd)
    # os.system(cmd)

    # # Make result file for IIIT-AR-13K_val/test FRCN test
    # cmd = "sudo mkdir /origin_results \n"
    # cmd += "sudo chmod -R 777 /origin_results \n"
    # cmd += "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    # cmd += "rm -r /detectron/tools/IIIT-AR-13K_evaluation_tool/input/val/ground-truth \n"
    # cmd += "ln -s /blob/data/IIIT-AR-13K_val/gt_txt_for_eval_tool /detectron/tools/IIIT-AR-13K_evaluation_tool/input/val/ground-truth \n"
    # cmd += "rm -r /detectron/tools/IIIT-AR-13K_evaluation_tool/input/test/ground-truth \n"
    # cmd += "ln -s /blob/data/IIIT-AR-13K_test/gt_txt_for_eval_tool /detectron/tools/IIIT-AR-13K_evaluation_tool/input/test/ground-truth \n"
    # cmd += "python /detectron/tools/demo_table_detection.py --config-file {} --im_or_folder /detectron/datasets/IIIT-AR-13K_val/validation_images --output /origin_results/ --confidence-threshold 0.5 --do_eval --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/txt \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.txt' | sudo zip -q /res_txt.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_txt.zip {}/{}/{}/{}/frcn/val_res_txt_th0.5.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/image \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.jpg' | sudo zip -q /res_ims.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_ims.zip {}/{}/{}/{}/frcn/val_res_ims.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /detectron \n"
    # cmd += "sudo mkdir /origin_results2 \n"
    # cmd += "sudo chmod -R 777 /origin_results2 \n"
    # cmd += "python /detectron/tools/demo_table_detection.py --config-file {} --im_or_folder /detectron/datasets/IIIT-AR-13K_test/test_images --output /origin_results2/ --confidence-threshold 0.5 --do_eval --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results2/txt \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.txt' | sudo zip -q /res_txt2.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_txt2.zip {}/{}/{}/{}/frcn/test_res_txt_th0.5.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results2/image \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.jpg' | sudo zip -q /res_ims2.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_ims2.zip {}/{}/{}/{}/frcn/test_res_ims.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # print(cmd)
    # os.system(cmd)

    # # Make result file for TableBank val/test FRCN test
    # cmd = "sudo mkdir /origin_results \n"
    # cmd += "sudo chmod -R 777 /origin_results \n"
    # cmd += "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    # cmd += "rm -r /detectron/tools/coco_eval_tool/tablebank_word_val.json \n"
    # cmd += "rm -r /detectron/tools/coco_eval_tool/tablebank_word_test.json \n"
    # cmd += "rm -r /detectron/tools/coco_eval_tool/tablebank_latex_val.json \n"
    # cmd += "rm -r /detectron/tools/coco_eval_tool/tablebank_latex_test.json \n"
    # cmd += "rm -r /detectron/tools/coco_eval_tool/tablebank_word_latex_val.json \n"
    # cmd += "rm -r /detectron/tools/coco_eval_tool/tablebank_word_latex_test.json \n"
    # cmd += "ln -s /blob/data/TableBank/annotations/tablebank_word_val.json /detectron/tools/coco_eval_tool/tablebank_word_val.json \n"
    # cmd += "ln -s /blob/data/TableBank/annotations/tablebank_word_test.json /detectron/tools/coco_eval_tool/tablebank_word_test.json \n"
    # cmd += "ln -s /blob/data/TableBank/annotations/tablebank_latex_val.json /detectron/tools/coco_eval_tool/tablebank_latex_val.json \n"
    # cmd += "ln -s /blob/data/TableBank/annotations/tablebank_latex_test.json /detectron/tools/coco_eval_tool/tablebank_latex_test.json \n"
    # cmd += "ln -s /blob/data/TableBank/annotations/tablebank_word_latex_val.json /detectron/tools/coco_eval_tool/tablebank_word_latex_val.json \n"
    # cmd += "ln -s /blob/data/TableBank/annotations/tablebank_word_latex_test.json /detectron/tools/coco_eval_tool/tablebank_word_latex_test.json \n"
    # cmd += "python /detectron/tools/demo_table_detection.py --config-file {} --im_or_folder /detectron/datasets/TableBank/images/ --name_list /blob/data/TableBank/namelists/tablebank_word_val.txt --output /origin_results/tablebank_word_val/ --confidence-threshold 0.5 --no_demo --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "python /detectron/tools/demo_table_detection.py --config-file {} --im_or_folder /detectron/datasets/TableBank/images/ --name_list /blob/data/TableBank/namelists/tablebank_word_test.txt --output /origin_results/tablebank_word_test/ --confidence-threshold 0.5 --no_demo --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "python /detectron/tools/demo_table_detection.py --config-file {} --im_or_folder /detectron/datasets/TableBank/images/ --name_list /blob/data/TableBank/namelists/tablebank_latex_val.txt --output /origin_results/tablebank_latex_val/ --confidence-threshold 0.5 --no_demo --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "python /detectron/tools/demo_table_detection.py --config-file {} --im_or_folder /detectron/datasets/TableBank/images/ --name_list /blob/data/TableBank/namelists/tablebank_latex_test.txt --output /origin_results/tablebank_latex_test/ --confidence-threshold 0.5 --no_demo --num_loader 64 --opts MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.RPN_ONLY False DEBUG False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/ \n"
    # cmd += "sudo zip -q -r /results.zip ./ \n"
    # cmd += "sudo cp /results.zip {}/{}/{}/{}/frcn/results.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /detectron/tools/coco_eval_tool/ \n"
    # cmd += "python ./evaluate_tablebank.py -tablebank_word_val /origin_results/tablebank_word_val/txt/ \n"
    # cmd += "echo tablebank_word_val \n"
    # cmd += "python ./evaluate_tablebank.py -tablebank_latex_val /origin_results/tablebank_latex_val/txt/ \n"
    # cmd += "echo tablebank_latex_val \n"
    # cmd += "python ./evaluate_tablebank.py -tablebank_word_latex_val /origin_results/tablebank_word_val/txt/ /origin_results/tablebank_latex_val/txt/ \n"
    # cmd += "echo tablebank_word_latex_val \n"
    # cmd += "python ./evaluate_tablebank.py -tablebank_word_test /origin_results/tablebank_word_test/txt/ \n"
    # cmd += "echo tablebank_word_test \n"
    # cmd += "python ./evaluate_tablebank.py -tablebank_latex_test /origin_results/tablebank_latex_test/txt/ \n"
    # cmd += "echo tablebank_latex_test \n"
    # cmd += "python ./evaluate_tablebank.py -tablebank_word_latex_test /origin_results/tablebank_word_test/txt/ /origin_results/tablebank_latex_test/txt/ \n"
    # cmd += "echo tablebank_word_latex_test \n"
    # print(cmd)
    # os.system(cmd)

    # # Make result file for MSRA_TSR test
    # cmd = "sudo mkdir /origin_results \n"
    # cmd += "sudo chmod -R 777 /origin_results \n"
    # cmd += "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    # cmd += "rm -r /detectron/tools/ctdar_measurement_tool/annotations/MSRA_TSR/trackB1 \n"
    # cmd += "rm -r /detectron/tools/ctdar_measurement_tool/annotations/MSRA_TSR/cropped_textbox_xml_for_eval_tool \n"
    # cmd += "ln -s /blob/data/MSRA_TSR_test/cropped_gt_xml_for_eval_tool /detectron/tools/ctdar_measurement_tool/annotations/MSRA_TSR/trackB1 \n"
    # cmd += "ln -s /blob/data/MSRA_TSR_test/cropped_textbox_xml_for_eval_tool /detectron/tools/ctdar_measurement_tool/annotations/MSRA_TSR/cropped_textbox_xml_for_eval_tool \n"
    # cmd += "python /detectron/tools/demo_TSR.py --config-file {} --im_or_folder /blob/data/MSRA_TSR_test/cropped_table_image/ --output /origin_results/ --no_demo --do_eval --num_loader 64 --opts MODEL.DEVICE cuda MODEL.WEIGHTS {}/{}/{}/{}/rpn/model_final.pth MODEL.MERGE_HEAD_ON False DEBUG False MODEL.SPLIT_HEAD.USE_CURVE_FITTING.ENABLED False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "python /detectron/tools/demo_TSR.py --config-file {} --im_or_folder /blob/data/MSRA_TSR_test/cropped_table_image/ --output /origin_results/ --do_eval --num_loader 64 --opts MODEL.DEVICE cuda MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.MERGE_HEAD_ON True DEBUG False MODEL.SPLIT_HEAD.USE_CURVE_FITTING.ENABLED False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/txt \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.xml' | sudo zip -q /res_xml.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_xml.zip {}/{}/{}/{}/frcn/res_xml.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/image \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.jpg' | sudo zip -q /res_ims.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_ims.zip {}/{}/{}/{}/frcn/res_ims.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # print(cmd)
    # os.system(cmd)

    # # Make result file for cTDaR2019_TSR test
    # cmd = "sudo mkdir /origin_results \n"
    # cmd += "sudo chmod -R 777 /origin_results \n"
    # cmd += "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    # cmd += "rm -r /detectron/tools/ctdar_measurement_tool/annotations/cTDaR2019/trackB1 \n"
    # cmd += "rm -r /detectron/tools/ctdar_measurement_tool/annotations/cTDaR2019/cropped_textbox_xml_for_eval_tool \n"
    # cmd += "ln -s /detectron/datasets/cTDaR2019_TSR_test/cropped_gt_xml_for_eval_tool /detectron/tools/ctdar_measurement_tool/annotations/cTDaR2019/trackB1 \n"
    # cmd += "ln -s /detectron/datasets/cTDaR2019_TSR_test/cropped_textbox_xml_for_eval_tool /detectron/tools/ctdar_measurement_tool/annotations/cTDaR2019/cropped_textbox_xml_for_eval_tool \n"
    # cmd += "python /detectron/tools/demo_TSR.py --config-file {} --im_or_folder /detectron/datasets/cTDaR2019_TSR_test/cropped_table_image/ --output /origin_results/ --no_demo --do_eval --num_loader 64 --opts MODEL.DEVICE cuda MODEL.WEIGHTS {}/{}/{}/{}/rpn/model_final.pth MODEL.MERGE_HEAD_ON False DEBUG False MODEL.SPLIT_HEAD.USE_CURVE_FITTING.ENABLED False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "python /detectron/tools/demo_TSR.py --config-file {} --im_or_folder /detectron/datasets/cTDaR2019_TSR_test/cropped_table_image/ --output /origin_results/ --do_eval --num_loader 64 --opts MODEL.DEVICE cuda MODEL.WEIGHTS {}/{}/{}/{}/frcn/model_final.pth MODEL.MERGE_HEAD_ON True DEBUG False MODEL.SPLIT_HEAD.USE_CURVE_FITTING.ENABLED False \n".format(config_file, root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/txt \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.xml' | sudo zip -q /res_xml.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_xml.zip {}/{}/{}/{}/frcn/res_xml.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # cmd += "cd /origin_results/image \n"
    # cmd += "sudo find ./ -mindepth 1 -maxdepth 1 -name '*.jpg' | sudo zip -q /res_ims.zip -@ \n"
    # #<========================Change Here============================================
    # cmd += "sudo cp /res_ims.zip {}/{}/{}/{}/frcn/res_ims.zip \n".format(root_path, args.expName, args.expCode, args.expVersion)
    # print(cmd)
    # os.system(cmd)

def init_on_aml():
    cmd = "bash chixma_init_on_aml.sh"
    print(cmd)
    os.system(cmd)


def copy_data_from_blob(config_file):
    cmd = "local=$(pwd) \n export PYTHONPATH=${local}/detectron2 \n"
    cmd += "echo $PYTHONPATH \n"
    cmd += "python copy_datasets_from_blob.py --config-file {}".format(config_file)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    main()
