from mmcv import Config
import argparse
import os
import time
import copy

DATASETS = {
    "POD_RevA": ("POD_RevA/JPEGImages", "POD_RevA/train.json"),
    "POD_RevB_combined": ("POD_RevB_combined/raw_images_horizontal", ""),
    "rotated_360_POD_RevB_combined": ("rotated_360_POD_RevB_combined/raw_images_horizontal",""),
    "rotated_45_POD_RevB_combined": ("rotated_45_POD_RevB_combined/raw_images_horizontal",""),
    "cTDaR2019_TRACKA": ("cTDaR2019_TRACKA/JPEGImages", "cTDaR2019_TRACKA/train.json"),
    "POD2017": ("POD2017/JPEGImages", "POD2017/train.json"),
    "publaynet_train": ("publaynet_train/train", "publaynet_train/train.json"),
    "publaynet_val": ("publaynet_val/val", "publaynet_val/val.json"),
    "IIIT-AR-13K_train": ("IIIT-AR-13K_train/training_images", "IIIT-AR-13K_train/train.json"),
    "IIIT-AR-13K_val": ("IIIT-AR-13K_val/validation_images", "IIIT-AR-13K_val/val.json"),
    "IIIT-AR-13K_test": ("IIIT-AR-13K_test/test_images", "IIIT-AR-13K_test/test.json"),
    "TableBank_Word_Train": ("TableBank/images", "TableBank/annotations/tablebank_word_train.json"),
    "TableBank_Latex_Train": ("TableBank/images", "TableBank/annotations/tablebank_latex_train.json"),
    "knowledge_lake_pod_part1": ("knowledge_lake_pod_part1/JPEGImages", "knowledge_lake_pod_part1/train.json"),
    "knowledge_lake_pod_part2": ("knowledge_lake_pod_part2/JPEGImages", "knowledge_lake_pod_part2/train.json"),
    "financial_report_pod": ("financial_report_pod/JPEGImages", "financial_report_pod/train.json"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    return args


def main(config_file):
    cfg = Config.fromfile(config_file)
    TRAIN_DATASETS = cfg.train_dataset
    TEST_DATASETS = cfg.test_dataset
    DATASETS_USED = TRAIN_DATASETS + TEST_DATASETS
    for dataset_name in DATASETS_USED:
        dataset_im_dir = DATASETS[dataset_name][0]
        dataset_dir = dataset_im_dir.split('/')[-2]
        dataset_path = os.path.join('./datasets', dataset_dir)
        if not os.path.exists(dataset_path):
            start_time = time.time()
            cmd = "cd ./datasets \n"
            if dataset_name == "SynthText":
                # Too many images to unzip, so we do not copy it
                cmd += "sudo ln -s /blob/data/SynthText ./SynthText \n"
                cmd += "cd /mmdetection \n"
                print(cmd)
                os.system(cmd)
            elif dataset_name == "SCUT_CTW1500":
                cmd += "mkdir SCUT_CTW1500 \n"
                cmd += "cp /blob/data/{}.zip ./SCUT_CTW1500/{}.zip \n".format(dataset_dir, dataset_dir)
                cmd += "cd ./SCUT_CTW1500 \n"
                cmd += "sudo unzip -d ./{}/ ./{}.zip \n".format(dataset_dir, dataset_dir)
                cmd += "sudo chmod -R 777 ./{}/ \n".format(dataset_dir)
                cmd += "cd /mmdetection \n"
                print(cmd)
                os.system(cmd)
                print("copy and unzip {}.zip time {} s".format(dataset_dir, time.time() - start_time))
            else:
                if os.path.isfile("/blob/data/{}.zip".format(dataset_dir)):
                    cmd += "cp /blob/data/{}.zip ./{}.zip \n".format(dataset_dir, dataset_dir)
                    cmd += "sudo unzip -d ./{}/ ./{}.zip \n".format(dataset_dir, dataset_dir)
                elif os.path.isfile("/blob/data/{}.tar".format(dataset_dir)):
                    cmd += "cp /blob/data/{}.tar ./{}.tar \n".format(dataset_dir, dataset_dir)
                    cmd += "sudo tar xvf ./{}.tar \n".format(dataset_dir)
                # if dataset_name == "POD_RevA":
                #     cmd += "cp /blob/data/{}_solid_line_pkl.zip ./{}_solid_line_pkl.zip \n".format(dataset_dir, dataset_dir)
                #     cmd += "sudo unzip -d ./{}/solid_line_pkl/ ./{}_solid_line_pkl.zip \n".format(dataset_dir, dataset_dir)
                cmd += "sudo chmod -R 777 ./{}/ \n".format(dataset_dir)
                cmd += "cd /mmdetection \n"
                print(cmd)
                os.system(cmd)
                print("copy and unzip {}.zip time {} s".format(dataset_dir, time.time() - start_time))

if __name__ == '__main__':
    args = parse_args()
    main(args.config_file)
