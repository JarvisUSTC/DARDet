import os
import sys

set_base_path = "/mnt/weihlin/POD/data/RevB_split"

set_list = ["business-documents-photo", "business-documents-scan", \
        "technical-publications-photo", "technical-publications-scan"]

set_image_list = {}

for folder in set_list:
    file_path = os.path.join(set_base_path, folder, "images")
    files = os.listdir(file_path)
    files.sort()
    print(folder, len(files))
    set_image_list[folder] = {}
    for img_file in files:
        img_name = img_file[:-4]
        set_image_list[folder][img_name] = True

f = open("detections_text0.8.txt", "r")
os.makedirs("split_result")
lines = f.readlines()
for folder in set_list:
    output_file = os.path.join("split_result", folder + ".txt")
    fout = open(output_file, "w")
    for line in lines:
        line_clean = line.strip()
        p = line_clean.split(" ")
        img_name = p[0]
        if img_name in set_image_list[folder]:
            fout.write(line_clean + "\n")
    fout.close()

