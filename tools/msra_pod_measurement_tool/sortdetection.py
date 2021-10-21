import os
import numpy as np
import sys

anno_path = sys.argv[1]
score_thresh_list = [float(_) for _ in sys.argv[2:]] if len(sys.argv) >= 3 else [0.8]
files = os.listdir(anno_path)
files.sort()
outputstr = "detections_text"
for iscore in score_thresh_list:
    with open(outputstr+str(iscore)+'.txt', "w") as f1:
        for ix, filename in enumerate(files):
            if filename[-4:] != '.txt':
                continue
            imagename = filename[:-4]

            with open(os.path.join(anno_path, filename), "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip().split(",")

                box = line[:-1]
                score = float(line[-1])

                # box = line[:]
                # score = 1.0

                # box = line[:-2]
                # cls = int(line[-2])
                # score = float(line[-1])
                # if cls != 4:
                #     continue

                if score < iscore:
                    continue
                assert(len(box) %2 == 0) ,'mismatch xy'
                out_str = "{} {}".format(imagename, 0.999)
                for i in box:
                    out_str = out_str+' '+str(i)
                f1.writelines(out_str + '\n')

