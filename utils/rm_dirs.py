import os
import argparse
import numpy as np
import math
from tqdm import tqdm
import shutil
import cv2
import tarfile
import pickle

parser = argparse.ArgumentParser(description='run scripts.')

parser.add_argument("--folders",nargs="+", type=str)
args = parser.parse_args()
folders = args.folders
for folder in folders:
    video_path = "../data/"+folder+"/"
    video_data_pairs = {}
    All_path = []
    for path in os.listdir(video_path):
        tmp_path = os.path.join(video_path, path)
        for path1 in os.listdir(tmp_path):
            tmp_path1 = os.path.join(tmp_path, path1)
            for path2 in os.listdir(tmp_path1):
                tmp_path2 = os.path.join(tmp_path1, path2)
                for path3 in os.listdir(tmp_path2):
                    tmp_path3 = os.path.join(tmp_path2, path3)
                    for path4 in os.listdir(tmp_path3):
                        tmp_path4 = os.path.join(tmp_path3, path4)
                        for path5 in os.listdir(tmp_path4):
                            tmp_path5 = os.path.join(tmp_path4, path5)
                            label = path3 + "_" + path4 + "_" + path5
                            for path6 in os.listdir(tmp_path5):
                                tmp_path6 = os.path.join(tmp_path5, path6)
                                for path7 in os.listdir(tmp_path6):
                                    target = os.path.join(tmp_path6, path7)
                                    output = "result/"+target[len("../data/"):-4]
                                    All_path.append([target, output])
    irregular = {}
    less_csv = []
    diff_csv = []
    output_path = "AI/"+folder+"/"
    csv_path = []
    tar_path = []
    redundancies = []
    for path in os.listdir(output_path):
        tmp_path = os.path.join(output_path, path)
        for path1 in os.listdir(tmp_path):
            tmp_path1 = os.path.join(tmp_path, path1)
            for path2 in os.listdir(tmp_path1):
                print(path2)
                tmp_path2 = os.path.join(tmp_path1, path2)
                for path3 in os.listdir(tmp_path2):
                    tmp_path3 = os.path.join(tmp_path2, path3)
                    for path4 in os.listdir(tmp_path3):
                        tmp_path4 = os.path.join(tmp_path3, path4)
                        for path5 in os.listdir(tmp_path4):
                            tmp_path5 = os.path.join(tmp_path4, path5)
                            #label = path3 + "_" + path4 + "_" + path5
                            for path6 in os.listdir(tmp_path5):
                                tmp_path6 = os.path.join(tmp_path5, path6)
                                for path7 in os.listdir(tmp_path6):
                                    tmp_path7 = os.path.join(tmp_path6, path7)
                                    csv_count = 0
                                    tar_count = 0
                                    for path8 in os.listdir(tmp_path7):
                                        tmp_path8 = os.path.join(tmp_path7, path8)
                                        for path9 in os.listdir(tmp_path8):
                                            target = os.path.join(tmp_path8, path9)
                                            if "csv" in target:
                                                with open(target) as f:
                                                    for row in f:
                                                        csv_count += 1
                                                    csv_last = int(row.split(",")[0].split(".")[0])
                                                csv_path.append(target)
                                            elif "tar" in target:
                                                try:
                                                    with tarfile.open(target) as archive:
                                                      tar_count = sum(1 for member in archive if member.isreg())
                                                except tarfile.ReadError:
                                                      print(target)
                                                      tar_count = 0
                                                tar_path.append([target, tar_count])
                                            else:
                                                redundancies.append(target)
                                    if csv_count > tar_count+1:
                                        if abs(csv_count-tar_count) not in irregular.keys(): 
                                            irregular[abs(csv_count-tar_count)] = [tmp_path7]
                                        else:
                                            irregular[abs(csv_count-tar_count)].append(tmp_path7)
                                    elif csv_count < tar_count:
                                        less_csv.append((csv_count, csv_last, tmp_path7))
                                    if csv_last+1 != tar_count:
                                        diff_csv.append((csv_last, tar_count, tmp_path7))
    print("Videos: ", len(All_path))
    print("CSV: ", len(csv_path))
    print("TAR: ", len(tar_path))
    print("Redundancies: ", len(redundancies))
    total = 0
    print("----------Irregular-------------")
    for key, value in irregular.items():
        print("Diff", key, ": ", value)
    for key, value in irregular.items():
        print("Diff", key, ": ", len(value))
        total += len(value)
    print("Total: ", total)
    print("---------less csv--------------")
    print("CSV (less): ", len(less_csv))
    print("---------diff csv--------------")
    print("CSV (diff than image)", len(diff_csv))
    for csv_last, tar_count, path in diff_csv:
        print(csv_last,",   ",  tar_count, path)
    #for csv_count, csv_last, path in less_csv:
    #    print(csv_last,": ", csv_count, path)

    with open("./"+folder+".pkl", "wb") as f:
        pickle.dump(tar_path,f )  
# [   0          1           2          3        4            5            6           7      8           9           ,  10 ]
# ['result', '20220705', '크로스핏', '초급', '시연자2', '데드리프트', '데드리프트', '오류1', '1', 'Motion2-1 (1).json', "1" ]
#    shutil.rmtree(output)
