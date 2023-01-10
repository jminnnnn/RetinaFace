import pickle
import cv2
from utils.find_every_path import FindEveryPath
import os
from tqdm import tqdm
import tarfile
import shutil
from os.path import isfile, join

FP = FindEveryPath()
FP.FindAll('./이랜서 최종 제출 파일 예시들/')
file_paths = FP.get_paths()

def frames_to_video(inputpath, outputpath, fps):
   image_array = []
   files = [f for f in os.listdir(inputpath) if isfile(join(inputpath, f))]
   
   for i in range(len(files)):

        img_path = inputpath+'/'+str(i) +'.jpg'
    
        img = cv2.imread(img_path)
        size =  (img.shape[1],img.shape[0])
        #    img = cv2.resize(img, size)
        image_array.append(img)
   
   fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
   out = cv2.VideoWriter(outputpath, fourcc, fps, size)
   
   for i in range(len(image_array)):
       out.write(image_array[i])
   out.release()

for path in tqdm(file_paths):   
    if 'tar' not in path:
        continue
    
    n_path = path.replace('.tar','') 
    print(path)
    # print(n_path)
    
    shutil.unpack_archive(path, n_path, "tar")
    frames_to_video(n_path, n_path+'.avi', 30)
    shutil.rmtree(n_path)
    



