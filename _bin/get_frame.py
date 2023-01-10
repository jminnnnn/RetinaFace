import os
import cv2
from tqdm import tqdm
import shutil

       

videos = os.listdir('image_creation_test')

path_list=[]

for video in videos:
    video_path = "image_creation_test/"+video
    out_path = "image_creation_test/images/"+video[:-4]
    path_list.append([video_path, out_path])

for _input, _output in tqdm(path_list):
    if os.path.exists(_output+ '.tar'):
        print(_output, ": exist")
        continue 
    if not os.path.exists(_output):
        os.makedirs(_output)
    if ".csv" in _input or ".json" in _input:
        continue
    vidcap = cv2.VideoCapture(_input)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(_output+"/%d.jpg" % count, image)     # save frame as JPEG file
        #print(os.path.join(image_out, name+"/%d.jpg" % count))
        success,image = vidcap.read()
        count += 1
    shutil.make_archive(_output, 'tar', _output)
    shutil.rmtree(_output)
    

print('Conversion Completed')