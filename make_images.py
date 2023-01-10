from __future__ import print_function
from utils.find_every_path import FindEveryPath
import os
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import tarfile
import shutil
import pickle
import time 
from time import sleep
# from time import time

class Make_Images:

    def __init__(self, gpu, file_path, trained_model, network, confidence_threshold, top_k,
    nms_threshold, keep_top_k, vis_thres, file_count):

        self.pkl_file = file_path
        self.video_paths=[]
        self.gpu = gpu
        self.device = torch.device( self.gpu if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        self.trained_model = trained_model
        self.network = network
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres
        self.num = str(file_path)[-5:-4]
        
        #should be initialized before every vid
        self.bb_list = {}
        self.no_bb_list = []
        self.sizes=[]
        self.notopen=[]
        self.error_list=[]
        self.file_count = file_count
        
    def Start(self):
        with open(self.pkl_file, 'rb') as f:
            self.video_paths = pickle.load(f)

        self._load_video_and_blur_each_frame()
    
    def _check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # print('Missing keys:{}'.format(len(missing_keys)))
        # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        # print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def _remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        # print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def _load_model(self, model, pretrained_path, load_to_cpu):
        # print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(self.device))
        
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self,_remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
        
        self._check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        
        return model.cuda()

    def _declare_net_device_cfg(self):
        torch.set_grad_enabled(False)

        if self.network == "mobile0.25":
            cfg = cfg_mnet
        elif self.network == "resnet50":
            cfg = cfg_re50


        # print('Device: ', self.device)

        net = RetinaFace(cfg=cfg, phase='test')

        net = self._load_model(net, self.trained_model, "cuda")
        
        net.eval()
        net = net.to(self.device)

        # print('Model Loaded')
        return net, cfg

    def _create_output_path(self, input_path):
        
        tmp = input_path['origin'].replace('.avi','')

        return tmp

    def _load_video_and_blur_each_frame(self):
        redo_list=[]
        net, cfg = self._declare_net_device_cfg()
        
    #1200:1270
    #1370
        for path in tqdm(self.video_paths):
            
            st = time.time()
            
            self.bb_list = {}
            self.no_bb_list = []

            output_path = self._create_output_path(path)
            # output_path example : '../CrossFit_Final/20220728/AI/크로스핏/클린/메디신볼클린/풀동작오류/고급/왕명지/2/camera6/color/Motion2-7'
        


            expected_final_file = output_path+'.tar'
            
            if os.path.exists(expected_final_file):
                print(expected_final_file, ': already exists')
                continue

            else:
                print('working on...: ', output_path)


            if not os.path.exists(output_path):
                os.makedirs(output_path)


            # try:
            cap = cv2.VideoCapture(path['origin'])
            # cap = cv2.VideoCapture(path)
            
            if cap.isOpened() == False:
                print("Can\'t open the video: ", path['origin'])
                a=1
                self.notopen.append(path['origin'])
                continue
                

            print("Video Loaded... Processing Start...")
            
            count = 0 
            while True:
                
                cur_list = []
        
                ret, frame = cap.read()

                if frame is None:
              
                    a=1
                    break

                self._blur(frame, output_path, net, cfg, count)
                
                count += 1

            cap.release()
            # out.release()
            cv2.destroyAllWindows()

            self._postprocess(output_path)
            ed= time.time()
            tm = ed-st
            tm_min = tm / 60
            print('Elapsed time: ', tm_min, ' mins')

            
            #blurring complete. Make zip files.
            if os.path.exists(output_path):
                shutil.make_archive(output_path, 'tar', output_path)
                sleep(1)

                tar = tarfile.open(output_path+'.tar')
                tar_frames = len(tar.getmembers()) - 1
                tar.close()
                
                if tar_frames < 70:
                    print('not enough frames: ', output_path)
                
                folder_list = os.listdir(output_path)
                folder_num = len(folder_list)
                
                if tar_frames != folder_num:
                    redo_list.append(expected_final_file)
                    print('Error AGAIN.....: ', expected_final_file)
                    print('----Files in the folder: ', folder_num)
                    print('----Files in TAR: ', tar_frames)
                    
                shutil.rmtree(output_path)

        fold = str(self.file_count) +'/'
        
        if not os.path.exists(fold):
            os.makedirs(fold)

        with open(fold+'files_to_redo.pkl','wb') as f:
            pass

        with open(fold+'files_to_redo.pkl','wb') as f:
            pickle.dump(redo_list, f)
            print('PKL SAVED')

    def _blur(self, frame, output_path, net, cfg, count):

        resize = 1

        #Face Detection
        # img_raw = cv2.imread(frame)
        img_raw = frame
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = net(img)  # forward pass

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
                
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        
        i=0
        cur_list = []
        blur_check = 0

        for b in dets:

            if b[4] < self.vis_thres:
                continue
            
            b = list(map(int, b))
            cur_list.append([b[0], b[1], b[2], b[3]])
            
            
            for bb in cur_list:
                _list = cur_list
                i += 1
                x1, x2, y1, y2 = int(bb[0]), int(bb[2]), int(bb[1]), int(bb[3])
                
                x1=x1-4
                x2=x2+4
                y1=y1-4
                y2=y2+4

                bbox = [x1, x2, y1, y2]

                self.bb_list[count] = bbox

                # face = img_raw[y1: y2, x1 : x2]

                # if len(face) < 1:
                #     continue

                try:
                    face = img_raw[y1: y2, x1 : x2]
                    width = x2 - x1
                    height = y2 - y1
                    s = width * height

                    if len(self.sizes) ==0:
                        self.sizes.append(s)

                    if s > 300*np.mean(self.sizes):
                        print('Box size: ', str(s))
                        print('Mean is: ', np.mean(self.sizes))

                        print(output_path)
                        print('Frame number: ', count)
                        self.error_list.append(output_path)
                        continue
                    else:
                        self.sizes.append(s)
                        face = cv2.blur(face, (35,35))
                        img_raw[y1:y2, x1:x2] = face
                        blur_check = 1

                except:
                    
                    x1 = np.maximum(x1+4, 0)
                    x2 = x2-4
                    y1 = np.maximum(y1+4, 0)
                    y2 = y2-4
                    
                    face = img_raw[y1: y2, x1 : x2]
                    width = x2 - x1
                    height = y2 - y1
                    s = width * height

                    if len(self.sizes) ==0:
                        self.sizes.append(s)

                    if s > 300*np.mean(self.sizes):
                        print('Box size: ', str(s))
                        print('Mean is: ', np.mean(self.sizes))

                        print(output_path)
                        print('Frame number: ', count)
                        self.error_list.append(output_path)
                        continue
                    else:
                        self.sizes.append(s)
                        face = cv2.blur(face, (35,35))
                        img_raw[y1:y2, x1:x2] = face
                        blur_check = 1

            
            if blur_check == 1:
                continue

        if blur_check == 0:
            self.no_bb_list.append(count)

        # img_out_path = output_path +'/'+ image_path[len('_bin/Unzipped_tmps'+str(self.range)[1:-1]):]
        cv2.imwrite(output_path +'/'+ str(count) +'.jpg', img_raw)

    def _postprocess(self, output_path):
        #get frame nums : 0,1... etc.
        for frame in self.no_bb_list:
            left_1_cand = frame - 1
            right_1_cand = frame + 1

            left_2_cand = frame - 2
            right_2_cand = frame + 2

            left_3_cand = frame - 3
            right_3_cand = frame + 3

            if left_1_cand in self.bb_list:
                bbox = self.bb_list[left_1_cand]
            elif right_1_cand in self.bb_list:
                bbox = self.bb_list[right_1_cand]
            elif left_2_cand in self.bb_list:
                bbox = self.bb_list[left_2_cand]
            elif right_2_cand in self.bb_list:
                bbox = self.bb_list[right_2_cand]
            # elif left_3_cand in self.bb_list:
            #     bbox = self.bb_list[left_3_cand]
            # elif right_3_cand in self.bb_list:
            #     bbox = self.bb_list[right_3_cand]
            else:
                bbox = None
                pass
            
            if not bbox == None:
                try:
                    img_raw = cv2.imread(output_path + '/'+str(frame)+'.jpg')
                    x1, x2, y1, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    face = img_raw[y1: y2, x1 : x2]
                    face2 = cv2.blur(face, (35,35))
                    img_raw[y1:y2, x1:x2] = face2
                    cv2.imwrite(output_path +'/' + str(frame) +'.jpg', img_raw)
                except:
                    pass



        