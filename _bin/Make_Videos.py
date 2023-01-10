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

class Make_Videos:
    def __init__(self):
        self.video_paths = []
        self.output_paths = []
        self.total_paths = []

    def Start(self):
        print('Starting...')
        self._find_input_video_paths()
        self._address_every_path()
        pass

    def _find_input_video_paths(self):
        path_finder = FindEveryPath()
        path_finder.FindAll('../data/')
        paths = path_finder.get_paths()
        
        for path in paths:
            if '.avi' in path:
                self.video_paths.append(path)
        print('Total {0} paths found'.format(len(paths)))

    def _check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def _remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def _load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self,_remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
        self._check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def _address_every_path(self):

        torch.set_grad_enabled(False)
        cfg = cfg_re50

        # device = torch.device("cpu" if args.cpu else "cuda")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('Device: ', device)

        net = RetinaFace(cfg=cfg, phase='test')
        net = self._load_model(net, trained_model, "cuda")
        net.eval()
        net = net.to(device)

        #Create [input_path, output_path] pairs
        for path in self.video_paths:
            #this exchanges '../data/' part with 'Blurred_videos' and removes .avi from the given path
            corresponding_output_path = 'Blurred_Videos(GB)/'+path[len('../data/'):-4]
            self.output_paths.append(corresponding_output_path)
            self.total_paths.append([path, corresponding_output_path])
        
        #make directories and save blurred vids into the new directories
        for input_path, output_path in tqdm(self.total_paths):

            if os.path.exists(output_path+'.avi'):
                print(output_path, ': already exists')
                continue 

            tmp = output_path.split('/')[:-1]
            output_directory = '/'.join(tmp)
            
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            print('Working on....{0}'.format(input_path))
            self._blur(input_path, output_path, net, cfg, device)

    def _blur(self, input_path, output_path, net, cfg, device):
        #Flow: caputre video frames --> blur face --> re-make vidoes using blurred images

        net = net
    
        #TODO
        net = nn.DataParallel(net, device_ids=[0,1,2,3])
        net = nn.DataParallel(net).to(device)

        cfg = cfg
        device = device

        resize = 1
        black = (0, 0, 0)
        count = 0

        VIDEO_FILE_PATH = input_path
        OUTPUT_FILE = output_path +'.avi'

        #Capture Video Frames
        cap = cv2.VideoCapture(VIDEO_FILE_PATH)
        if cap.isOpened() == False:
            print("Can\'t open the video(%d)" % (VIDEO_FILE_PATH))
            exit()

        # get video frame's width, height, frames per second info    
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (int(width), int(height)))

        # read one file, and conduct blurring on each frame
        while True:
            ret, frame = cap.read()
            
            if frame is None:
                print("Video Conversion Completed!")
                break

            #Face Detection
            img_raw = frame
            img = np.float32(img_raw)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)
            scale = scale.to(device)

            loc, conf, landms = net(img)  # forward pass

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:keep_top_k, :]
            landms = landms[:keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # blur on image. But does not save.
            i=0
            cur_list = []
            for b in dets:
                if b[4] < vis_thres:
                    continue
                b = list(map(int, b))
                # face = img_raw[b[1]:b[2], b[3]:b[4]]  # 공통 영역
                # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), black, -1)
                # # print(face)
                # # face = cv2.blur(face, (10,10))
                
                # img_raw[b[1]:b[2], b[3]:b[4]] = face

                # cx = b[0]
                # cy = b[1] + 12
                
                cur_list.append([b[0], b[1], b[2], b[3]])
                for bb in cur_list:
                    _list = cur_list
                    i += 1
                    x1, x2, y1, y2 = int(bb[0]), int(bb[2]), int(bb[1]), int(bb[3])
                    
                    x1=x1-3
                    x2=x2+3
                    y1=y1-3
                    y2=y2+3

                    face = img_raw[y1: y2, x1 : x2] 
                    face = cv2.blur(face, (20,20))
                    img_raw[y1:y2, x1:x2] = face

            out.write(img_raw)

        cap.release()
        out.release()
        cv2.destroyAllWindows()