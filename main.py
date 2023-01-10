from make_images import *
from utils.find_every_path import * 
import argparse
import pickle

################################## CODE INFO #############################################
# INPUT : paths of videos
# OUTPUT : blurred image frames archived as tar file
# USED FACE LOCALIZATION MODEL: RetinaFace
##########################################################################################

# -*- coding: euc-kr -*-
parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--func', default=None, help='select make_images or other func')
parser.add_argument('--gpu', default='cuda:0', help='Use cpu inference')
parser.add_argument('--file', default='../AlphaPose/final.pkl', help='video folder paths saved in pkl file')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.45, type=float, help='visualization_threshold')
parser.add_argument('--file_count', default=0, type=float, help='file_count')
args = parser.parse_args()


def make_images():
    #make pkl of all video files in the given path
    FP = FindEveryPath()
    FP.FindAll(args.file)
    total_paths= []
    for p in FP.get_paths():
        d = {}
        d['original'] = p
        total_paths.append(d)

    with open('./total_paths.pkl', 'w' ) as f:
        pass
    with open('./total_paths.pkl', 'w' ) as f:
        pickle.dump(total_paths, f)

    #perform blurring 
    MI = Make_Images(args.gpu, './total_paths.pkl', args.trained_model, args.network, args.confidence_threshold, args.top_k,
    args.nms_threshold, args.keep_top_k, args.vis_thres, args.file_count)
  
    MI.Start()


if __name__ == '__main__':
    if args.func == 'make_images':
        print('Blurring Images...') 
        make_images()

    else:
        print('No function selected!')
        