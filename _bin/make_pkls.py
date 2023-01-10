import os
import pickle
import cv2
from utils.find_every_path import *
import numpy as np
from tqdm import tqdm
import shutil

def read_basic_pkls_from_hy():
    with open('../AlphaPose/0.pkl', 'rb') as f:
        data0 = pickle.load(f)
        
    with open('../AlphaPose/1.pkl', 'rb') as f:
        data1 = pickle.load(f)
    
    with open('../AlphaPose/2.pkl', 'rb') as f:
        data2 = pickle.load(f)
    with open('../AlphaPose/3.pkl', 'rb') as f:
        data3 = pickle.load(f)
        # print(data2)

    # with open('../AlphaPose/3.pkl', 'rb') as f:
    #     data3 = pickle.load(f)
        
    # with open('../AlphaPose/4.pkl', 'rb') as f:
    #     data4 = pickle.load(f)
    
    # pickle.dump(final, f)

    final_list = []
    for p in data0:
        final_list.append(p)
    for p in data1:
        final_list.append(p)
    for p in data2:
        final_list.append(p)    
    for p in data3:
        final_list.append(p)    
    # for p in data3:
    #     final_list.append(p)
    # for p in data4:
    #     final_list.append(p)
    
    print(len(final_list))
    print(final_list[:3])
    return final_list

def divide_pkls(final_list):
    # folder_path = '../AlphaPose/pkls8/'
    folder_path = 'pkls_1018/'


    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    count = 0
    for i in range(0, len(final_list), 4000):
        f_list = []
        tmp_list = final_list[i : i+4000]
        
        for j in tmp_list:
            # tmp_dict = {}
            # tmp_dict['origin'] = j 
            # f_list.append([tmp_dict])   
            f_list.append(j)
        

        with open(folder_path + str(count)+'.pkl', 'wb') as f:
            pass

        with open(folder_path + str(count)+'.pkl', 'wb') as f:
            pickle.dump(f_list, f)
        
        count+=1

def read_sep_pkls_from_folders_and_save_in_final_pkls(number, base_path):

    base = base_path
    fs = os.listdir(base)
    
    pkl_list = [base + f for f in fs]

    tot_list = []

    for pkl in pkl_list:
        if 'final' in pkl:
            continue
        
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        
        for d in data:
            tot_list.append(d)

    print('files in ', base_path, ' is: ', len(tot_list))

    with open('../AlphaPose/final_pkls/' + str(number)+'.pkl', 'wb') as f:
        pass
    with open('../AlphaPose/final_pkls/' + str(number)+'.pkl', 'wb') as f:
        pickle.dump(tot_list,f)

# final_list = read_basic_pkls_from_hy()
def delete_small_tars(final_list):
    for tar in final_list:
        if '.tar' in tar:
            os.remove(tar)
            print('Deleted: ', tar)

def create_pkl():
    FP = FindEveryPath()
    FP.FindAll('crossfit_new/')
    total_paths = FP.get_paths()
    print(len(total_paths))
    
    
    tots=[]
    for p in total_paths:
        dd = {}
        dd['origin'] = p
        tots.append(dd)
    
    print(len(tots))
    
    #여기서 만든걸로 다시 ret face 돌리면 됨
    with open('retina_new.pkl', 'wb') as f:
        pass
    with open('retina_new.pkl', 'wb') as f:
        pickle.dump(tots,f)


def check_existence():
    with open('not_exist.pkl', 'rb') as f:
        data = pickle.load(f)
        
    base = '../CrossFit/'
    count=0
    for path in tqdm(data):
        print(path)
        tmp = path.split('/')
        tmp = tmp[7:]
        path = '/'.join(tmp)
        path = path.replace('tar','avi')
        
        b4_new = tmp[:-1]
        b4_new = '/'.join(b4_new)
        new_path= base + path
        
        # print(b4_new)
        # print(new_path)
        
        targ_path = './crossfit_new/CrossFit/' + path
        # print(targ_path)
        
        # raise ValueError
    
        if not os.path.exists(new_path):
            print('path does not exist: ', new_path)
            
        else:
            print('path exists..copying')
            if not os.path.exists('crossfit_new/CrossFit/' + b4_new):
                os.makedirs('crossfit_new/CrossFit/' + b4_new)
                
            copyfile(new_path, targ_path)
            count +=1
            
            
    print('처리한 파일: ', count)
    print('--vs original...check below num---')
    print(len(data))

# with open('retina_new.pkl', 'rb') as f:
#     dd = pickle.load(f)
        
# print(dd)
# print(len(dd))
            
# check_existence()
a = read_basic_pkls_from_hy()
divide_pkls(a)


# create_pkl()