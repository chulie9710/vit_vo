import os
import random
import cv2
import glob
import torch
import torch.nn.functional as F
import time
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from opts import *
from numpy.linalg import inv as inv



def get_data_info_chair(partition=0.8, seq_len=2, shuffle=False):
    df_list = []
    for part in ['training_mask', 'testing_mask']:
        start_t = time.time()
        total_flow_paths = glob.glob('{}/{}/{}/*.npy'.format(args.dataset_dir, part, 'total_motion'))
        rigid_flow_paths = glob.glob('{}/{}/{}/*.npy'.format(args.dataset_dir, part, 'camera_motion'))
        depth_paths = glob.glob('{}/{}/{}/*.npy'.format(args.dataset_dir, part, 'depth_t0'))
        tran_paths = glob.glob('{}/{}/{}/*.npy'.format(args.dataset_dir,part,'relative_camera_translation'))
        rot_paths = glob.glob('{}/{}/{}/*.npy'.format(args.dataset_dir, part,'relative_camera_angle'))
        mask_paths = glob.glob('{}/{}/{}/*.npy'.format(args.dataset_dir,part,'object_mask'))
        K_paths = glob.glob('{}/{}/{}/*.npy'.format(args.dataset_dir,part,'camera_intrinsic'))
        
        total_flow_paths.sort()
        rigid_flow_paths.sort()
        depth_paths.sort()
        tran_paths.sort()
        rot_paths.sort()
        mask_paths.sort()
        K_paths.sort()

        poses = []
        Ks = []
        for i in range(len(rot_paths)):
            rot = np.load(rot_paths[i])
            tran = np.load(tran_paths[i])
            pose = np.concatenate((tran, rot), axis=-1)
            poses += [pose]

            K = np.load(K_paths[i])
            Ks += [K]
            
        
        n_frames = len(total_flow_paths)
        print('has {} frames'.format(n_frames))
        F_path = [total_flow_paths[i] for i in range(n_frames)]
        R_path = [rigid_flow_paths[i] for i in range(n_frames)]
        D_path = [depth_paths[i] for i in range(n_frames)]
        Y = [poses[i] for i in range(n_frames)]
        M_path = [mask_paths[i] for i in range(n_frames)]
        K_path = [Ks[i] for i in range(n_frames)]
        

        print('finish in {} sec'.format(time.time()-start_t))
    
        # Convert to pandas dataframes
        data = {'total_flow_path': F_path, 'rigid_flow_path': R_path, 'depth_path': D_path, 'pose': Y, 'mask_path': M_path, 'K_path': K_path}
        df = pd.DataFrame(data, columns = ['total_flow_path', 'rigid_flow_path', 'depth_path', 'pose', 'mask_path', 'K_path'])
        # Shuffle through all videos
        if shuffle:
            df = df.sample(frac=1)
        df_list.append(df)
    
    return df_list




class Chair_Dataset(Dataset):
    def __init__(self, info_dataframe, flow='total'):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        
        self.data_info = info_dataframe
        self.flow = flow
        self.total_arr = np.asarray(self.data_info.total_flow_path)  # flow paths
        self.rigid_arr = np.asarray(self.data_info.rigid_flow_path)  # flow paths
        if self.flow=='total' or self.flow=='masked':
            self.flow_arr = self.total_arr 
        else: # 'rigid'
            self.flow_arr = self.rigid_arr
        self.depth_arr = np.asarray(self.data_info.depth_path)  # depth paths
        self.mask_arr = np.asarray(self.data_info.mask_path)  # mask paths
        self.K_arr = np.asarray(self.data_info.K_path)
        self.groundtruth_arr = np.asarray(self.data_info.pose)

          
    def __getitem__(self, index):
        groundtruth_sequence = self.groundtruth_arr[index]
        groundtruth_tensor = torch.Tensor(groundtruth_sequence)

        K_sequence = self.K_arr[index]
        K_tensor = torch.Tensor(K_sequence)
        
        flow_path = self.flow_arr[index]
        with open(flow_path, 'rb') as f:
            flow_tensor = np.load(f, allow_pickle=True).astype(np.float32) # (436, 1024, 2)
            #flow_tensor = cv2.resize(flow_tensor, dsize=(256,128), interpolation=cv2.INTER_LINEAR)
            flow_tensor = self.to_tensor(flow_tensor)
        
        depth_path = self.depth_arr[index]
        with open(depth_path, 'rb') as f:
            depth_tensor = np.load(f, allow_pickle=True).astype(np.float32)
            #depth_tensor = cv2.resize(depth_tensor, dsize=(256,128), interpolation=cv2.INTER_LINEAR)
            depth_tensor = self.to_tensor(depth_tensor)
            disparity_tensor = 1./depth_tensor

        mask_path = self.mask_arr[index]
        with open(mask_path, 'rb') as f:
            mask_tensor = np.load(f, allow_pickle=True).astype(np.float32)
            
            
            if self.flow == 'masked':
                mask_tensor_ = self.to_tensor(mask_tensor)
                flow_tensor = flow_tensor * mask_tensor_
            if args.mask == 'soft':
                total_path = self.total_arr[index]
                total_tensor = np.load(total_path, allow_pickle=True).astype(np.float32)
                rigid_path = self.rigid_arr[index]
                rigid_tensor = np.load(rigid_path, allow_pickle=True).astype(np.float32)
                object_flow = np.abs(total_tensor - rigid_tensor)
                mask_tensor = np.exp(-1*np.sum(object_flow, axis=2, keepdims=True))
                #mask_tensor = cv2.resize(mask_tensor, dsize=(32,16), interpolation=cv2.INTER_LINEAR)
                #print(mask_tensor.min(), mask_tensor.max())

            mask_tensor = cv2.resize(mask_tensor, dsize=(32,16), interpolation=cv2.INTER_LINEAR)
            mask_tensor = self.to_tensor(mask_tensor)
            
        #print(flow_tensor.shape, depth_tensor.shape)
        X_tensor = torch.cat((flow_tensor, disparity_tensor), axis=0)
        
        if args.vis:
            return (X_tensor, groundtruth_tensor, mask_tensor, K_tensor, depth_tensor)
        else:
            return (X_tensor, groundtruth_tensor, mask_tensor)

    def __len__(self):
        return len(self.data_info.index)
        #return 16


def get_data_info_sintel(partition=0.8, seq_len=2, shuffle=False):
    df_list = []
    for part in range(2):
        start_t = time.time()
        pose_paths = glob.glob('{}/{}/*/*.npy'.format(args.dataset_dir,'pose'))
        flow_paths = glob.glob('{}/{}/*/*.npy'.format(args.dataset_dir, 'flow'))
        depth_paths = glob.glob('{}/{}/*/*.npy'.format(args.dataset_dir, 'depth'))
        mask_paths = glob.glob('{}/{}/*/*.png'.format(args.dataset_dir,'rigidity'))
        K_paths = glob.glob('{}/{}/*/*.cam'.format(args.dataset_dir,'camdata_left'))
        Kinv_paths = glob.glob('{}/{}/*/*.npy'.format(args.dataset_dir,'K_inv'))
        flow_paths.sort()
        depth_paths.sort()
        pose_paths.sort()
        mask_paths.sort()
        K_paths.sort()
        Kinv_paths.sort()

        n_val = int((1-partition)*len(pose_paths))
        st_val = int((len(pose_paths)-n_val)/2)
        ed_val = st_val + n_val
        
        poses = []
        Ks = []
        Kinvs = []

        for i in range(len(pose_paths)):
            pose = np.load(pose_paths[i])
            pose = np.concatenate((pose[3:], pose[:3]), axis=-1)
            poses += [pose]

            f = open(K_paths[i],'rb')
            K = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
            Ks += [K]
            
            inv_K = np.load(Kinv_paths[i])
            Kinvs += [inv_K]

        if part == 1:
            poses = poses[st_val:ed_val]
            flow_paths = flow_paths[st_val:ed_val]
            depth_paths = depth_paths[st_val:ed_val]
            mask_paths = mask_paths[st_val:ed_val]
            Ks = Ks[st_val:ed_val] 
            Kinvs = Kinvs[st_val:ed_val] 
        else:
            poses = np.concatenate((poses[:st_val], poses[ed_val:]), axis=0)
            flow_paths = flow_paths[:st_val] + flow_paths[ed_val:]
            depth_paths = depth_paths[:st_val] + depth_paths[ed_val:]
            mask_paths = mask_paths[:st_val] + mask_paths[ed_val:]
            Ks = np.concatenate((Ks[:st_val] , Ks[ed_val:]), axis=0)
            Kinvs = np.concatenate((Kinvs[:st_val], Kinvs[ed_val:]), axis=0)
        
        n_frames = len(flow_paths)
        print('has {} frames'.format(n_frames))
        X_path = [flow_paths[i] for i in range(n_frames)]
        D_path = [depth_paths[i] for i in range(n_frames)]
        Y = [poses[i] for i in range(n_frames)]
        M_path = [mask_paths[i] for i in range(n_frames)]
        K = [Ks[i] for i in range(n_frames)]
        Kinv = [Kinvs[i] for i in range(n_frames)]

        print('finish in {} sec'.format(time.time()-start_t))
    
        # Convert to pandas dataframes
        data = {'flow_path': X_path, 'depth_path': D_path, 'pose': Y, 'mask_path': M_path, 'K': K, 'Kinv': Kinv }
        df = pd.DataFrame(data, columns = ['flow_path', 'depth_path', 'pose', 'mask_path', 'K', 'Kinv'])
        # Shuffle through all videos
        if shuffle:
            df = df.sample(frac=1)
        df_list.append(df)
    
    return df_list


class Sintel_Dataset(Dataset):
    def __init__(self, info_dataframe):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        
        self.data_info = info_dataframe
        self.flow_arr = np.asarray(self.data_info.flow_path)  # flow paths
        self.depth_arr = np.asarray(self.data_info.depth_path)  # depth paths
        self.mask_arr = np.asarray(self.data_info.mask_path)  # mask paths
        self.K_arr = np.asarray(self.data_info.K)
        self.Kinv_arr = np.asarray(self.data_info.Kinv)
        self.groundtruth_arr = np.asarray(self.data_info.pose)

          
    def __getitem__(self, index):
        groundtruth_sequence = self.groundtruth_arr[index]
        groundtruth_tensor = torch.Tensor(groundtruth_sequence)

        K_sequence = self.K_arr[index]
        K_tensor = torch.DoubleTensor(K_sequence)

        Kinv_sequence = self.Kinv_arr[index]
        Kinv_tensor = torch.DoubleTensor(Kinv_sequence)
        
        flow_path = self.flow_arr[index]
        with open(flow_path, 'rb') as f:
            flow_tensor = np.load(f, allow_pickle=True).astype(np.float32) # (436, 1024, 2)
            flow_tensor = cv2.resize(flow_tensor, dsize=(256,128), interpolation=cv2.INTER_LINEAR)
            flow_tensor = self.to_tensor(flow_tensor)
            flow_tensor[0,:,:] = flow_tensor[0,:,:]*args.img_width/1024
            flow_tensor[1,:,:] = flow_tensor[1,:,:]*args.img_height/436
        
        depth_path = self.depth_arr[index]
        with open(depth_path, 'rb') as f:
            depth_tensor = np.load(f, allow_pickle=True).astype(np.float32)
            depth_tensor = cv2.resize(depth_tensor, dsize=(256,128), interpolation=cv2.INTER_LINEAR)
            depth_tensor = self.to_tensor(depth_tensor)
            disparity_tensor = 1./depth_tensor

        mask_path = self.mask_arr[index]
        
        with open(mask_path, 'rb') as f:
            mask_tensor = np.array(Image.open(mask_path).convert('L')).astype(np.float32)
            mask_tensor = cv2.resize(mask_tensor, dsize=(32,16), interpolation=cv2.INTER_LINEAR)
            #mask_tensor = 1-mask_tensor
            mask_tensor = self.to_tensor(mask_tensor)
            mask_tensor = np.where(mask_tensor==255, 1, 0)
            #mask_tensor = mask_tensor / np.sum(mask_tensor)
            
        #print(flow_tensor.shape, depth_tensor.shape)
        X_tensor = torch.cat((flow_tensor, disparity_tensor), axis=0)
        
        return (X_tensor, groundtruth_tensor, mask_tensor, K_tensor, depth_tensor, Kinv_tensor)

    def __len__(self):
        return len(self.data_info.index)
