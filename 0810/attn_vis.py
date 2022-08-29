from __future__ import print_function

import glob
import os
import random
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from functools import partial
from thop.profile import profile


from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from models.vit import VisionTransformer
from utils import progress_bar
from utils.flow_vis import find_rad_minmax, flow_to_color
from utils.data_utils_torch import deg2mat_xyz, pixel_coord_generation, create_motion
from dataloader import *
from opts import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# create heatmap from mask on image
def show_cam_on_image(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #heatmap = np.float32(heatmap) / 255
    cam = heatmap 
    #cam = cam / np.max(cam)
    return cam

def main():

    args.vis == True
    
    # Path
    load_model_path = args.load_model_path
    save_dir = 'result/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device,' used.')

    if args.model == 'vit':
        model = VisionTransformer(
            patch_size=8, 
            embed_dim=384, 
            depth=12, 
            num_heads=6, 
            mlp_ratio=4, 
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ).to(device)

    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint['net'])
    print('Load model from: ', load_model_path, ', epoch: ', checkpoint['epoch'])
    exp_id = load_model_path.replace('checkpoint/','')


    #model = Recorder(model)
    
    if os.path.isfile(args.train_datainfo_path) and os.path.isfile(args.valid_datainfo_path):
        print('Load data info from {}'.format(args.train_datainfo_path))
        train_df = pd.read_pickle(args.train_datainfo_path)
        valid_df = pd.read_pickle(args.valid_datainfo_path)
    else:
        print('Create new data info')
        if args.dataset == 'chair':
            train_df, valid_df = get_data_info_chair()
        elif args.dataset == 'sintel':
            train_df, valid_df = get_data_info_sintel()	
        # save the data info
        train_df.to_pickle(args.train_datainfo_path)
        valid_df.to_pickle(args.valid_datainfo_path)
    
    if args.dataset == 'chair':
        valid_dataset = Chair_Dataset(valid_df, flow=args.flow)
        valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=args.n_processors, shuffle=False)
    elif args.dataset == 'sintel':
        valid_dataset = Sintel_Dataset(valid_df)
        valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=args.n_processors, shuffle=False)
    # Predict
    model.eval()
    answer = []
    st_t = time.time()
    n_batch = len(valid_loader)

    for batch_idx, (data, label, mask, K, depth) in enumerate(valid_loader):
        
        data = data.to(device)
        K = K.to(device)
        label = label.to(device)
        mask = mask.to(device)

        if batch_idx == 0:
            total_ops, total_params = profile(model, inputs=(data,mask ))
            print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
            print("---|---|---")
            print("%s | %.2f | %.2f" % (exp_id, total_params / (1000 ** 2), total_ops / (1000 ** 3)))
        
        
        pred_pose, attns = model(data, mask)
        relative_camera_translation = pred_pose[:,:3]
        relative_camera_rotation = deg2mat_xyz(pred_pose[:,3:])
        pixel_coord_t0 = pixel_coord_generation(depth[:,0,:,:])

        _, rigid_flow = create_motion(K, K, relative_camera_rotation, relative_camera_translation, \
                                                    pixel_coord_t0, depth[:,0,:,:], False)

        total_flow = data[0,:2,:,:].permute(1,2,0).cpu().detach().numpy()
        rigid_flow = rigid_flow[0].permute(1,2,0).cpu().detach().numpy()
        object_flow = total_flow - rigid_flow

        # depth vis
        depth = depth[0,0].cpu().detach().numpy()
        # total flow vis
        rad_max, _min, _max = find_rad_minmax(total_flow) #only for no past 
        total_flow_ = flow_to_color(total_flow, convert_to_bgr=False, clip_flow=[_min, _max] ,rad_max=rad_max)
        # rigid flow vis
        rigid_flow_ = flow_to_color(rigid_flow, convert_to_bgr=False, clip_flow=[_min, _max] ,rad_max=rad_max)
        # object flow vis
        object_flow_ = flow_to_color(object_flow, convert_to_bgr=False, clip_flow=[_min, _max] ,rad_max=rad_max)
        # object mask vis
        #object_mask = object_mask[0]
        # gt mask vis
        mask = mask[0,0].cpu().numpy()

        
        fig, axes = plt.subplots(4,2)
        
        att_mat = torch.squeeze(attns[:,:,:,0:3].mean(dim=-1).mean(dim=0)) # head, 513, 513
        att_mat_h = att_mat[:,:] 
        residual_att = torch.eye(att_mat_h.size(0)).to(device) # (53, 53)
        aug_att_mat = att_mat_h + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        v = aug_att_mat.cpu().detach().numpy()
        lsa_mean = v[0,1:].reshape(16, 32)
        attn_max = lsa_mean.max()
        attn_min = lsa_mean.min()
        lsa_mean = lsa_mean / lsa_mean.max()
        lsa_mean = cv2.resize(lsa_mean, (256,128))[..., np.newaxis]

        mean = lsa_mean 
        result = show_cam_on_image(mean)
        
    
        # save
        
        axes[0,0].set_title('Original:{:.1f},{:.1f}'.format(total_flow.max(),total_flow.min()))
        _ = axes[0,0].imshow(total_flow_)
        axes[0,1].set_title('Depth:{:.1f},{:.1f}'.format(depth.max(),depth.min()))
        _ = axes[0,1].imshow(depth)
        axes[1,0].set_title('GT Mask')
        _ = axes[1,0].imshow(mask)
        axes[1,1].set_title('Rigid Flow:{:.1f},{:.1f}'.format(rigid_flow.max(),rigid_flow.min()))
        _ = axes[1,1].imshow(rigid_flow_)
        axes[2,0].set_title('Object Flow:{}'.format(object_flow_.max()))
        _ = axes[2,0].imshow(object_flow_)
        axes[2,1].set_title('Attn:{:.1f},{:.1f}'.format(attn_max, attn_min))
        _ = axes[2,1].imshow(result)
        
        label = label.cpu().detach().numpy()
        pred_pose = pred_pose.cpu().detach().numpy()
        axes[3,0].set_title('GT pose')
        _ = axes[3,0].text(0.1, 0, label[0,0] ,fontsize=8)
        _ = axes[3,0].text(0.1, 0.15, label[0,1] ,fontsize=8)
        _ = axes[3,0].text(0.1, 0.3, label[0,2] ,fontsize=8)
        _ = axes[3,0].text(0.1, 0.45, label[0,3] ,fontsize=8)
        _ = axes[3,0].text(0.1, 0.6, label[0,4] ,fontsize=8)
        _ = axes[3,0].text(0.1, 0.75, label[0,5] ,fontsize=8)

        axes[3,1].set_title('Pred pose')
        _ = axes[3,1].text(0.1, 0, pred_pose[0,0] ,fontsize=8)
        _ = axes[3,1].text(0.1, 0.15, pred_pose[0,1] ,fontsize=8)
        _ = axes[3,1].text(0.1, 0.3, pred_pose[0,2] ,fontsize=8)
        _ = axes[3,1].text(0.1, 0.45, pred_pose[0,3] ,fontsize=8)
        _ = axes[3,1].text(0.1, 0.6, pred_pose[0,4] ,fontsize=8)
        _ = axes[3,1].text(0.1, 0.75, pred_pose[0,5] ,fontsize=8)




        exp_id = load_model_path.split('/')[-2]
        if not os.path.exists('{}{}/{}_{}'.format(save_dir, args.dataset, exp_id, checkpoint['epoch'])):
            os.makedirs('{}{}/{}_{}'.format(save_dir, args.dataset, exp_id, checkpoint['epoch']))
        save_name = '{}{}/{}_{}/attn_{}.png'.format(save_dir, args.dataset, exp_id, checkpoint['epoch'], batch_idx)
        plt.savefig(save_name)
        print('Save ',save_name)
        
        
    
    

if __name__ == '__main__':
    main()
