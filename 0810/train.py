from __future__ import print_function

import glob
import os
import time
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import DataLoader
from models.vit_head import VisionTransformer
from models.conv import VOFlowRes
from functools import partial
from utils import progress_bar, format_time, cosine_scheduler
from dataloader import *
from opts import *

from torch.utils.tensorboard import SummaryWriter
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device,' used.')
    

    # Prepare data
    if os.path.isfile(args.train_datainfo_path) and os.path.isfile(args.valid_datainfo_path):
        print('Load data info from {}'.format(args.train_datainfo_path))
        train_df = pd.read_pickle(args.train_datainfo_path)
        valid_df = pd.read_pickle(args.valid_datainfo_path)
    else:
        print('Create new data info')
        if args.dataset == 'chair':
            train_df, valid_df = get_data_info_chair()	
        # save the data info
        train_df.to_pickle(args.train_datainfo_path)
        valid_df.to_pickle(args.valid_datainfo_path)

    
    if args.dataset == 'chair':
        train_dataset = Chair_Dataset(train_df, flow=args.flow)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_processors, shuffle=True)
        valid_dataset = Chair_Dataset(valid_df, flow=args.flow)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.n_processors, shuffle=True)

    print('Number of samples in training dataset: ', len(train_dataset))
    print('Number of samples in validation dataset: ', len(valid_dataset))
    print('='*50)


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
    elif args.model == 'conv':
        model = VOFlowRes(
        ).to(device)

    start_epoch = 0
    # optimizer
    optimizer = create_optimizer(args, model)
    
    loss_scaler = NativeScaler()
    criterion = PoseLoss().cuda()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    min_loss_v = 1e10

    # tensorboard
    writer = SummaryWriter('runs/{}'.format(args.exp_id))

    if args.resume:
        print('Resuming from checkpoint ', args.load_model_path)
        checkpoint = torch.load(args.load_model_path)
        model.load_state_dict(checkpoint['net']) # load model weight 
        start_epoch = checkpoint['epoch']
        min_loss_v = checkpoint['min_loss_v']

    model.train()
    for epoch in range(start_epoch+1, args.epochs+1):
        # Train
        model.train()
        loss_mean = 0.0
        rot_loss_mean = 0.0
        tran_loss_mean = 0.0
        l1rot_loss_mean = 0.0
        l1tran_loss_mean = 0.0
        pred_attn_loss_mean = 0.0
        pred_maxattn_loss_mean = 0.0

        num_of_samples = len(train_loader)
        for batch_idx, (data, label, mask) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            mask = mask.to(device)
            
            pred_pose, attns = model(data, mask)
            
            total_ls, rot_ls, tran_ls, rot_l1, tran_l1, pred_attn_ls, pred_maxattn_ls = criterion(pred_pose, label, attns, mask)
            optimizer.zero_grad()
            loss_scaler(total_ls, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=False)
            lr_scheduler.step(epoch)

            total_ls = total_ls.data.cpu().numpy()
            rot_ls = rot_ls.data.cpu().numpy()
            tran_ls = tran_ls.data.cpu().numpy()
            rot_l1 = rot_l1.data.cpu().numpy()
            tran_l1 = tran_l1.data.cpu().numpy()
            pred_attn_ls = pred_attn_ls.data.cpu().numpy()
            pred_maxattn_ls = pred_maxattn_ls.data.cpu().numpy()

            loss_mean += float(total_ls)
            rot_loss_mean += float(rot_ls)
            tran_loss_mean += float(tran_ls)
            l1rot_loss_mean += float(rot_l1)
            l1tran_loss_mean += float(tran_l1)
            pred_attn_loss_mean += float(pred_attn_ls)
            pred_maxattn_loss_mean += float(pred_maxattn_ls)

            progress_bar(batch_idx, num_of_samples, 'Rot:%.3f, Tran:%.3f, Pred_Attn:%.3f'%(rot_ls, tran_ls, pred_attn_ls)) 
            
        loss_mean /= num_of_samples
        rot_loss_mean /= num_of_samples
        tran_loss_mean /= num_of_samples
        l1rot_loss_mean /= num_of_samples
        l1tran_loss_mean /= num_of_samples
        pred_attn_loss_mean /= num_of_samples
        pred_maxattn_loss_mean /= num_of_samples

        writer.add_scalar("Train Loss / mse_rot", rot_loss_mean, epoch)
        writer.add_scalar("Train Loss / mse_tran", tran_loss_mean, epoch)
        writer.add_scalar("Train Loss / l1_rot", l1rot_loss_mean, epoch)
        writer.add_scalar("Train Loss / l1_tran", l1tran_loss_mean, epoch)
        writer.add_scalar("Train Loss / bce_attn", pred_attn_loss_mean, epoch)
        writer.add_scalar("Train Loss / ms_attn", pred_maxattn_loss_mean, epoch)

    
        # Validation
        model.eval()
        loss_mean_valid = 0.0
        rot_loss_mean_valid = 0.0
        tran_loss_mean_valid = 0.0
        l1rot_loss_mean_valid = 0.0
        l1tran_loss_mean_valid = 0.0
        gt_attn_loss_mean_valid = 0.0
        gt_maxattn_loss_mean_valid = 0.0

        num_of_samples = len(valid_loader)
        for batch_idx, (data, label, mask) in enumerate(valid_loader):
            data = data.to(device)
            label = label.to(device)
            mask = mask.to(device)
            
            pred_pose, attns = model(data, mask)

            total_ls, rot_ls, tran_ls, rot_l1, tran_l1, gt_attn_ls, gt_maxattn_ls = criterion(pred_pose, label, attns, mask)
            
            total_ls = total_ls.data.cpu().numpy()
            rot_ls = rot_ls.data.cpu().numpy()
            tran_ls = tran_ls.data.cpu().numpy()
            rot_l1 = rot_l1.data.cpu().numpy()
            tran_l1 = tran_l1.data.cpu().numpy()
            gt_attn_ls = gt_attn_ls.data.cpu().numpy()
            gt_maxattn_ls = gt_maxattn_ls.data.cpu().numpy()

            loss_mean_valid += float(total_ls)
            rot_loss_mean_valid += float(rot_ls)
            tran_loss_mean_valid += float(tran_ls)
            l1rot_loss_mean_valid += float(rot_l1)
            l1tran_loss_mean_valid += float(tran_l1)
            gt_attn_loss_mean_valid += float(gt_attn_ls)
            gt_maxattn_loss_mean_valid += float(gt_maxattn_ls)
        
            progress_bar(batch_idx, num_of_samples, 'Rot:%.3f, Tran:%.3f, GT_Attn:%.3f'%(rot_l1, tran_l1, gt_attn_ls)) 


        loss_mean_valid /= num_of_samples
        rot_loss_mean_valid /= num_of_samples
        tran_loss_mean_valid /= num_of_samples
        l1rot_loss_mean_valid /= num_of_samples
        l1tran_loss_mean_valid /= num_of_samples
        gt_attn_loss_mean_valid /= num_of_samples
        gt_maxattn_loss_mean_valid /= num_of_samples

        writer.add_scalar("Valid Loss / mse_rot", rot_loss_mean_valid, epoch)
        writer.add_scalar("Valid Loss / mse_tran", tran_loss_mean_valid, epoch)
        writer.add_scalar("Valid Loss / l1_rot", l1rot_loss_mean_valid, epoch)
        writer.add_scalar("Valid Loss / l1_tran", l1tran_loss_mean_valid, epoch)
        writer.add_scalar("Valid Loss / bce_attn", gt_attn_loss_mean_valid, epoch)
        writer.add_scalar("Valid Loss / ms_attn", gt_maxattn_loss_mean_valid, epoch)

        print("Epoch : ", epoch," - rot loss : ", l1rot_loss_mean_valid," - tran loss : ",l1tran_loss_mean_valid, " - attn loss : ", gt_attn_loss_mean_valid)    

        # Save model
        if (epoch % 20 == 0) or (epoch == args.epochs) or (epoch == 3) or (epoch == 5):
            print("Saving Model ...")
            state = {
                    'net': model.state_dict(),
                    'min_loss_v' : min_loss_v,
                    'epoch': epoch,
                }
            
            if not os.path.isdir(os.path.join(args.checkpoint_dir, args.exp_id)):
                os.mkdir(os.path.join(args.checkpoint_dir, args.exp_id))
            print("Your model is stored in ",os.path.join(args.checkpoint_dir, args.exp_id, 'model_{}.pth'.format(epoch)))
            torch.save(state, os.path.join(args.checkpoint_dir, args.exp_id, 'model_{}.pth'.format(epoch)))
        
        if (loss_mean_valid < min_loss_v):
            min_loss_v = loss_mean_valid
            print("Saving Model ...")
            state = {
                    'net': model.state_dict(),
                    'min_loss_v' : min_loss_v,
                    'epoch': epoch,
                }
            
            if not os.path.isdir(os.path.join(args.checkpoint_dir, args.exp_id)):
                os.mkdir(os.path.join(args.checkpoint_dir, args.exp_id))
            print("Your model is stored in ",os.path.join(args.checkpoint_dir, args.exp_id, 'model_best.pth'))
            torch.save(state, os.path.join(args.checkpoint_dir, args.exp_id, 'model_best.pth'))

    
    writer.flush()
    
class PoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predicted, y, attns, mask):
       
        angle_loss = torch.nn.functional.mse_loss(predicted[:,3:], y[:,3:])
        translation_loss = torch.nn.functional.mse_loss(predicted[:,:3], y[:,:3])
        #angle_loss = torch.mean(torch.norm(y[:,3:] - predicted[:,3:], dim=1))
        #translation_loss = torch.mean(torch.norm(F.normalize(y[:,:3], p=2, dim=1) - F.normalize(predicted[:,:3], p=2, dim=1), dim=1))
        l1_rot = F.l1_loss(predicted[:,3:], y[:,3:])
        l1_tran = F.l1_loss(predicted[:,:3], y[:,:3])
        pose_loss = (10 * angle_loss + translation_loss)

        attn_loss = 0
        max_attn_loss = 0
        mask = mask.float()
        B, _, H, W = mask.shape
        mask = mask.reshape(B, H*W)
        
        attn_dev = attns
        #print(mask.min(), mask.max(), attn_dev.min(), attn_dev.max())
        
        for h in range(attn_dev.size(1)):
            for l in range(attn_dev.size(4)):
                

                attn_loss += torch.nn.functional.binary_cross_entropy(attn_dev[:,h,0,1:,l], mask)
                max_attn_loss += torch.mean(attn_dev[:,h,0,1:,l]/attn_dev[:,h,0,1:,l].max())
                #attn_loss += -(mask*attn_dev[:,h,0,1:,l]).sum() / (mask).sum()
                #attn_loss += -((1.-mask)*(1.-attn_dev[:,h,0,1:,l])).sum() / (1.-mask).sum()

        
        loss = pose_loss
        if args.attn :
            loss = loss + 0.05*attn_loss
        if args.max_attn :
            loss = loss - 10*max_attn_loss
 
            
        return (loss, angle_loss, translation_loss, l1_rot, l1_tran, attn_loss/(3.*attn_dev.size(1)), max_attn_loss/(3.*attn_dev.size(1)))


if __name__ == '__main__':
    main()
        
