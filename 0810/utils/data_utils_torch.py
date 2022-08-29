import numpy as np
import torch
from collections import Counter

pixel_coords = None


def intrinsic_inverse(K):
    fx = K[0, 0]
    fy = K[1, 1]
    x0 = K[0, 2]
    y0 = K[1, 2]
    
    
    K_inv = torch.DoubleTensor([
        1.0 / fx, 0.0, -x0 / fx,
        0.0, 1.0 / fy, -y0 / fy,
        0.0, 0.0, 1.0
    ]).reshape(3, 3)
    
    
    return K_inv

def pixel_coord_generation(depth, is_homogeneous=True):
    global pixel_coords
    b, h, w = depth.size()

    n_x = torch.linspace(-1.0, 1.0, w, dtype=torch.double).unsqueeze(-1)
    n_y = torch.linspace(-1.0, 1.0, h, dtype=torch.double).unsqueeze(-1)
    
    x_t = torch.mul(torch.ones((h, 1)) , n_x.T )
    y_t = torch.mul(n_y, torch.ones((1, w)))
    
    x_t = (x_t + 1.0) * 0.5 * (w - 1.0)
    y_t = (y_t + 1.0) * 0.5 * (h - 1.0)
    
    xy_coord = torch.cat([x_t.unsqueeze(-1), y_t.unsqueeze(-1)], dim=-1)
    
    if is_homogeneous:
        xy_coord = torch.cat([xy_coord, torch.ones((h, w, 1))], dim=-1)
    
    pixel_coords = xy_coord.unsqueeze(0).expand(b,-1,-1,-1).float().cuda()
    return pixel_coords
    """
    b, h, w = depth.size()
    i_range = torch.arange(-1, 1, 2.0/h).view(1, h, 1).expand(1,h,w)  # [1, H, W]
    j_range = torch.arange(-1, 1, 2.0/w).view(1, 1, w).expand(1,h,w)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1).expand(b,-1,-1,-1)  # [1, 3, H, W]

    return pixel_coords
    """


def deg2mat_xyz(deg, left_hand=False):
    if left_hand:
        deg = -deg
    
    B = deg.size(0)
    rad = torch.deg2rad(deg)
    x, y, z = rad[:,0], rad[:,1], rad[:,2]
    
    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    R = xmat @ ymat @ zmat
    return R
  

def get_skew_mat(transx,rot):
    trans = -rot.permute(0,2,1).matmul(transx[:,:,np.newaxis])[:,:,0]
    rot = rot.permute(0,2,1)
    tx = torch.zeros(transx.shape[0],3,3).cuda()
    tx[:,0,1] = -transx[:,2]
    tx[:,0,2] = transx[:,1]
    tx[:,1,0] = transx[:,2]
    tx[:,1,2] = -transx[:,0]
    tx[:,2,0] = -transx[:,1]
    tx[:,2,1] = transx[:,0]
    return rot.matmul(tx)

def sampson_err(x1h, x2h, F):
    l2 = F.permute(0,2,1).matmul(x1h)
    l1 = F.matmul(x2h)
    algdis = (l1 * x1h).sum(1)
    dis = algdis**2 /  (1e-9+l1[:,0]**2+l1[:,1]**2+l2[:,0]**2+l2[:,1]**2)
    return dis

def angular3D(K_t0, K_t1, R, T, p_coord_t0, d_t0, gen_depth=False):

    B, hh, ww = d_t0.shape
    
    flat_p_coord_t0 = p_coord_t0.permute(0, 3, 1, 2).reshape(B, 3, -1).cuda()
    flat_d_t0 = d_t0.reshape(B, -1)
    
    
    # pose_vec2mat
    translation = T.unsqueeze(-1)
    transform_mat = torch.cat((R, T.unsqueeze(-1)), dim=2)
    aux_mat = torch.zeros([B,4]).cuda().unsqueeze(1)
    aux_mat[:,:,3] = 1
    M = torch.cat((transform_mat, aux_mat), dim=1)
    
    
    K_t0_inv = K_t1.inverse()
    K_t1_pad = torch.cat((torch.cat((K_t0, torch.zeros(B, 3, 1).cuda()), dim=2), torch.Tensor([[0, 0, 0 ,1]]).expand(B,-1).unsqueeze(1).cuda()),dim=1)
    filler = torch.ones((B, 1, hh * ww)).cuda()
    
    
    c_coord_t0 = torch.bmm(K_t0_inv, flat_p_coord_t0).cuda() * flat_d_t0.cuda().unsqueeze(1)
    c_coord_t0 = torch.cat((c_coord_t0, filler), dim=1)
    #print(M.dtype, c_coord_t0.dtype)
    c_coord_t1 = torch.bmm(M, c_coord_t0)
    
    flat_scene_motion = c_coord_t1[:, :3, :] - c_coord_t0[:, :3, :]
    
    unnormal_p_coord_t1 = torch.bmm(K_t1_pad, c_coord_t1)
    # Absolute for avioding reflection
    p_coord_t1 = torch.div(unnormal_p_coord_t1,(torch.abs(unnormal_p_coord_t1[:, 2, :]).unsqueeze(1) + 1e-12))
    
    flat_f = p_coord_t1[:, :2, :] - flat_p_coord_t0[:, :2, :]
        
    scene_motion = flat_scene_motion.reshape(B, 3, hh, ww).permute(0, 2, 3, 1)
    f = flat_f.reshape(B, 2, hh, ww) #.permute(1, 2, 0)

    H01 = torch.bmm(torch.bmm(K_t0,R),K_t0_inv)
    comp_hp1 = torch.bmm(H01,p_coord_t1[:,:3,:])
    parallax2d = (comp_hp1/ comp_hp1[:,-1:]- flat_p_coord_t0[:,:3,:])[:,:2]
    p2dmag = parallax2d.norm(2,1)[:, np.newaxis]
    p2dnorm = parallax2d / (1e-9 + p2dmag)
    foe_cam = torch.bmm(K_t0_inv, translation)
    foe_cam = foe_cam[:,:2] / (1e-9+foe_cam[:,-1:])
    direct = foe_cam - flat_p_coord_t0[:,:2]
    directn = direct / (1e-9+direct.norm(2,1)[:,np.newaxis])

    #print((p2dnorm-directn).mean())

    #mcost = ((-1*p2dnorm-directn)**2).sum(1,keepdims=True)
    mcost = -(translation[:,-1:]).sign()*(directn*p2dnorm).sum(1,keepdims=True)
    mcost = mcost.reshape(B, 1, hh, ww).float()
    #mask = torch.where(mcost<=0., 0, 1)

    # get skew matrix
    Ex = get_skew_mat(T.cuda(),R.cuda())
    sampom = sampson_err(torch.bmm(K_t0_inv, flat_p_coord_t0[:,:3,:]),
                torch.bmm(K_t0_inv, p_coord_t1[:,:3,:]),
                Ex.cuda().permute(0,2,1)).reshape(B, hh, ww).float()

    return mcost, f

def create_motion(K_t0, K_t1, R, T, p_coord_t0, d_t0, gen_depth=False):

    B, hh, ww = d_t0.shape
    
    flat_p_coord_t0 = p_coord_t0.permute(0, 3, 1, 2).reshape(B, 3, -1).cuda()
    flat_d_t0 = d_t0.reshape(B, -1)
    
    # pose_vec2mat
    translation = T.unsqueeze(-1)
    transform_mat = torch.cat((R, T.unsqueeze(-1)), dim=2)
    aux_mat = torch.zeros([B,4]).cuda().unsqueeze(1)
    aux_mat[:,:,3] = 1
    M = torch.cat((transform_mat, aux_mat), dim=1)
    
    
    K_t0_inv = K_t1.inverse()
    K_t1_pad = torch.cat((torch.cat((K_t0, torch.zeros(B, 3, 1).cuda()), dim=2), torch.Tensor([[0, 0, 0 ,1]]).expand(B,-1).unsqueeze(1).cuda()),dim=1)
    
    filler = torch.ones((B, 1, hh * ww)).cuda()
    
    
    c_coord_t0 = torch.bmm(K_t0_inv, flat_p_coord_t0)# * flat_d_t0
    
    c_coord_t0 = c_coord_t0.cuda() * flat_d_t0.cuda().unsqueeze(1)
    c_coord_t0 = torch.cat((c_coord_t0, filler), dim=1)
    #print(M.dtype, c_coord_t0.dtype)
    c_coord_t1 = torch.bmm(M, c_coord_t0)
    
    flat_scene_motion = c_coord_t1[:, :3, :] - c_coord_t0[:, :3, :]
    
    unnormal_p_coord_t1 = torch.bmm(K_t1_pad, c_coord_t1)
    # Absolute for avioding reflection
    p_coord_t1 = torch.div(unnormal_p_coord_t1,(torch.abs(unnormal_p_coord_t1[:, 2, :]).unsqueeze(1) + 1e-12))
    
    flat_f = p_coord_t1[:, :2, :] - flat_p_coord_t0[:, :2, :]
        
    scene_motion = flat_scene_motion.reshape(B, 3, hh, ww).permute(0, 2, 3, 1)
    f = flat_f.reshape(B, 2, hh, ww) #.permute(1, 2, 0)
    
    return scene_motion, f