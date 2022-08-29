import os
import argparse


parser = argparse.ArgumentParser(description='Transformer Visual Odometry')
# basic
parser.add_argument('dataset', type=str, default='chair', help='dataset')
parser.add_argument('--model', type=str, default='vit', help='model name of vit')
parser.add_argument('--img_height', type=int, default=128, help='image height')
parser.add_argument('--img_width', type=int, default=256, help='image width')
parser.add_argument('--n_processors',type=int, default=16, help='number of threads')
parser.add_argument('--flow', type=str, default='total', help='total or rigid or masked')
parser.add_argument('--attn', action='store_true', help='use mask loss to attn')
parser.add_argument('--max_attn', action='store_true', help='use mask loss to attn')
parser.add_argument('--mask', type=str, default='hard', help='use hard/soft mask')


# path
parser.add_argument('--dataset_dir', type=str, default='/warehouse/chulie9710/Chair', help='path to the dataset images root')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='directory name to save the checkpoints')
parser.add_argument('--train_datainfo_path', type=str, default='datainfo/train_chair.pickle')
parser.add_argument('--valid_datainfo_path', type=str, default='datainfo/valid_chair.pickle')

# train
#parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='number of iteration')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--exp_id', type=str, default='default', help='experiment name')

# test
parser.add_argument('--vis', action='store_true', help='for attn vis')
parser.add_argument('--load_model_path', type=str, default='checkpoint/default/model_best.pth', help='path to loaded model')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip-grad', type=float, default=5, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')


args = parser.parse_args()

if args.dataset == 'sintel':
    args.train_datainfo_path = 'datainfo/train_sintel.pickle'
    args.valid_datainfo_path = 'datainfo/valid_sintel.pickle'
    args.dataset_dir = '/warehouse/chulie9710/Sintel/training/' 
    args.pose_dir = '/warehouse/chulie9710/Sintel/training/'
    args.img_height = 104
    args.img_width = 256

if args.dataset == 'kitti':
    args.train_datainfo_path = 'datainfo/train_kitti.pickle'
    args.valid_datainfo_path = 'datainfo/valid_kitti.pickle'
    args.dataset_dir = '/warehouse/chulie9710/KITTI_left' 
    args.ori_height = 352
    args.ori_width = 1216
    args.img_width = 416


if args.dataset == 'tartan':
    args.train_datainfo_path = 'datainfo/train_tartan.pickle'
    args.valid_datainfo_path = 'datainfo/valid_tartan.pickle'
    args.dataset_dir = '/warehouse/chulie9710/Tartan/training/' 
    args.pose_dir = '/warehouse/chulie9710/Tartan/training/'
    args.img_height = 240
    args.img_width = 320

