import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import trunc_normal_, lecun_normal_, to_2tuple



def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm_layer = norm_layer

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attend = nn.Softmax(dim=-1)
        self.attend_obj = nn.Sigmoid()
        self.proj = nn.Linear(int(dim/2), dim)

    def forward(self, x, gt_mask=None):

        B, N, C = x.shape # [16, 513, 768]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        

        attn = (q @ k.transpose(-2, -1)) * self.scale
        #print(q.shape, attn.shape) # [16, 6, 513, 128], [16, 6, 513, 513]
        attn_obj = (q[:,0:3] @ k[:,0:3].transpose(-2, -1)) * self.scale
        attn_pose = (q[:,3:] @ k[:,3:].transpose(-2, -1)) * self.scale
        
        attn_obj = self.attend_obj(attn_obj)
        attn_pose = self.attend(attn_pose)
        
        #print(attn_obj.shape, v[:,3:].shape) # [16, 3, 513, 513], [16, 3, 513, 128]
        # x = [16, 513, 384] 
        x = (attn_pose @ (attn_obj @ v[:,3:])).transpose(1, 2).reshape(B, N, int(C/2)) 
 
        
        x = self.proj(x)

        return  x, attn_obj


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.mlp_hidden_dim = mlp_hidden_dim

    def forward(self, x, gt_mask=None):
       
        if type(x) is tuple:
            x, attn = x
        B, N, C = x.shape

        tmp, attn = self.attn(self.norm1(x), gt_mask)
        x = x + tmp
        x = x + self.mlp(self.norm2(x))

        return x, attn


class VisionTransformer(nn.Module):

    def __init__(self, img_size=(128,256), patch_size=16, in_chans=3, num_classes=6, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', keep_rate=(1, ), fuse_token=False):
        
        super().__init__()
        self.img_size = img_size
        self.depth = depth
        self.first_shrink_idx = depth
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.rot_head = nn.Linear(self.embed_dim, 3)
        self.tran_head = nn.Linear(self.embed_dim, 3)

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.rot_head
       

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.rot_head = nn.Linear(self.embed_dim, 3)
        self.tran_head = nn.Linear(self.embed_dim, 3)
     
    @property
    def name(self):
        return "vit_simple_topk"

    def forward_features(self, x, gt_mask):
        _, _, h, w = x.shape

        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        x = torch.cat((cls_token, x), dim=1)
       
        # for input with another resolution, interpolate the positional embedding.
        # used for finetining a ViT on images with larger size.
        pos_embed = self.pos_embed
        if x.shape[1] != pos_embed.shape[1]:
            assert h == w  # for simplicity assume h == w
            real_pos = pos_embed[:, self.num_tokens:]
            hw = int(math.sqrt(real_pos.shape[1]))
            true_hw = int(math.sqrt(x.shape[1] - self.num_tokens))
            real_pos = real_pos.transpose(1, 2).reshape(1, self.embed_dim, hw, hw)
            new_pos = F.interpolate(real_pos, size=true_hw, mode='bicubic', align_corners=False)
            new_pos = new_pos.reshape(1, self.embed_dim, -1).transpose(1, 2)
            pos_embed = torch.cat([pos_embed[:, :self.num_tokens], new_pos], dim=1)

        x = x + pos_embed

        attns = []
        init = x
        
        for i, blk in enumerate(self.blocks):
            if i<6:
                x, attn = blk(x)
                attns.append(attn)
            else:
                x, attn = blk(x)
            
            
        attns = torch.stack(attns, dim=-1).squeeze()
    
        x = self.norm(x)
        
        return x[:, 0], attns

    def forward(self, x, gt_mask):
        x, attns = self.forward_features(x, gt_mask)
        
        rot = self.rot_head(x)
        tran = self.tran_head(x)
        x = torch.cat((tran, rot), dim=-1)

        x = 0.001*x
        
        return x, attns

