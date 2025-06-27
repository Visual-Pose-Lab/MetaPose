import math
from functools import partial
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_

from timm.models.layers import DropPath
from collections import defaultdict


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, latent_dim, text_latent_dim, num_head, dropout=0.):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim) 
        self.key = nn.Linear(text_latent_dim, latent_dim)  
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, xf):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head

        query = self.query(self.norm(x))
        key = self.key(self.text_norm(xf))

        query = query.unsqueeze(2)
        key = key.unsqueeze(1)
        key = key.repeat(int(B/key.shape[0]), 1, 1, 1)    

        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)

        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).unsqueeze(1)
        value = value.repeat(int(B/value.shape[0]), 1, 1, 1)
        value = value.view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        return y
    

class DeformableBlock(nn.Module):
    def __init__(self, feature_dim_list, dim, num_heads, num_samples, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_samples = num_samples
        head_dim = dim // num_heads
        self.attention_weights = nn.Linear(dim, num_heads * num_samples)
        self.sampling_offsets = nn.Linear(dim, 2 * num_heads * num_samples)
        self.embed_proj = nn.ModuleList([nn.Linear(dim_in, head_dim) for dim_in in feature_dim_list])
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = 0.01 * (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 2).repeat(1, self.num_samples, 1)
        for i in range(self.num_samples):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

    def forward(self, x, ref, features_list):
        b, l, p, c = x.shape
        l_ref = ref.shape[1] 

        weights = self.attention_weights(x).view(b, l, p, self.num_heads, self.num_samples)
        weights = F.softmax(weights, dim=-1).unsqueeze(-1) # b, l, p, num_heads, num_samples, 1
        offsets = self.sampling_offsets(x).reshape(b, l, p, self.num_heads*self.num_samples, 2).tanh()
        pos = offsets + ref.view(b, l_ref, p, 1, -1)

        features_sampled = [ F.grid_sample(features, pos[:, idx], align_corners=True).permute(0, 2, 3, 1).contiguous() \
                             for idx, features in enumerate(features_list)]

        # b, p, num_heads*num_samples, c
        features_sampled = [embed(features_sampled[idx]) for idx, embed in enumerate(self.embed_proj)]
        features_sampled = torch.stack(features_sampled, dim=1) # b, l, p, num_heads*num_samples, c // num_heads

        x = (weights * features_sampled.view(b, l, p, self.num_heads, self.num_samples, -1)).sum(dim=-2).view(b, l, p, -1)
        # b l p c
        return x


class SpatialEnhanceModule(nn.Module):
    def __init__(self, feature_dim_list, dim, num_heads, num_samples, qkv_bias=False, drop_path=0., mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.num_samples = num_samples

        # self attention
        self.de_att = DeformableBlock(feature_dim_list, dim, num_heads, num_samples, qkv_bias=False) 
        self.norm1 = norm_layer(dim)

        # ffn
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, guide_emb, ref, src):
        residual = x
        x = self.norm1(x + guide_emb)

        # self att
        x = self.de_att(x, ref, src)

        # ffn
        x = residual + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class SemanticEnhanceModule(nn.Module):
    def __init__(self, feature_dim_list, dim, num_heads, num_samples, qkv_bias=False, drop_path=0., mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.num_samples = num_samples
        # self attention
        self.de_att = DeformableBlock(feature_dim_list, dim, num_heads, num_samples, qkv_bias=False) 
        self.norm1 = norm_layer(dim)
        
        # gen ref
        self.offsets = nn.Linear(dim, 2)
        self.norm2 = norm_layer(dim)
        self.embed_proj = nn.ModuleList([nn.Linear(dim_in, dim) for dim_in in feature_dim_list])

        # ffn
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.offsets.weight.data, 0.)
        theta = torch.tensor(2.0 * math.pi)  
        grid_init = torch.stack([theta.cos(), theta.sin()])
        grid_init = 0.01 * (grid_init / grid_init.abs().max())
        with torch.no_grad():
            self.offsets.bias = nn.Parameter(grid_init.view(-1))

    def forward(self, x, query_pos, guide_emb, ref, src):
        residual = x
        x = self.norm1(x + guide_emb)

        offset = self.offsets(x).tanh()
        ref = ref + offset          # b l p c

        features_sampled = [
            F.grid_sample(features, ref[:,idx].unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous() \
            for idx, features in enumerate(src)]
        features_sampled = [embed(features_sampled[idx]) \
                                    for idx, embed in enumerate(self.embed_proj)]
        features_sampled = torch.stack([*features_sampled], dim=1)              # [b, 4, p, c]
        x = self.norm2(x + features_sampled)

        # self att
        x = x + query_pos
        x = self.de_att(x, ref, src)

        # ffn
        x = residual + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, ref
    
def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos

class PoseNet(nn.Module):
    def __init__(self, config=None, image_encoder='hrnet_32', num_joints=17, in_chans=2,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2,  norm_layer=None, device=None):
        """    ##########hybrid_image_encoder=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        base_dim = config.pose_net.base_dim
        embed_dim_ratio = config.pose_net.embed_dim_ratio
        depth = config.pose_net.levels
        out_dim = 3                         #### output dimension is num_joints * 3
        self.levels = config.pose_net.levels
        embed_dim = embed_dim_ratio * (self.levels+1)
        text_feat_channel = 512 if config.text_encoder.type == 'ViT-B/32' else 768
        if image_encoder in ['hrnet_32', 'hrnet_48']:
            feature_dim_list = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

        self.Spatial_img_embed = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, self.levels, num_joints, embed_dim_ratio)))
        self.Spatial_pos_embed = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, num_joints, embed_dim_ratio)))
        self.Spatial_txt_embed = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, num_joints, embed_dim_ratio)))

        self.scale_pos_embed = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, self.levels+1, embed_dim_ratio)))
        self.joint_pos_embed = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, num_joints, embed_dim)))

        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)
        # self.coord_embed_3d = nn.Linear(3, embed_dim_ratio)
        self.text_proj = nn.Linear(text_feat_channel, embed_dim_ratio)

        self.feat_embed = nn.ModuleList([nn.Linear(dim_in, embed_dim_ratio) for dim_in in feature_dim_list])
        self.query_embed = nn.Embedding(num_joints*(self.levels), embed_dim_ratio)

        self.fusion = nn.ModuleList([nn.Linear(embed_dim_ratio*2, embed_dim_ratio) for i in range(5)])
        self.dist_proj = nn.Linear(embed_dim_ratio, 2)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, depth)]  # stochastic depth decay rule

        self.joint_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                  for i in range(depth)])

        self.scale_blocks = nn.ModuleList([
            Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                  for i in range(depth)])

        self.SpatialEn = nn.ModuleList([
            SpatialEnhanceModule(feature_dim_list=feature_dim_list, dim=embed_dim_ratio, num_heads=4, num_samples=4, qkv_bias=qkv_bias, drop_path=dpr[i])
                                 for i in range(depth)])
        
        self.SemanticEn = nn.ModuleList([
            SemanticEnhanceModule(feature_dim_list=feature_dim_list, dim=embed_dim_ratio, num_heads=4, num_samples=4, qkv_bias=qkv_bias, drop_path=dpr[i])
                                  for i in range(depth)])
        
        self.dis_cross = CrossAttention(latent_dim=embed_dim_ratio, text_latent_dim=embed_dim_ratio, num_head=4)
        # self.dis_cross = nn.ModuleList([CrossAttention(latent_dim=embed_dim_ratio, text_latent_dim=embed_dim_ratio, num_head=4) for i in range(2)])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        self.inter_feat = defaultdict(list)
    
    def DiagonalGuassianDistribution(self, mean, var, logvar, dims=None):
        """
        mean: b, l, p, c
        std: b, l, p, c
        """
        return 0.5 * torch.sum(torch.pow(mean, 2) + var -1.0 - logvar, dim=dims)

    def forward(self, keypoints_2d, ref, features_list, text_feat, keypoints_3d_gt=None):
        b, p, c = keypoints_2d.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        # for i in range(1,len(features_list)+1):
        #     # torch.save(features_list[i-1].cpu(),f'features/hrnet/posing1__hrnet_lvl{i}.pt')
        #     self.inter_feat[f'fm{i}'].append(features_list[i-1].detach().cpu().numpy())

        points_emb = self.coord_embed(keypoints_2d).unsqueeze(1)                  # b, 1, p, c
        points_emb += self.Spatial_pos_embed

        # import pdb; pdb.set_trace()
        # text_emb = self.text_proj(text_feat).unsqueeze(0).repeat(b,1,1).unsqueeze(1)  # b, 1, p, c 
        text_emb = self.text_proj(text_feat).unsqueeze(1) # b, 1, p, c 
        text_emb += self.Spatial_txt_embed
        
        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous() \
            for features in features_list]

        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(self.feat_embed)]

        x = torch.stack([*features_ref_list], dim=1)        # b, 4, p, c
        x += self.Spatial_img_embed
        x = self.pos_drop(x)

        # Spatial-Driven Modality Enhancement
        x_p = x
        ref = ref.unsqueeze(1)                              # b 1 p c
        for blk in self.SpatialEn:
            x_p = blk(x_p, points_emb, ref, features_list)
        x_p = torch.cat([x_p, points_emb], dim=1) # b, 5, p, c

        # Semantic-Driven Modality Enhancement
        x_t = x
        b, l, p ,c = x_t.shape
        query_pos = self.query_embed.weight
        query_pos = query_pos.unsqueeze(0).expand(b, -1, -1).view(b,l,p,c)
        for blk in self.SemanticEn:
            x_t, ref = blk(x_t, query_pos, text_emb, ref, features_list)
        x_t = torch.cat([x_t,text_emb], dim=1) # b, 5, p, c

        
        # Transformation Alignment
        mean, logvar = self.dist_proj(x_p).split(1, dim=-1)
        esp = torch.randn_like(x_p)     # b l p c
        # logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)
        guassian_prompt = mean +  std * esp
        # kl_loss = self.DiagonalGuassianDistribution(mean, var, logvar, dims=[1,2])
        img_feat = [fusion(torch.cat([x_p[:,idx],x_t[:,idx]],dim=-1)).unsqueeze(1) \
                        for idx, fusion in enumerate(self.fusion)]
        x = torch.cat(img_feat, dim=1)
        x = rearrange(x, 'b l p c -> b (l p) c')
        guassian_prompt = rearrange(guassian_prompt, 'b l p c -> b (l p) c')
        x = x + self.dis_cross(x, guassian_prompt)
        x = rearrange(x, 'b (l p) c -> b l p c', p=p)

        # # ---------------------------------------------------------------------------------
        # mix_feat = [fusion(torch.cat([x_p[:,idx],x_t[:,idx]],dim=-1)).unsqueeze(1) \
        #         for idx, fusion in enumerate(self.fusion)]
        # x = torch.cat(mix_feat, dim=1)
        # mean, logvar = self.dist_proj(points_emb[:,0]).split(1, dim=-1)
        # esp = torch.randn_like(points_emb[:,0])     # b l p c
        # logvar = torch.clamp(logvar, -30.0, 20.0)
        # std = torch.exp(0.5 * logvar)
        # var = torch.exp(logvar)
        # guassian_prompt = mean + std * esp
        # kl_loss = self.DiagonalGuassianDistribution(mean, var, logvar, dims=[1,2])

        # x_lvl = rearrange(x[:,:-1], 'b l p c -> b (l p) c')
        # # guassian_prompt = rearrange(guassian_prompt, 'b l p c -> b (l p) c')
        # for blk in self.dis_cross:
        #     x_lvl = x_lvl + blk(x_lvl, guassian_prompt)
        # x_lvl = rearrange(x_lvl, 'b (l p) c -> b l p c', p=p)
        # x = torch.cat([x_lvl, x[:,-1:]], dim=1)
        # # ---------------------------------------------------------------------

        # emb_3d = self.coord_embed_3d(keypoints_3d_gt)
        # # import pdb; pdb.set_trace() 
        # x = torch.cat([x, emb_3d], dim=1) 

        # Transformer (Lifting Model)
        x = rearrange(x, 'b l p c -> (b p) l c', p=p)

        x += self.scale_pos_embed
        for blk in self.scale_blocks:
            x = blk(x)

        x = rearrange(x, '(b p) l c -> b p (l c)', p=p)

        x += self.joint_pos_embed
        for blk in self.joint_blocks:
            x = blk(x)

        x = self.head(x).view(b, 1, p, -1)
        return x

