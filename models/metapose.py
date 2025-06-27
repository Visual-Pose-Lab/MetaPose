import torch
from torch import nn
from models import pose_hrnet
from models.pose_net import PoseNet
import clip

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.
    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class MetaPose(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.image_encoder.num_joints

        if config.model.image_encoder.type in ['hrnet_32', 'hrnet_48']:
            self.image_encoder = pose_hrnet.get_pose_net(config.model.image_encoder)

        if config.model.image_encoder.fix_weights:
            print("model image_encoder weights are fixed")
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        
        self.text_encoder, _ = clip.load(config.model.text_encoder.type, device='cpu')
        set_requires_grad(self.text_encoder, False)
        text_feat_channel = 512 if config.model.text_encoder.type == 'ViT-B/32' else 768
        token_weights = self.text_encoder.token_embedding.weight.data
        mean = token_weights.mean()
        std = token_weights.std()
        self.learnable_prompts = nn.Parameter(torch.normal(
            mean=mean,
            std=std,
            size=(17 * 10, text_feat_channel)
        ))

        # self.learnable_prompts = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(17 * 10, text_feat_channel)))

        # self.learnable_prompts = nn.Parameter(torch.zeros(17*20, 512)) # 512

        # self.learnable_prompts = nn.Parameter(torch.zeros(17*77, 512)) # 512

        # self.learnable_prompts = torch.normal(mean=mean,
        #                                       std=std,
        #                                       size=(17*77, 512)) # 512
        # self.learnable_prompts = nn.Parameter(torch.full((17 * 20, 512), 1e-6))


        self.pose_net = PoseNet(config.model, image_encoder=config.model.image_encoder.type, device=device)
    
    def encode_prompt(self, device): 
        text_token_arg = torch.ones(17, device=device, dtype=torch.long) * 76
        x = self.learnable_prompts.view(17, 77, 512).to(device)

        x = x + self.text_encoder.positional_embedding.type(self.text_encoder.dtype)
        x = x.permute(1, 0, 2)      # NLD -> LND
        x = self.text_encoder.transformer(x)
        x = x.permute(1, 0, 2)      # LND -> NLD
        x = self.text_encoder.ln_final(x).type(self.text_encoder.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        prompts = x[torch.arange(x.shape[0]), text_token_arg] @ self.text_encoder.text_projection

        return  prompts
    
    def encode_text(self, text_token):
        x = self.text_encoder.token_embedding(text_token).type(self.text_encoder.dtype).squeeze(1)
        
        prompts = []
        for i in range(len(x)):
            # learnable_prompt = self.learnable_prompts[10*i:10*(i+1)]
            # prompts.append(torch.cat([x[i,0:-10,:], learnable_prompt],dim=0).unsqueeze(0))   # [77, 512] 

            learnable_prompt = self.learnable_prompts[20*i:20*(i+1)]
            prompts.append(torch.cat([learnable_prompt, x[i,0:-20,:]],dim=0).unsqueeze(0))   # [77, 512] 

        x = torch.cat(prompts,dim=0)

        x = x + self.text_encoder.positional_embedding.type(self.text_encoder.dtype)
        x = x.permute(1, 0, 2)      # NLD -> LND
        x = self.text_encoder.transformer(x)
        x = x.permute(1, 0, 2)      # LND -> NLD
        x = self.text_encoder.ln_final(x).type(self.text_encoder.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        prompts = x[torch.arange(x.shape[0]), text_token[:, :].argmax(dim=-1)] @ self.text_encoder.text_projection

        return  prompts

    def encode_text_all(self, text_token, action):         
        action_set = list(set(action))                                                          # [a, 17, 77]
        x = self.text_encoder.token_embedding(text_token).type(self.text_encoder.dtype)         # [a, 17, 77, 512]
        a,p,s,c = x.shape

        x_p = []
        for i in range(p):
            learnable_prompt = self.learnable_prompts[10*i:10*(i+1), :].unsqueeze(0).repeat(text_token.shape[0], 1, 1)
            ind = (text_token[:, i] == 0).sum(dim=1)
            if any(ind[:] < 10):
                print('ind:', ind)
            x_ = torch.cat([learnable_prompt, x[:, i, :-10, :]], dim=1)                     # [a, 17, 77, 512]

            x_ = x_ + self.text_encoder.positional_embedding.type(self.text_encoder.dtype)
            x_ = x_.permute(1, 0, 2)   # NLD -> LND
            x_ = self.text_encoder.transformer(x_)
            x_ = x_.permute(1, 0, 2)  # LND -> NLD
            x_ = self.text_encoder.ln_final(x_).type(self.text_encoder.dtype)

            # x_.shape = [batch_size, n_ctx_, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x_ = x_[torch.arange(x_.shape[0]), text_token[:, i].argmax(dim=-1)] @ self.text_encoder.text_projection
            x_p.append(x_)

        prompts_act_set = torch.stack(x_p, dim=1)                      # [a, 17, 512]

        prompts = []
        for act in action:
            prompts.append(prompts_act_set[action_set.index(act)])
        prompts = torch.stack(prompts, dim=0)                           # [b, 17, 512]

        return  prompts
    

    def forward(self, images, keypoints_2d_cpn, keypoints_2d_cpn_crop, keypoints_3d_gt=None, 
                text_feat=None, action_feat=None,action=None):

        # joints
        device = keypoints_2d_cpn.device
        b,l,p  = keypoints_2d_cpn.shape
        keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192//2, 256//2], device=device)
        keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        # image
        images = images.permute(0, 3, 1, 2).contiguous()
        features_list = self.image_encoder(images)

        # text
        text_feat = self.encode_text_all(action_feat, action) 
        # text_feat = self.encode_text(text_feat) 
        # text_feat = self.encode_prompt(device) 

        
        # 3d output
        keypoints_3d = self.pose_net(keypoints_2d_cpn, keypoints_2d_cpn_crop, features_list, text_feat, keypoints_3d_gt)

        return keypoints_3d

