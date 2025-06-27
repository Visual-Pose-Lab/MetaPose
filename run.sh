# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
#    --nproc_per_node=8 --master_port=2343 train.py \
#    --config experiments/human36m/human36m.yaml \
#    --image_encoder hrnet_32 --text_encoder ViT-B/32  --logdir ./logsdebug \
#    --eval

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
#     --nproc_per_node=1 --master_port=2345 train.py \
#     --config experiments/human36m/human36m.yaml \
#     --image_encoder hrnet_32 --text_encoder ViT-B/32  --logdir ./logsdebug \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#     --nproc_per_node=4 --master_port=2345 train.py \ 
#     --config experiments/human36m/human36m.yaml \
#     --image_encoder hrnet_48 --text_encoder ViT-B/32  --logdir ./logs48/meta

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#     --nproc_per_node=4 --master_port=2344 train.py \
#     --config experiments/human36m/human36m.yaml \
#     --image_encoder hrnet_32 --text_encoder ViT-B/32  --logdir ./logs32/Z_KL 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=2345 train.py \
    --config experiments/human36m/human36m.yaml \
    --image_encoder hrnet_32 --text_encoder ViT-B/32  --logdir ./logs32/HRN_KL
   
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
#     --nproc_per_node=1 --master_port=2344 train.py \
#     --config experiments/human36m/human36m.yaml \
#     --image_encoder hrnet_48 --text_encoder ViT-B/32  --logdir ./logs48/test --eval




