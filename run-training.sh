# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 train.py \
#   --depth=20 --bs=64 --ep=100 --fp16=1 --alng=1e-3 --wpe=0.1 --use_pretrained_ckpt True --workers=8 --first_N=-1

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 train.py \
  --depth=20 --bs=32 --ep=100 --fp16=1 --alng=1e-3 --wpe=0.1 --use_pretrained_ckpt True --workers=8 --first_N=60000


# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 train.py \
#   --depth=20 --bs=8 --ep=250 --fp16=1 --alng=1e-3 --wpe=0.1 --use_pretrained_ckpt True