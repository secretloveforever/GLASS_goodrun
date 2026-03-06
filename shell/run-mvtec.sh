datapath=/home/c379/lzy/GLASS-main/datasets/MVTec
augpath=/home/c379/lzy/GLASS-main/datasets/dtd/images
classes=('carpet' 'grid' 'leather' 'tile' 'wood' 'bottle' 'cable' 'capsule' 'hazelnut' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

#此处有修改
cd ..
python /home/c379/lzy/GLASS-main/main.py \
    --results_path /home/c379/lzy/GLASS-main/results \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 64 \
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
  dataset \
    --distribution 0 \
    --mean 0.5 \
    --std 0.1 \
    --fg 0 \
    --rand_aug 1 \
    --batch_size 8 \
    --resize 288 \
    --imagesize 288 "${flags[@]}" mvtec $datapath $augpath