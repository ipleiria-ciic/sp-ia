CUDA_VISIBLE_DEVICES='0' 
python imagenet_attack.py \
    --data_dir /home/joseareia/Documents/SPIA/NB07-0224/Datasets/Imagewoof/train/ \
    --uaps_save ./uaps_save/sga/ \
    --batch_size 2 \
    --minibatch 1 \
    --alpha 10 \
    --epoch 10 \
    --spgd 0 \
    --num_images 50 \
    --model_name vgg16 \
    --Momentum 0 \
    --cross_loss 1