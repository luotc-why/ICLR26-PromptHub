#!/bin/bash

python3 ICLR26-PromptHub/val_vp_detection.py \
 --mode spimg_spmask \
 --output_dir Datasets_PromptHub/output/logs\
 --device cuda:0\
 --base_dir Datasets_PromptHub/pascal-5i/ \
 --batch-size 8 \
 --lr 0.03\
 --epoch 150\
 --arr a1\
 --vp-model Prompt\
 --p-eps 1\
 --ckpt Datasets_PromptHub/weights/checkpoint-1000.pth\
 --vq_ckpt_dir Datasets_PromptHub/weights/vqgan\
 --save_base_dir Datasets_PromptHub/\
 --simidx 16\
 --dropout 0.25\
 --sigma 0.5\
 --save_model_path Datasets_PromptHub/ckpt/Det_K_16.pth