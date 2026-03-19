#!/bin/bash

python3 ICLR26-PromptHub/val_vp_segmentation_for_coco.py \
 --fold 1 \
 --mode spimg_spmask \
 --output_dir Datasets_PromptHub/output/logs\
 --device cuda:0\
 --base_dir Datasets_PromptHub/coco/ \
 --batch-size 8 \
 --lr 0.03\
 --epoch 150\
 --arr a1\
 --vp-model Prompt\
 --p-eps 1\
 --ckpt Datasets_PromptHub/weights/checkpoint-1000.pth\
 --vq_ckpt_dir Datasets_PromptHub/weights/vqgan\
 --save_base_dir Datasets_PromptHub/\
 --simidx 1\
 --dropout 0.25\
 --sigma 0.65\
 --save_model_path Datasets_PromptHub/ckpt/Coco_K_1_Fold_1.pth