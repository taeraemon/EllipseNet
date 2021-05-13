cd src
# train
python main.py eldet --exp_id exp10_base_iou_sincos --dataset coco_fhd --batch_size 24 --lr 1e-4 --gpus 4 --num_workers 16 --num_epochs 500 --input_h 608 --input_w 896 --iou_weight 1 --theta_weight 0 --sincos_weight 1 --not_rand_crop 

# --resume --load_model ../exp/eldet/exp8_base_theta5_iou_sincos/model_best.pth

cd ..
