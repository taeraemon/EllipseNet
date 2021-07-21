cd src
# train
python main.py eldet --exp_id exp6_base_onlyiou --dataset coco_fhd --batch_size 24 --lr 5e-4 --gpus 5 --num_workers 16 --num_epochs 500 --input_h 608 --input_w 896 --iou_weight 1 --ellipse_weight 0 --not_rand_crop 

cd ..
