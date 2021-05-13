cd src
# train
python main.py eldet --exp_id fake_exp3_base_theta5_iou --dataset coco_fhd --data_dir /data/cc/Data/CHD/fake_data/ --batch_size 24 --lr 5e-4 --gpus 5 --num_workers 16 --num_epochs 500 --input_h 608 --input_w 896 --iou_weight 1 --theta_weight 5 --not_rand_crop 

cd ..
