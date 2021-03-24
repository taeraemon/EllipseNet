cd src
# train
python main.py eldet --exp_id fake_0214_eldet_fhd_dla_iouw5_nocrop_896x608 --dataset coco_fhd --data_dir /data/cc/Data/CHD/fake_data/ --batch_size 24 --lr 1e-3 --gpus 1 --num_workers 16 --num_epochs 1000 --input_h 608 --input_w 896 --iou_weight 2 --not_rand_crop --scale 0 --shift 0
# --iou_weight 1 --ellipse_weight 0.1 --theta_weight 30 --hm_weight 0.1 --off_weight 0.1 --wh_weight 0.01
# test
# python test.py eldet --exp_id det_fhd_dla --keep_res --resume --debug 4
# flip test
# python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test 
# multi scale test
# python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..


# Fix size testing.
# training chunk_sizes: [24]
# The output will be saved to  /data/cc/workspace/Repository/CenterNet/src/lib/../../exp/eldet/0204_eldet_fhd_dla_iouw5_nocrop_896x608
# heads {'hm': 2, 'reg': 2, 'l': 1, 'ratio_al': 1, 'ratio_ba': 1, 'theta': 1}
# Namespace(K=2, aggr_weight=0.0, agnostic_ex=False, arch='dla_34', aug_ddd=0.5, aug_rot=0, batch_size=24, cat_spec_wh=False, center_thresh=0.1, chunk_sizes=[24], data_dir='/data/cc/Data/CHD/detection/', dataset='coco_fhd', debug=0, debug_dir='/data/cc/workspace/Repository/CenterNet/src/lib/../../exp/eldet/0204_eldet_fhd_dla_iouw5_nocrop_896x608/debug', debugger_theme='white', demo='', dense_hp=False, dense_wh=False, dep_weight=1, dim_weight=1, down_ratio=4, ellipse_reg_weight=0, ellipse_weight=1, eval_oracle_dep=False, eval_oracle_hm=False, eval_oracle_hmhp=False, eval_oracle_hp_offset=False, eval_oracle_kps=False, eval_oracle_offset=False, eval_oracle_wh=False, exp_dir='/data/cc/workspace/Repository/CenterNet/src/lib/../../exp/eldet', exp_id='0204_eldet_fhd_dla_iouw5_nocrop_896x608', fix_res=True, flip=0.5, flip_test=False, gpus=[0], gpus_str='5', head_conv=256, heads={'hm': 2, 'reg': 2, 'l': 1, 'ratio_al': 1, 'ratio_ba': 1, 'theta': 1}, hide_data_time=False, hm_hp=True, hm_hp_weight=1, hm_weight=1, hp_weight=1, input_h=608, input_res=896, input_w=896, iou_weight=2.0, keep_res=False, kitti_split='3dop', load_model='', lr=0.0005, lr_step=[90, 120], master_batch_size=24, mean=array([[[0.216, 0.216, 0.216]]], dtype=float32), metric='loss', mse_loss=False, nms=False, no_color_aug=False, norm_wh=False, not_cuda_benchmark=False, not_hm_hp=False, not_prefetch_test=False, not_rand_crop=True, not_reg_bbox=False, not_reg_hp_offset=False, not_reg_offset=False, num_classes=2, num_epochs=2000, num_iters=-1, num_stacks=1, num_workers=16, off_weight=1, output_h=152, output_res=224, output_w=224, pad=31, peak_thresh=0.2, print_iter=0, rect_mask=False, reg_bbox=True, reg_hp_offset=True, reg_loss='l1', reg_offset=True, resume=False, root_dir='/data/cc/workspace/Repository/CenterNet/src/lib/../..', rot_weight=1, rotate=0, save_all=False, save_dir='/data/cc/workspace/Repository/CenterNet/src/lib/../../exp/eldet/0204_eldet_fhd_dla_iouw5_nocrop_896x608', scale=0.4, scores_thresh=0.1, seed=317, shift=0.1, std=array([[[0.222, 0.222, 0.222]]], dtype=float32), task='eldet', test=False, test_scales=[1.0], theta_weight=1, trainval=False, val_intervals=5, vis_thresh=0.3, wh_weight=0.1)
# Creating model...
