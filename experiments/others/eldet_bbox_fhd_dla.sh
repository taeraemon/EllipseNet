cd src
# train
# python main.py eldet --exp_id det_fhd_dla --dataset coco_fhd --batch_size 32 --lr 5e-4 --gpus 7 --num_workers 16
# test
python test.py eldet --exp_id det_fhd_dla --keep_res --resume --debug 4
# flip test
# python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test 
# multi scale test
# python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
