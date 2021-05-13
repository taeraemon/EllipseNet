#/bin/bash

MODEL_FILE=model_120
echo "Begin!"
python evaluate.py eldet --load_model ../exp/eldet/exp1_base_theta5/${MODEL_FILE}.pth > ../exp/logs/exp1_${MODEL_FILE}.txt
echo "Exp1 Done!"

python evaluate.py eldet --load_model ../exp/eldet/exp2_base_theta5_noise/${MODEL_FILE}.pth > ../exp/logs/exp2_${MODEL_FILE}.txt
echo "Exp2 Done!"

python evaluate.py eldet --load_model ../exp/eldet/exp3_base_theta5_iou/${MODEL_FILE}.pth > ../exp/logs/exp3_${MODEL_FILE}.txt
echo "Exp3 Done!"

python evaluate.py eldet --load_model ../exp/eldet/exp4_base_theta5_iou01/${MODEL_FILE}.pth > ../exp/logs/exp4_${MODEL_FILE}.txt
echo "Exp4 Done!"

# python evaluate.py eldet --load_model ../exp/eldet/exp5_base_theta5_iou_step/${MODEL_FILE}.pth > ../exp/logs/exp5_${MODEL_FILE}.txt
# echo "Exp5 Done!"

# python evaluate.py eldet --load_model ../exp/eldet/exp6_base_onlyiou/${MODEL_FILE}.pth > ../exp/logs/exp6_${MODEL_FILE}.txt
# echo "All Done!"