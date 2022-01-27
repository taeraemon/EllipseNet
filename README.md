# EllipseNet: Anchor-Free Ellipse Detection for Automatic Cardiac Biometrics in Fetal Echocardiography (MICCAI-2021)
by Jiancong Chen, Yingying Zhang, Jingyi Wang, Xiaoxue Zhou, Yihua He, Tong Zhang. 

This repo contains the **official pytorch implemetation** for EllipseNet.

## Framework

![](readme/EllipseNet.png)

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Experiments

| Methods  |  Setting | Dice<sub>T</sub> |  Dice<sub>C</sub>  | Dice<sub>all</sub> | P<sub>avg</sub> |
|----------|----------|----------|------------|---------------|-------------|
|EllipseNet (exp6) | only IoU loss | 0.8813 | 0.8520 | 0.8666 | 0.8855 |
|EllipseNet (exp1) | w/o IoU loss  | 0.9338 | 0.9108 | 0.9224 | 0.8841 |
|EllipseNet (exp3) | w/ IoU loss   | **0.9430** | **0.9224** | **0.9336** | **0.8949** |

## Training and Evaluation

Prepare the elliptical dataset in coco-format. An example script is given in [scripts](scripts)/prepare_label.ipynb. We provide scripts for all the experiments in the [experiments](experiments) folder.

Usage:
~~~
chmod +x experiments/miccai21/*.sh
./experiments/miccai21/exp3_base_theta5_iou.sh
~~~



## Reproduction

If you need the docker for reproduction, please contact via email. We will provide the docker image. 




## License

EllipseNet itself is released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from [CenterNet](https://github.com/xingyizhou/CenterNet), [Rotated_IoU](https://github.com/lilanxiao/Rotated_IoU), [human-pose-estimation.pytorch](https://github.com/Microsoft/human-pose-estimation.pytorch) (image transform, resnet), [CornerNet](https://github.com/princeton-vl/CornerNet) (hourglassnet, loss functions), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2) (deformable convolutions), [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) (Pascal VOC evaluation) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

Chen J., Zhang Y., Wang J., Zhou X., He Y., Zhang T. (2021) EllipseNet: Anchor-Free Ellipse Detection for Automatic Cardiac Biometrics in Fetal Echocardiography. In: de Bruijne M. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science, vol 12907. Springer, Cham. https://doi.org/10.1007/978-3-030-87234-2_21

## Contact

If you have any questions about this paper, welcome to email to zhangt02@pcl.ac.cn

