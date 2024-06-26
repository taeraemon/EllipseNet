#
```
Forked From
https://openi.pcl.ac.cn/capepoint/EllipseNet
```



# 
<h1 align="left"> An Official PyTorch Implementation of “EllipseNet: Anchor-Free Ellipse Detection for Automatic Cardiac Biometrics in Fetal Echocardiography (MICCAI-2021)” <a href="https://arxiv.org/abs/2109.12474"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a></h1>
by Jiancong Chen#, Yingying Zhang#, Jingyi Wang, Xiaoxue Zhou, Yihua He*, Tong Zhang*. 

This repo contains the **official PyTorch implemetation** for EllipseNet.

Please refer to https://git.openi.org.cn/OpenMedIA/EllipseFit.Mindspore for a MindSpore version. Please be noted that the MindSpore version is not an Ellipse Detection Framework but using a 2D Unet to train a segmentation network and then using ellipses to fit the segmentation results.  

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

If you find this project useful for your research, please cit our work use the following BibTeX entry.

@inproceedings{chen2021ellipsenet,
  title={Ellipsenet: Anchor-free ellipse detection for automatic cardiac biometrics in fetal echocardiography},
  author={Chen, Jiancong and Zhang, Yingying and Wang, Jingyi and Zhou, Xiaoxue and He, Yihua and Zhang, Tong},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={218--227},
  year={2021},
  organization={Springer}
}

## Contact

If you have any questions about this paper, welcome to email to  [zhangt02@pcl.ac.cn](mailto:zhangt02@pcl.ac.cn)

