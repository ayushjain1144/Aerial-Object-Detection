# Aerial-Object-Detection
This repository contains our research work on Aerial Object Detection.

# Instructions

To train RetinaNet with VGG16 or ResNet50 feature extractor : 
```python
python keras-retinanet/keras_retinanet/bin/train.py --gpu <gpu_id> --backbone <vgg16 | resnet50> --epochs <total_epochs> --tensorboard-dir <tensorboard_dir> --compute-val-loss --config <path_to_config> --snapshot-path <snapshot_save_dir> --random-transform --snapshot <resume_snapshot> csv <train_csv> <class_mapping_csv> --val-annotations <val_csv>
```

To train RetinaNet with ResNeSt50 feature extractor:
```python
python detectron2-ResNeSt/tools/train_net.py --num-gpus <num_gpus> --config-file <path_to_config>
```

# Nice Readings

## Object Tracking

- https://arxiv.org/pdf/1707.00569.pdf

## Retinanet

- Retinanet Paper: https://arxiv.org/pdf/1708.02002.pdf
- Blog: https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/
- https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4

## RRNet

- Paper :       http://openaccess.thecvf.com/content_ICCVW_2019/papers/VISDrone/Chen_RRNet_A_Hybrid_Detector_for_Object_Detection_in_Drone-Captured_Images_ICCVW_2019_paper.pdf
 
## Feature Pyramid Network

- https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610
- https://arxiv.org/pdf/1612.03144.pdf

## Anchor Boxes

- https://towardsdatascience.com/neural-networks-intuitions-5-anchors-and-object-detection-fc9b12120830
- https://medium.com/@andersasac/anchor-boxes-the-key-to-quality-object-detection-ddf9d612d4f9
- https://www.youtube.com/watch?v=0frKXR-2PBY

## Receptive Fields

- https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807

## RetinaMask

- https://blog.zenggyu.com/en/post/2019-01-07/beyond-retinanet-and-mask-r-cnn-single-shot-instance-segmentation-with-retinamask/#fnref1
- https://towardsdatascience.com/instance-segmentation-using-mask-r-cnn-7f77bdd46abd
- https://arxiv.org/pdf/1901.03353.pdf
- https://www.youtube.com/watch?v=g7z4mkfRjI4

## Retinanet Exemplar Implementation

- https://medium.com/data-from-the-trenches/object-detection-with-deep-learning-on-aerial-imagery-2465078db8a9

## Aiskyeye Dataset

- https://arxiv.org/pdf/1804.07437.pdf

## VisDrone2019 Paper
 
- https://openaccess.thecvf.com/content_ICCVW_2019/papers/VISDrone/Du_VisDrone-DET2019_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ICCVW_2019_paper.pdf

## VisDrone2018 Paper
 
- http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Zhu_VisDrone-DET2018_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ECCVW_2018_paper.pdf

## Loss Functions

- https://www.ine.pt/revstat/pdf/rs070102.pdf
- https://arxiv.org/pdf/1511.08861.pdf
- https://mlblr.com/includes/mlai/index.html#yolov2

## Feature Visualization

- https://buzzrobot.com/using-t-sne-to-visualise-how-your-deep-model-thinks-4ba6da0c63a0
- https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
- https://ieeexplore.ieee.org/document/8402455
- https://github.com/cvondrick/ihog
