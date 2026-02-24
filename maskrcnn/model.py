# model.py
# Mask R-CNN with ResNet50 FPN V2 backbone
# Pretrained on COCO — we replace only the classification heads

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model(num_classes):
    """
    num_classes = number of pear classes + 1 (background)
    e.g. Middle-Ripe, Ripe, Unripe = 3 classes → num_classes = 4

    Architecture:
        ResNet50 + FPN  (pretrained — feature extraction)
            ↓
        Region Proposal Network  (pretrained — find candidate regions)
            ↓
        Box Predictor Head  ← replaced for your classes
        Mask Predictor Head ← replaced for your classes
    """
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model   = maskrcnn_resnet50_fpn_v2(weights=weights)

    # Replace box classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask classifier head
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels, 256, num_classes
    )

    return model
