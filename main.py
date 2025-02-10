import logging
import sys
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from model import (
    Mask2Former,
    VisionTransformer,
    SimpleFPN,
    MSDeformAttnPixelDecoder,
    TransformerDecoder,
    MultiScalePixelDecoder,
)
from criterion import SetCriterion, HungarianMatcher
from dataloader import build_dataloader

logger = logging.getLogger(__file__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel("INFO")

if __name__ == "__main__":
    num_classes = 1
    mask2former = (
        Mask2Former(
            backbone=VisionTransformer(
                patch_size=16,
                embed_dim=256,
                depth=1,  # 12
                num_heads=4,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=7,
                use_rel_pos=True,
                img_size=1024,
            ),
            neck=SimpleFPN(
                backbone_channel=256,
                in_channels=[64, 128, 256, 256],
                out_channels=256,
                num_outs=4,
                norm_layer="layernorm2d",
            ),
            pixel_decoder=MSDeformAttnPixelDecoder([256, 256, 256, 256], 256, 256),
            #pixel_decoder=MultiScalePixelDecoder([256, 256, 256, 256], 256, 256),
            transformer_decoder=TransformerDecoder(256, 256, num_classes=num_classes),
        )
        .cuda()
        .train()
    )

    dataloader = build_dataloader(
        [
            dict(
                img_root="/shared/data/chromo_coco/cropped_datasets/allcrop-segm-coco/train",
                ann_file="/shared/data/chromo_coco/cropped_datasets/allcrop-segm-coco/annotations/chromosome_train.json",
            )
        ],
        batch_size=2
    )

    criterion = SetCriterion(
        num_classes,
        HungarianMatcher(num_points=112*112),
        num_points=112*112
    ).cuda()

    optimizer = AdamW(mask2former.parameters(), lr=1e-5)
    lr_scheduler = StepLR(optimizer, step_size=100)
    scaler = torch.amp.GradScaler() 
    
    with torch.autocast('cuda'):

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            images, targets = batch
            images = images.cuda()
            gt_classes = [t["labels"].cuda() for t in targets]
            gt_masks = [t["masks"].cuda() for t in targets]

            pred_logits, pred_masks = mask2former(images)

            loss = criterion(pred_logits, pred_masks, gt_classes, gt_masks)
            total_loss = sum(loss.values())
            print("loss :",total_loss)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
    