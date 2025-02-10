import logging
import sys
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from model import (
    Mask2Former,
    ResNet,
    VisionTransformer,
    ResNet,
    SimpleFPN,
    MSDeformAttnPixelDecoder,
    TransformerDecoder,
)
from criterion import SetCriterion
from matcher import HungarianMatcher
from dataloader import build_train_dataloader
from cocoapi import ChromoConcatCOCO
from evaluator import coco_evaluate

logger = logging.getLogger(__file__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel("INFO")

if __name__ == "__main__":

    num_classes = 1
    device = "cuda:3"

    mask2former = (
        Mask2Former(
            # backbone=VisionTransformer(
            #     patch_size=16,
            #     embed_dim=256,
            #     depth=1,  # 12
            #     num_heads=4,
            #     global_attn_indexes=[2, 5, 8, 11],
            #     window_size=7,
            #     use_rel_pos=True,
            #     img_size=1024,
            # ),
            backbone=ResNet((3, 4, 6, 3)),
            # backbone=VisionTransformer(
            #     patch_size=16,
            #     embed_dim=256,
            #     depth=1,  # 12
            #     num_heads=4,
            #     global_attn_indexes=[2, 5, 8, 11],
            #     window_size=7,
            #     use_rel_pos=True,
            #     img_size=1024,
            # ),
            backbone=ResNet((3, 4, 6, 3)),
            neck=SimpleFPN(
                backbone_channel=2048,
                in_channels=[512, 1024, 2048, 2048],
                backbone_channel=2048,
                in_channels=[512, 1024, 2048, 2048],
                out_channels=256,
                num_outs=4,
                norm_layer="layernorm2d",
            ),
            pixel_decoder=MSDeformAttnPixelDecoder([256, 256, 256, 256], 256, 256),
            #pixel_decoder=MultiScalePixelDecoder([256, 256, 256, 256], 256, 256),
            transformer_decoder=TransformerDecoder(256, 256, num_classes=num_classes,num_queries=100),
        )
        .to(device)
        .train()
    )

    train_dataloader = build_train_dataloader(
        ChromoConcatCOCO(
            [
            dict(
                img_root="/shared/data/chromo_coco/cropped_datasets/allcrop-segm-coco/train",
                ann_file="/shared/data/chromo_coco/cropped_datasets/allcrop-segm-coco/annotations/chromosome_train.json",
            )
        ]),
        batch_size=2
    )

    val_coco = ChromoConcatCOCO([
       dict(
                img_root="/shared/data/chromo_coco/cropped_datasets/allcrop-segm-coco/val",
                ann_file="/shared/data/chromo_coco/cropped_datasets/allcrop-segm-coco/annotations/chromosome_val.json",
            ) 
    ])

    criterion = SetCriterion(
        num_classes,
        HungarianMatcher(num_points=112*112),
        num_points=112*112
    ).to(device)

    optimizer = AdamW(mask2former.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler(device)

    num_steps = len(train_dataloader)

    with torch.autocast(device):
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            images, targets = batch
            images = images.to(device)
            gt_classes = [t["labels"].to(device) for t in targets]
            gt_masks = [t["masks"].to(device) for t in targets]
            metainfos = [t['metainfo'] for t in targets]
            pred_logits, pred_masks = mask2former(images)
            loss = criterion(pred_logits, pred_masks, gt_classes, gt_masks)
            total_loss = sum(loss.values())
            logger.info(f"[epoch 0] step:{i}/{num_steps}, loss dice: {loss['loss_dice']}, loss mask:{loss['loss_mask']}, loss cls:{loss['loss_ce']}, total loss:{total_loss}")
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()


    mask2former.eval()
    with torch.no_grad():
        eval_results = coco_evaluate(mask2former,val_coco, device)
        print(eval_results)