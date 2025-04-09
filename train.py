import logging
from datetime import datetime
import sys
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from mask2former import (
    Mask2Former,
    MSDeformAttnPixelDecoder,
    MultiScaleMaskedTransformerDecoder,
)
from mask2former.backbones.resnet import resnet50
from mask2former.datasets import (
    build_train_dataloader,
    build_val_dataloader,
    build_coco_dataset,
)
from mask2former.utils.criterion import SetCriterion
from mask2former.utils.matcher import HungarianMatcher
from mask2former.utils.evaluator import coco_evaluate
from mask2former.utils.original_ckpt_loader import original_resnet50_ckpt_loader
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    num_classes = 80 # background + chromosome, +1 for the no_object
    device = "cuda:1"
    start_epoch = 0
    total_epochs = 100
    base_lr = 1e-5
    batch_size = 4
    output_dir = os.path.join(
        "outputs", "mask2former-resnet50", datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("mask2former")
    tb_logger = SummaryWriter(log_dir=output_dir)

    mask2former = Mask2Former(
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
        backbone=resnet50(pretrained=True),
        neck=None,
        pixel_decoder=MSDeformAttnPixelDecoder(in_channels=[256, 512, 1024, 2048],in_strides=[4,8,16,32]),
        # pixel_decoder=MultiScalePixelDecoder([256, 256, 256, 256], 256, 256),
        transformer_decoder=MultiScaleMaskedTransformerDecoder(
            256, num_classes=num_classes, num_queries=100
        ),
    )

    original_resnet50_ckpt_loader(mask2former, "model_final_3c8ec9.pkl")
    mask2former.to(device)

    train_dataset = build_coco_dataset(
        img_root="/shared/data/chromo_coco/cropped_datasets/6kv2_dj_mcls/train_v2_6000",
        ann_file="/shared/data/chromo_coco/cropped_datasets/6kv2_dj_mcls/annotations_category_ordered/train_v2_6000.json",
    )

    val_dataset = build_coco_dataset(
        img_root="/shared/data/chromo_coco/cropped_datasets/6kv2_dj_mcls/val_500",
        ann_file="/shared/data/chromo_coco/cropped_datasets/6kv2_dj_mcls/annotations_category_ordered/val_500.json",
        is_train=False,
    )

    train_dataloader = build_train_dataloader(
        train_dataset, batch_size=batch_size, num_workers=4
    )

    val_dataloader = build_val_dataloader(
        val_dataset, batch_size=batch_size, num_workers=4
    )

    criterion = SetCriterion(
        num_classes,
        HungarianMatcher(num_points=112 * 112),
        num_points=112 * 112,
    ).to(device)

    optimizer = AdamW(mask2former.parameters(), lr=base_lr, weight_decay=0.05)
    lr_scheduler =  MultiStepLR(optimizer, [35,65])
    scaler = torch.amp.GradScaler()

    num_steps = len(train_dataloader)

    for epoch in range(start_epoch, total_epochs):

        # train loop
        mask2former.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            images, targets = batch
            images = images.to(device)

            for t in targets:
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t[k] = v.to(device)

            with torch.autocast(device_type="cuda"):
                outputs = mask2former(images)
                losses = criterion(outputs, targets)

                for k in losses.keys():
                    if "dice" in k:
                        losses[k] *= criterion.weight_dict["loss_dice"]
                        tb_logger.add_scalar(f"loss/{k}", losses[k].item(), epoch * num_steps + i)

                    elif "mask" in k:
                        losses[k] *= criterion.weight_dict["loss_mask"]
                        tb_logger.add_scalar(f"loss/{k}", losses[k].item(), epoch * num_steps + i)

                    elif "ce" in k:
                        losses[k] *= criterion.weight_dict["loss_ce"]
                        tb_logger.add_scalar(f"loss/{k}", losses[k].item(), epoch * num_steps + i)

                total_loss = sum(losses.values())
                tb_logger.add_scalar("loss/total_loss", total_loss.item(), epoch * num_steps + i)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"[epoch {epoch}] step:{i}/{num_steps}, loss dice: {losses['loss_dice']}, loss mask:{losses['loss_mask']}, loss cls:{losses['loss_ce']}, total loss:{total_loss}"
                )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        lr_scheduler.step()
        tb_logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # # save model state dict
        torch.save(
            mask2former.state_dict(),
            os.path.join(output_dir, f"mask2former_{epoch}.pth"),
        )
        if (epoch + 1) % 5 == 0:  # evaluate every 5 epochs
            summarize = coco_evaluate(mask2former, val_dataloader, device)
            logger.info(summarize)
            tb_logger.add_scalar("val/map", summarize["map"], epoch)
            tb_logger.add_scalar("val/map_75", summarize["map_75"], epoch)
            tb_logger.add_scalar("val/map_50", summarize["map_50"], epoch)
            tb_logger.add_scalar("val/map_medium", summarize["map_medium"], epoch)
            tb_logger.add_scalar("val/map_large", summarize["map_large"], epoch)
            tb_logger.add_scalar("val/mar_1", summarize["mar_1"], epoch)
            tb_logger.add_scalar("val/mar_10", summarize["mar_10"], epoch)
            tb_logger.add_scalar("val/mar_100", summarize["mar_100"], epoch)
            tb_logger.add_scalar("val/mar_small", summarize["mar_small"], epoch)
            tb_logger.add_scalar("val/mar_medium", summarize["mar_medium"], epoch)
            tb_logger.add_scalar("val/mar_large", summarize["mar_large"], epoch)
    
