import logging
from datetime import datetime
import sys
import os
import torch
from mask2former import (
    Mask2Former,
    MSDeformAttnPixelDecoder,
    MultiScaleMaskedTransformerDecoder,
)
from mask2former.backbones.resnet import resnet50
from mask2former.utils.matcher import HungarianMatcher
from mask2former.utils.criterion import SetCriterion
from utils.datasets import (
    build_train_dataloader,
    build_val_dataloader,
    build_coco_dataset,
)
from utils.evaluator import coco_evaluate
from utils.original_ckpt_loader import original_resnet50_ckpt_loader
from torch.utils.tensorboard import SummaryWriter
from mmengine.optim import build_optim_wrapper

if __name__ == "__main__":
    amp_enabled = True
    num_classes = 80 
    device = "cuda:0"
    start_epoch = 0
    total_epochs = 100
    base_lr = 1e-4 # 1e-5
    batch_size = 4
    output_dir = os.path.join(
        "outputs",
        "mask2former-resnet50-ep100",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    load_from = ""
    resume_from = ""

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
        backbone=resnet50(pretrained=True),
        neck=None,
        pixel_decoder=MSDeformAttnPixelDecoder(
            in_channels=[256, 512, 1024, 2048], in_strides=[4, 8, 16, 32]
        ),
        transformer_decoder=MultiScaleMaskedTransformerDecoder(
            256, num_classes=num_classes, num_queries=100
        ),
    )

    # 加载原始权重
    original_resnet50_ckpt_loader(mask2former, "model_final_3c8ec9.pkl")

    if load_from:
        mask2former.load_state_dict(
            torch.load(load_from, map_location=device)["model"], strict=True
        )
    
    train_dataset = build_coco_dataset(
        img_root="xxx/train",
        ann_file="xxx/train.json",
    )

    val_dataset = build_coco_dataset(
        img_root="xxx/val",
        ann_file="xxx/val.json",
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
    )
    
    num_steps = len(train_dataloader)

    optimizer_wrapper = build_optim_wrapper(
        mask2former,
        dict(
            clip_grad=dict(max_norm=0.01, norm_type=2),
            optimizer=dict(
                betas=(
                    0.9,
                    0.999,
                ),
                eps=1e-08,
                lr=base_lr,
                type="AdamW",
                weight_decay=0.05,
            ),
            paramwise_cfg=dict(
                custom_keys=dict(
                    backbone=dict(decay_mult=1.0, lr_mult=0.1),
                    level_embed=dict(decay_mult=0.0, lr_mult=1.0),
                    query_embed=dict(decay_mult=0.0, lr_mult=1.0),
                    query_feat=dict(decay_mult=0.0, lr_mult=1.0),
                ),
                norm_decay_mult=0.0,
            ),
            type="AmpOptimWrapper",
        ),
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_wrapper.optimizer,
        milestones=[70,90],
        gamma=0.1,
        last_epoch=-1,
    )

    if resume_from:
        logger.info(f"[NOTICE] train process will resume from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        mask2former.load_state_dict(ckpt["model"])
        optimizer_wrapper.load_state_dict(ckpt["optimizer_wrapper"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        # scaler.load_state_dict(ckpt["scaler"])

        for state in optimizer_wrapper.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = ckpt["epoch"]

    mask2former.to(device)
    criterion.to(device)

    for epoch in range(start_epoch, total_epochs):

        # train loop
        mask2former.train()
        for i, batch in enumerate(train_dataloader):
            tb_logger.add_scalar(
                "lr", optimizer_wrapper.optimizer.param_groups[0]["lr"], epoch * num_steps + i
            )
            images, targets = batch
            images = images.to(device)

            for t in targets:
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t[k] = v.to(device)

            with optimizer_wrapper.optim_context(mask2former):
                outputs = mask2former(images)
                losses = criterion(outputs, targets)

                for k in losses.keys():
                    if "dice" in k:
                        losses[k] *= criterion.weight_dict["loss_dice"]
                        tb_logger.add_scalar(
                            f"loss/{k}", losses[k].item(), epoch * num_steps + i
                        )

                    elif "mask" in k:
                        losses[k] *= criterion.weight_dict["loss_mask"]
                        tb_logger.add_scalar(
                            f"loss/{k}", losses[k].item(), epoch * num_steps + i
                        )

                    elif "ce" in k:
                        losses[k] *= criterion.weight_dict["loss_ce"]
                        tb_logger.add_scalar(
                            f"loss/{k}", losses[k].item(), epoch * num_steps + i
                        )

                total_loss = sum(losses.values())
                tb_logger.add_scalar(
                    "loss/total_loss", total_loss.item(), epoch * num_steps + i
                )

            if (i + 1) % 10 == 0:
                logger.info(
                    f"[epoch {epoch}] step:{i}/{num_steps}, loss dice: {losses['loss_dice']}, loss mask:{losses['loss_mask']}, loss cls:{losses['loss_ce']}, total loss:{total_loss}"
                )

            optimizer_wrapper.update_params(total_loss)
        lr_scheduler.step()

        torch.save(
            {
                "optimizer_wrapper": optimizer_wrapper.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "model": mask2former.state_dict(),
            },
            os.path.join(output_dir, f"mask2former_ckpt_{epoch}.pth"),
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
