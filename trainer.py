import sys
import logging
import torch
logger = logging.getLogger(__file__)
logger.setLevel('INFO')
logger.addHandler(sys.stdout)


def train_one_epoch(epoch, model, dataloader, optimizer, scaler, criterion):
    model.train()
    num_steps = len(dataloader)
    with torch.autocast('cuda'):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            images, targets = batch
            images = images.cuda()
            gt_classes = [t["labels"].cuda() for t in targets]
            gt_masks = [t["masks"].cuda() for t in targets]
            pred_logits, pred_masks = model(images)
            loss = criterion(pred_logits, pred_masks, gt_classes, gt_masks)
            total_loss = sum(loss.values())
            logger.info(f"[epoch {epoch}] step:{i}/{num_steps}, loss dice: {loss['loss_dice']}, loss mask:{loss['loss_mask']}, loss cls:{loss['loss_ce']}, total loss:{total_loss}")
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()