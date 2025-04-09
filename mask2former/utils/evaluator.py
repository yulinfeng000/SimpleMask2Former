import sys
import logging
from tqdm import tqdm
from pycocotools.coco import COCO
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from accelerate import Accelerator
from accelerate.utils import tqdm as atqdm
from ..mask2former import postprocessing_instance_segmentation

logger = logging.getLogger("mask2former")


@torch.no_grad()
def coco_evaluate(model, dataloader, device):
    metric_ap = MeanAveragePrecision(iou_type="segm", backend="faster_coco_eval")
    model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        images, targets = batch
        images = images.to(device)
        metainfos = [t["metainfo"] for t in targets]
        with torch.no_grad():
            outputs = model(images)
            pred_logits = outputs["pred_logits"]
            pred_masks = outputs["pred_masks"]

            pred_results = postprocessing_instance_segmentation(
                pred_logits, pred_masks, metainfos
            )

            dtResults = []
            gtResults = []
            for target, pred_result in zip(targets, pred_results):
                gtResult = {}
                gtResult["masks"] = target["masks"].to(device=device, dtype=torch.bool)
                gtResult["labels"] = target["labels"].to(
                    device=device, dtype=torch.long
                )
                gtResults.append(gtResult)
                dtResults.append(pred_result)

            metric_ap.update(dtResults, gtResults)

    summarize = metric_ap.compute()
    return summarize


@torch.no_grad()
def accelerator_coco_evaluate(accelerator: Accelerator, model, dataloader, device):
    metric_ap = MeanAveragePrecision(iou_type="segm", backend="faster_coco_eval")
    model.eval()
    for i, batch in enumerate(atqdm(dataloader)):
        images, targets = batch
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            outputs = accelerator.gather_for_metrics(outputs)
            targets = accelerator.gather_for_metrics(targets)

            pred_logits = outputs["pred_logits"]
            pred_masks  = outputs["pred_masks"]
            metainfos = [t["metainfo"] for t in targets]

            dtResults = postprocessing_instance_segmentation(
                pred_logits, pred_masks, metainfos
            )

            gtResults = []
            for target in targets:
                gtResult = {}
                gtResult["masks"] = target["masks"].to(device=device, dtype=torch.bool)
                gtResult["labels"] = target["labels"].to(
                    device=device, dtype=torch.long
                )
                gtResults.append(gtResult)

            metric_ap.update(dtResults, gtResults)

    summarize = metric_ap.compute()
    return summarize
