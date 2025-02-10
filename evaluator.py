import sys
import logging
from tqdm import tqdm
import rlemasklib as rle
from pycocotools.coco import COCO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pycocotools.mask as maskutils
import numpy as np
import torch
from model import postprocessing_instance_segmentation
from dataloader import build_val_dataloader
logger = logging.getLogger(__file__)
logger.setLevel('INFO')
logger.addHandler(sys.stdout)

def binary_mask_to_rle(mask):
    """
    Converts given binary mask of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        mask (`torch.Tensor` or `numpy.array`):
            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target
            segment_id or class_id.
    Returns:
        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE
        format.
    """
    if isinstance(mask,torch.Tensor):
        mask = mask.numpy()

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)

def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

    return run_length_encodings

def create_cocoeval_ann_info(
    annotation_id: int,
    image_id: int,
    category_id: int,
    binary_mask: np.ndarray,
    score: float,
):
    binary_mask_encoded = maskutils.encode(
        np.asfortranarray(binary_mask.astype(np.uint8))
    )
    area = maskutils.area(binary_mask_encoded)
    bbox = maskutils.toBbox(binary_mask_encoded)

    temp = rle.encode(binary_mask, compressed=False)
    segmentation = dict(counts=temp["ucounts"].tolist(), size=temp["size"])

    return dict(
        id=annotation_id,
        image_id=image_id,
        category_id=category_id,
        iscrowd=0,
        area=float(area),
        bbox=bbox.tolist(),
        segmentation=segmentation,
        width=binary_mask.shape[1],
        height=binary_mask.shape[0],
        score=score,
    )

def convert_pred_results_to_coco_format(pred_masks, pred_classes, pred_scores, metainfos, ann_id_generator):
    annotations = []
    for masks, labels, scores, metainfo in zip(pred_masks, pred_classes, pred_scores, metainfos):
        coco_img_id = metainfo['img_id']
        
        for mask,label,score in zip(masks, labels, scores):
            annotations.append(create_cocoeval_ann_info(
                next(ann_id_generator),
                image_id=coco_img_id,
                category_id=label.numpy(),
                binary_mask=mask.numpy(),
                score=score.numpy()
            ))
    
    return annotations
    

def coco_evaluate(model,cocoGt:COCO, device):
    evaluator = MeanAveragePrecision(iou_type="segm",backend="faster_coco_eval")
    dataloader = build_val_dataloader(cocoGt, batch_size=3)
    model.eval()
    for i ,batch in enumerate(tqdm(dataloader)):
        images, targets = batch
        images = images.to(device)
        metainfos = [t['metainfo'] for t in targets]

        with torch.no_grad():
            pred_logits, pred_masks = model(images) 
            pred_results = postprocessing_instance_segmentation(pred_logits, pred_masks, metainfos)
        for metainfo, pred_result in zip(metainfos, pred_results):
            gtResults = {}

            gtAnnObjs = cocoGt.loadAnns(metainfo['img_id'])
            gtResults['masks'] = torch.as_tensor([cocoGt.annToMask(gtAnnObj) for gtAnnObj in gtAnnObjs],dtype=torch.bool)
            gtResults['labels'] = torch.as_tensor([gtAnnObj['category_id'] for gtAnnObj in gtAnnObjs],dtype=torch.long)

            evaluator.update([pred_result], [gtResults])
    
    summarize = evaluator.compute()
    return summarize

