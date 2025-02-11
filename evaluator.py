import sys
import logging
from tqdm import tqdm
from pycocotools.coco import COCO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from model import postprocessing_instance_segmentation
from dataloader import build_val_dataloader


logger = logging.getLogger(__file__)
logger.setLevel('INFO')
logger.addHandler(sys.stdout)
    

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
            gtResults['masks'] = torch.as_tensor([cocoGt.annToMask(gtAnnObj) for gtAnnObj in gtAnnObjs],dtype=torch.bool, device=device)
            gtResults['labels'] = torch.as_tensor([gtAnnObj['category_id'] for gtAnnObj in gtAnnObjs],dtype=torch.long, device=device)

            evaluator.update([pred_result], [gtResults])
    
    summarize = evaluator.compute()
    return summarize

