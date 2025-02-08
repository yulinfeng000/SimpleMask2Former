import numpy as np
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from cocoapi import ChromoConcatCOCO

class CoCoDataset(Dataset):
    COCOAPI = ChromoConcatCOCO

    def __init__(self, datasets, transforms:"TransformPipeline"=None):
        self.cocoapi = self.COCOAPI(datasets)
        self.ids = list(sorted(self.cocoapi.imgs.keys()))
        self.transforms = transforms

    def _load_image(self, img_id: int):
        imgObj = self.cocoapi.loadImgs(img_id)[0]
        return cv2.cvtColor(cv2.imread(imgObj['file_path']),cv2.COLOR_BGR2RGB)
    
    def _load_target(self, img_id: int):
        annObjs = self.cocoapi.loadAnns(self.cocoapi.getAnnIds(img_id))
        masks = np.stack([self.cocoapi.annToMask(annObj) for annObj in annObjs], axis=0)
        labels = np.asarray([annObj["category_id"] for annObj in annObjs], dtype="int64")
        bboxes = np.asarray([annObj["bbox"] for annObj in annObjs], dtype="float32")
        return {"masks": masks, "labels": labels, "bboxes": bboxes}

    def __getitem__(self, index):
        img_id = self.ids[index]
        sample = self._load_image(img_id)
        target = self._load_target(img_id)
        sample, targets = self.transforms(
            sample,
            target
        )
        return sample, targets

    def __len__(self):
        return len(self.ids)  


def _collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets

train_augmentation = A.Compose(
    [
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.CLAHE(),
                A.ColorJitter(),
                A.Rotate(limit=(-15, 15)),
            ]
        ),
        A.RandomScale(scale_limit=(0.5, 1.5)),
        A.Resize(1024, 1024),
        A.Normalize(),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
)

val_augmentation = A.Compose(
    [
        A.Resize(1024, 1024),
        A.Normalize(),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
)
    


class TransformPipeline:
    def __init__(self, augmentation):
        self.augmentation:A.Compose = augmentation
    

    def __call__(self, sample, targets):
        results = self.augmentation(image=sample, bboxes=targets["bboxes"], labels=targets["labels"], masks=targets['masks'])
        image   = torch.as_tensor(results['image'],dtype=torch.float32).permute(2,0,1)
        bboxes  = torch.as_tensor(results['bboxes'],dtype=torch.float32)
        masks   = torch.as_tensor(results['masks'],dtype=torch.float32)
        labels  = torch.as_tensor(results['labels'],dtype=torch.int64)
        return image, dict(boxes=bboxes,masks=masks,labels=labels)
    
def build_dataloader(datasets,batch_size=1, is_train=True):
    if is_train:
        dataloader = DataLoader(
            CoCoDataset(datasets,transforms=TransformPipeline(train_augmentation)),
            shuffle=True,batch_size=batch_size,collate_fn=_collate_fn
        )
    else:
        dataloader = DataLoader(
            CoCoDataset(datasets,transforms=TransformPipeline(val_augmentation)),
            shuffle=True, batch_size=batch_size, collate_fn=_collate_fn
        )

    return dataloader