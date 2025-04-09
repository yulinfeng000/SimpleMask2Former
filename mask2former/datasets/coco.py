from typing import Dict, List, Union
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from typing import Dict
import numpy as np
import torch
import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


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
        A.Normalize(
            mean=tuple(IMAGENET_DEFAULT_MEAN),
            std=tuple(IMAGENET_DEFAULT_STD),
        ),
    ]
)

val_augmentation = A.Compose(
    [
        A.Resize(1024, 1024),
        A.Normalize(
            mean=tuple(IMAGENET_DEFAULT_MEAN),
            std=tuple(IMAGENET_DEFAULT_STD),
        ),
    ]
)


class TransformPipeline:
    def __init__(self, augmentation):
        self.augmentation: A.Compose = augmentation

    def __call__(self, sample: np.ndarray, targets: Dict[str, np.ndarray]):
        results = self.augmentation(
            image=sample,
            masks=targets["masks"],
        )
        image = torch.as_tensor(results["image"], dtype=torch.float32).permute(2, 0, 1)
        masks = torch.as_tensor(results["masks"], dtype=torch.float32)
        labels = torch.as_tensor(targets["labels"], dtype=torch.long)
        return image, dict(masks=masks, labels=labels, metainfo=targets["metainfo"])


class CoCoDataset(Dataset):

    def __init__(self, img_root, coco, transforms=None):
        self.img_root = img_root
        self.cocoapi = coco
        self.img_ids = list(sorted(self.cocoapi.imgs.keys()))
        self.cat_ids = list(sorted(self.cocoapi.cats.keys()))
        self.cat_label_maps = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.transforms = transforms

    def _load_image(self, img_id: int):
        imgObj = self.cocoapi.loadImgs(img_id)[0]
        return cv2.cvtColor(cv2.imread(
            os.path.join(self.img_root, imgObj["file_name"])
        ), cv2.COLOR_BGR2RGB)

    def _load_target(self, img_id: int):
        imgObj = self.cocoapi.loadImgs(img_id)[0]
        annObjs = self.cocoapi.loadAnns(self.cocoapi.getAnnIds(img_id))
        masks = np.stack([self.cocoapi.annToMask(annObj) for annObj in annObjs], axis=0)
        labels = np.asarray(
            [self.cat_label_maps[annObj["category_id"]] for annObj in annObjs],
            dtype="int64",
        )
        bboxes = np.asarray([annObj["bbox"] for annObj in annObjs], dtype="float32")
        metainfo = dict(img_id=img_id, img_shape=(imgObj["height"], imgObj["width"]))
        return {
            "masks": masks,
            "labels": labels,
            "bboxes": bboxes,
            "metainfo": metainfo,
        }

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        sample = self._load_image(img_id)
        target = self._load_target(img_id)
        sample, targets = self.transforms(sample, target)
        return sample, targets

    def __len__(self):
        return len(self.img_ids)


def build_coco_dataset(
    img_root: str,
    ann_file: str,
    is_train=True,
    coco_class=None,
):
    if coco_class is None:
        from pycocotools.coco import COCO

    if is_train:
        transform = TransformPipeline(train_augmentation)

    else:
        transform = TransformPipeline(val_augmentation)

    return CoCoDataset(
        img_root,
        COCO(ann_file),
        transforms=transform,
    )
