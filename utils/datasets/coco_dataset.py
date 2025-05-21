from typing import Dict, List, Union
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from typing import Dict
import numpy as np
import torch
import albumentations as A


CHROMO_MEAN = np.array([241.72, 241.72, 241.72])
CHROMO_STD = np.array([21.32, 21.32, 21.32])

train_augmentation = A.Compose(
    [
        A.RandomBrightnessContrast(),
        # A.RandomGamma(),
        # A.CLAHE(),
        A.ColorJitter(),
        A.VerticalFlip(),
        A.HorizontalFlip(),
        A.Rotate(limit=(-15, 15)),
        A.RandomScale(scale_limit=(0.5, 1.0)),
        A.Resize(1024, 1024),
        # A.Normalize(mean=tuple(CHROMO_MEAN / 255), std=tuple(CHROMO_STD / 255)),
    ]
)

val_augmentation = A.Compose(
    [
        A.Resize(1024, 1024),
        # A.Normalize(mean=tuple(CHROMO_MEAN / 255), std=tuple(CHROMO_STD / 255)),
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

        return image, dict(
            masks=masks, labels=labels, metainfo=targets["metainfo"]
        )


class CoCoDataset(Dataset):

    def __init__(self, img_root, ann_file, transforms=None, cat_label_maps=None):
        from pycocotools.coco import COCO

        self.img_root = img_root
        self.cocoapi = COCO(ann_file)
        self.img_ids = list(sorted(self.cocoapi.imgs.keys()))
        self.cat_ids = list(sorted(self.cocoapi.cats.keys()))
        self.cat_label_maps = cat_label_maps or {
            cat_id: i for i, cat_id in enumerate(self.cat_ids)
        }
        self.transforms = transforms

    def _load_image(self, img_id: int):
        imgObj = self.cocoapi.loadImgs(img_id)[0]
        return cv2.cvtColor(
            cv2.imread(
                os.path.join(self.img_root, imgObj["file_name"]),
            ),
            cv2.COLOR_BGR2RGB,
        )

    def _load_target(self, img_id: int):
        imgObj = self.cocoapi.loadImgs(img_id)[0]
        annObjs = self.cocoapi.loadAnns(self.cocoapi.getAnnIds(img_id))
        

        masks  = []
        labels = []
        bboxes = []
        for annObj in annObjs:
            if annObj['segmentation']:
                masks.append(self.cocoapi.annToMask(annObj))

            if annObj['bbox']:
                bboxes.append(annObj['bbox'])

            labels.append(self.cat_label_maps[annObj["category_id"]])
            

        masks = np.asarray(masks, dtype='float32')
        labels = np.asarray(labels, dtype='int64')
        bboxes = np.asarray(bboxes,dtype='float32')

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
        if self.transforms:
            sample, target = self.transforms(sample, target)
        return sample, target

    def __len__(self):
        return len(self.img_ids)


def build_coco_dataset(
    img_root: Union[List, str],
    ann_file: Union[List, str],
    is_train=True,
):

    from torch.utils.data import ConcatDataset

    if isinstance(img_root, str):
        img_root = [img_root]
    if isinstance(ann_file, str):
        ann_file = [ann_file]

    if is_train:
        transform = TransformPipeline(train_augmentation)

    else:
        transform = TransformPipeline(val_augmentation)

    return ConcatDataset(
        [
            CoCoDataset(
                img_root=imgroot,
                ann_file=annfile,
                transforms=transform,
            )
            for imgroot, annfile in zip(img_root, ann_file)
        ]
    )
