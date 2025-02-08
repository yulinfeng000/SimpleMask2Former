import os
import time
try:
    from typing import List, TypedDict
except ImportError:
    from typing_extensions import List, TypedDict
from collections import defaultdict
from itertools import count
from pycocotools.coco import COCO as OriginCoCo


class DatasetConfig(TypedDict):
    img_root: str
    ann_file: str

class CatMergeTool:
    def __init__(self):
        self._cat_ids = {}
        self._cats = {}
    
    def get_cat_id(self, cat: dict):
        uuid = "{0}:{1}".format(cat["supercategory"], cat["name"])
        cat_id = self._cat_ids.get(uuid, None)

        if not cat_id:
            cat_id = cat['id']
            self._cat_ids[uuid] = cat_id

            cat.update({"id": cat_id})
            self._cats[cat_id]=cat

        return self._cat_ids.get(uuid)

    def get_cats(self):
        return self._cats

class ConcatCOCO(OriginCoCo):

    def __init__(
        self, datasets: List[DatasetConfig]
    ) -> None:
        super().__init__()

        self.datasets = datasets
        print("loading annotations into memory...")
        tic = time.time()
        self.createIndex()
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        # for adapt mmdetection's coco api
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def createIndex(self) -> None:
        # create index
        print("creating index...")

        img_id_generator = count(1)
        ann_id_generator = count(1)

        anns, imgs, catMerge = {}, {}, CatMergeTool()

        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

        for dataset_info in sorted(self.datasets, key=lambda x: x["ann_file"]):

            ann_file = dataset_info["ann_file"]
            img_root = dataset_info["img_root"]

            _coco = OriginCoCo(ann_file)

            for imgObj in _coco.loadImgs(_coco.getImgIds()):
                new_img_id = next(img_id_generator)
                old_img_id = imgObj['id']
                imgObj["file_name"] = os.path.basename(imgObj["file_name"])
                imgObj["id"] = new_img_id
                imgObj["file_path"] = os.path.join(img_root, imgObj["file_name"])
                # update indexes
                imgs[new_img_id] = imgObj

                annObjs = _coco.loadAnns(_coco.getAnnIds(imgIds=old_img_id))

                for annObj in annObjs:
                    
                    new_ann_id = next(ann_id_generator)
                    annObj["id"] = new_ann_id
                    annObj["image_id"] = new_img_id
                    
                    new_cat_id = catMerge.get_cat_id(
                        _coco.loadCats(annObj["category_id"])[0]
                    )
                    annObj["category_id"] = new_cat_id

                    anns[new_ann_id] = annObj
                    imgToAnns[new_img_id].append(annObj)
                    catToImgs[new_cat_id].append(imgObj['id'])

        print("index created!")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = catMerge.get_cats()

        self.dataset["categories"] = list(self.cats.values())
        self.dataset["annotations"] = list(self.anns.values())
        self.dataset["images"] = list(self.imgs.values())

    # for follow mmdetection's code rule
    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)



class ChromoConcatCOCO(ConcatCOCO):


    def createIndex(self) -> None:
        # create index
        print("creating index...")

        img_id_generator = count(1)
        ann_id_generator = count(1)

        anns, imgs, cats = {}, {}, {1: dict(name="chromosome", id=1, supercategory="chromosome")}

        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

        for dataset_info in sorted(self.datasets, key=lambda x: x["ann_file"]):

            ann_file = dataset_info["ann_file"]
            img_root = dataset_info["img_root"]

            _coco = OriginCoCo(ann_file)

            for imgObj in _coco.loadImgs(_coco.getImgIds()):
                new_img_id = next(img_id_generator)
                old_img_id = imgObj['id']
                imgObj["file_name"] = os.path.basename(imgObj["file_name"])
                imgObj["id"] = new_img_id
                imgObj["file_path"] = os.path.join(img_root, imgObj["file_name"])
                # update indexes
                imgs[new_img_id] = imgObj

                annObjs = _coco.loadAnns(_coco.getAnnIds(imgIds=old_img_id))

                for annObj in annObjs:
                    
                    new_ann_id = next(ann_id_generator)
                    annObj["id"] = new_ann_id
                    annObj["image_id"] = new_img_id
                    
                    new_cat_id = 1
                    annObj["category_id"] = new_cat_id

                    anns[new_ann_id] = annObj
                    imgToAnns[new_img_id].append(annObj)
                    catToImgs[new_cat_id].append(imgObj['id'])

        print("index created!")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

        self.dataset["categories"] = list(self.cats.values())
        self.dataset["annotations"] = list(self.anns.values())
        self.dataset["images"] = list(self.imgs.values())