import sys
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
this_dir = os.path.dirname(__file__)
coco = COCO('{}/../../data/COCO/annotations/person_keypoints_val2017.json'.format(this_dir))
coco_dets = coco.loadRes(sys.argv[1])
coco_eval = COCOeval(coco, coco_dets, "keypoints")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
