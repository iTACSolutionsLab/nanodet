from coco_assistant import COCO_Assistant

coco = COCO_Assistant(img_dir='images', ann_dir='annotations')
coco.merge()