from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import cv2

cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "model_final.pth"
predictor = DefaultPredictor(cfg)

image = cv2.imread("image.jpg")
outputs = predictor(image)

v = Visualizer(image[:, :, ::-1])
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Detectron2", v.get_image()[:, :, ::-1])
cv2.waitKey(0)
