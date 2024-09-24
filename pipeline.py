from segment import CustomSegmentor
from ensemble import TeacherGuidedSegmentation
from ultralytics import YOLO
import argparse
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./datasets/test/images")
    parser.add_argument(
        "--yolo_path",
        type=str,
        default="./food-seg-modified/yolov8x-seg-modified/weights/last.pt",
    )
    parser.add_argument("--sam_type", type=str, default="vit_h")
    parser.add_argument(
        "--sam_checkpoint_path", type=str, default="./sam_vit_h_4b8939.pth"
    )
    parser.add_argument("--save_path", type=str, default="./output")
    args = parser.parse_args()

    yolo = YOLO(args.yolo_path)

    custom_color_list = np.load("./color_list.npy")
    print(custom_color_list.shape)
    
    img_paths = glob(args.image_folder + "/*.jpg")[157:]
    for img_path in tqdm(img_paths):
        img_no = img_path.split("/")[-1].split(".")[0]
        save_folder_path = os.path.join(args.save_path, img_no)

        custom_segmentor = CustomSegmentor(img_path, save_folder_path)        
        custom_segmentor.sam_segmentation()   # sam inference
        
        org_w, org_h = Image.open(img_path).size    # original image size

        yolo_results = yolo(img_path, verbose=False)    # yolo inference
        assert len(yolo_results[0].boxes) == len(yolo_results[0].masks.data)

        if not yolo_results[0].masks:
            continue
        
        seg_color_idx = (
            yolo_results[0].boxes.cls
            if yolo_results[0].boxes
            else range(len(yolo_results[0].masks.data))
        ).detach().cpu().numpy().astype(np.int16)
        
        seg_colors = [tuple(custom_color_list[x]) for x in seg_color_idx]

        yolo_masks = yolo_results[0].masks.data
        
        if len(yolo_masks) > 1:
            detected_classes = yolo_results[0].boxes.cls.detach().cpu().tolist()
            cls_masks = np.zeros(yolo_results[0].masks[0].data.shape, dtype=np.int16)
            for i in range(len(yolo_results[0].masks)):
                cls_mask = yolo_results[0].masks[i].data.detach().cpu().numpy()
                cls_mask[cls_mask == 1] = int(detected_classes[i])
                cls_masks[cls_masks == 0] = cls_mask[cls_masks == 0] 
        else:
            detected_class = yolo_results[0].boxes.cls.detach().cpu().tolist()[0]
            cls_masks = yolo_results[0].masks[0].data.detach().cpu().numpy().astype(np.int16)
            cls_masks[cls_masks == 1] = int(detected_class)
            
        cls_masks = cls_masks.squeeze(axis=0)
        cls_masks = cv2.resize(cls_masks, (org_w, org_h))
        np.save(os.path.join(save_folder_path, "yolo_mask.npy"), cls_masks)

        custom_segmentor.yolo_segmentation(yolo_masks, seg_colors)
        custom_ensembler = TeacherGuidedSegmentation(save_folder_path, img_path)
        
        ensembled_masks = custom_ensembler.ensemble_segmentation()
        custom_ensembler.visualization_save(ensembled_masks, custom_color_list)
