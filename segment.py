from ultralytics.utils import ops
import cv2
import numpy as np
import os
from typing import List
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class CustomSegmentor:
    def __init__(self, img_path: str, save_path: str):
        self.img_path = img_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def yolo_segmentation(
        self, yolo_masks: torch.Tensor, color_list: List, alpha: float = 0.5
    ) -> None:
        h, w = yolo_masks.data[0].detach().cpu().numpy().shape
        img = cv2.imread(self.img_path)
        org_img = img.copy()
        
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        color_tensor = (
            torch.tensor(color_list, device=yolo_masks.device, dtype=torch.float32)
            / 255.0
        )  # shape(n,3)
        color_tensor = color_tensor[:, None, None]  # shape(n,1,1,3)
        masks = yolo_masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = masks * (color_tensor * alpha)  # shape(n,h,w,3)

        inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

        img_tensor = (
            torch.as_tensor(img, dtype=torch.float16, device=masks.data.device)
            .permute(2, 0, 1)
            .flip(0)
            .contiguous()
            / 255.0
        )  # shape(3,h,w)

        img_tensor = img_tensor.flip(dims=[0])  # flip channel
        img_tensor = img_tensor.permute(1, 2, 0).contiguous()  # shape(h,w,3)
        img_tensor = img_tensor * inv_alpha_masks[-1] + mcs
        semantic_img = img_tensor * 255  # denormalization
        semantic_img = semantic_img.byte().cpu().numpy()

        img = ops.scale_image(semantic_img, org_img.shape)
        cv2.imwrite(os.path.join(self.save_path, "yolo_vis.png"), img)

    def sam_segmentation(
        self,
        sam_type: str = "vit_h",
        sam_checkpoint_path: str = "./sam_vit_h_4b8939.pth",
    ) -> None:
        # Load Segment Anything Model
        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint_path)
        sam = sam.to(device="cuda:0" if torch.cuda.is_available() else "cpu")
        sam.eval()
        sam_mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

        # Generate Segment Anything Masks
        img = cv2.imread(self.img_path)
        sam_masks = sam_mask_generator.generate(img)

        header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
        metadata = [header]
        os.makedirs(os.path.join(self.save_path, "sam_mask"), exist_ok=True)
        masks_array = []
        for i, mask_data in enumerate(sam_masks):
            mask = mask_data["segmentation"]
            masks_array.append(mask.copy())
            filename = f"{i}.png"
            cv2.imwrite(os.path.join(self.save_path, "sam_mask", filename), mask * 255)
            mask_metadata = [
                str(i),
                str(mask_data["area"]),
                *[str(x) for x in mask_data["bbox"]],
                *[str(x) for x in mask_data["point_coords"][0]],
                str(mask_data["predicted_iou"]),
                str(mask_data["stability_score"]),
                *[str(x) for x in mask_data["crop_box"]],
            ]
            row = ",".join(mask_metadata)
            metadata.append(row)

        masks_array = np.stack(masks_array, axis=0)
        np.save(os.path.join(self.save_path, "sam_mask", "masks.npy"), masks_array)
        metadata_path = os.path.join(self.save_path, "sam_metadata.csv")
        with open(metadata_path, "w") as f:
            f.write("\n".join(metadata))
