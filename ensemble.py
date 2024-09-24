import numpy as np
import os
import cv2
from typing import List


class TeacherGuidedSegmentation:
    def __init__(
        self,
        save_path: str,
        img_path: str,
        category_txt: str = "category_info.txt",
        sam_mask_path: str = "sam_mask/masks.npy",
        sam_label_pred_filename: str = "sam_label_predictions.txt",
        yolo_mask_filename: str = "yolo_mask.npy",
        ensembled_mask_filename: str = "ensembled_mask.npy",
        ensembled_vis_filename: str = "ensembled_vis.png",
    ):
        self.category_txt = category_txt
        self.save_path = save_path
        self.img_path = img_path
        self.yolo_mask_filename = yolo_mask_filename
        self.ensembled_mask_filename = ensembled_mask_filename
        self.ensembled_vis_filename = ensembled_vis_filename
        self.category_list = self.load_categories()
        self.sam_mask_path = os.path.join(save_path, sam_mask_path)
        self.yolo_mask_path = os.path.join(save_path, yolo_mask_filename)
        self.sam_label_prediction_path = os.path.join(save_path, sam_label_pred_filename)
        self.yolo_mask_data = np.load(self.yolo_mask_path)
        self.sam_mask_data = np.load(self.sam_mask_path)
        
        
    def load_categories(self) -> List:
        with open(self.category_txt, "r") as f:
            category_lines = f.readlines()
            categories = [
                " ".join(line_data.split(" ")[1:]).strip()
                for line_data in category_lines
            ]
        return categories

    def predict_sam_label(self) -> None:
        shape_size = self.yolo_mask_data.shape[0] * self.yolo_mask_data.shape[1]
        with open(self.sam_label_prediction_path, "w") as f:
            f.write(
                "id,category_id,category_name,category_count_ratio,mask_count_ratio\n"
            )
            for i in range(self.sam_mask_data.shape[0]):
                single_mask = self.sam_mask_data[i]
                # single_mask_labels = pred_mask_img[single_mask]
                single_mask_labels = self.yolo_mask_data[single_mask]
                unique_values, counts = np.unique(
                    single_mask_labels, return_counts=True, axis=0
                )
                max_idx = np.argmax(counts)

                single_mask_category_label = unique_values[max_idx]
                count_ratio = counts[max_idx] / counts.sum()

                print(
                    f"{self.save_path}/sam_mask/{i} assigned label: [ {single_mask_category_label}, {self.category_list[single_mask_category_label]}, {count_ratio:.2f}, {counts.sum()/shape_size:.4f} ]"
                )
                f.write(
                    f"{i},{single_mask_category_label},{self.category_list[single_mask_category_label]},{count_ratio:.2f},{counts.sum()/shape_size:.4f}\n"
                )
        f.close()

    def ensemble_segmentation(
        self,
        num_class=324,
        area_thr=0,
        ratio_thr=0.5,
        top_k=80,
    ) -> np.ndarray:
        self.predict_sam_label()
        save_path = os.path.join(self.save_path, self.ensembled_mask_filename)

        f = open(self.sam_label_prediction_path, "r")
        category_info = f.readlines()[1:]
        category_area = np.zeros((num_class,))
        f.close()
        
        for info in category_info:
            label, area = int(info.split(",")[1]), float(info.split(",")[-1])
            category_area[label] += area

        category_info = sorted(
            category_info, key=lambda x: float(x.split(",")[-1]), reverse=True
        )
        category_info = category_info[:top_k]

        ensembled_masks = self.yolo_mask_data.copy()

        for info in category_info:
            idx, label, count_ratio, area = (
                info.split(",")[0],
                int(info.split(",")[1]),
                float(info.split(",")[3]),
                float(info.split(",")[4]),
            )
            if area < area_thr:
                continue
            if count_ratio < ratio_thr:
                continue
            sam_mask = self.sam_mask_data[int(idx)].astype(bool)
            assert (
                sam_mask.sum() / (sam_mask.shape[0] * sam_mask.shape[1]) - area
            ) < 1e-4
            ensembled_masks[sam_mask] = label
        np.save(save_path, ensembled_masks)
        return ensembled_masks
        
    def visualization_save(self, mask, color_list):
        ensembled_vis_path = os.path.join(self.save_path, self.ensembled_vis_filename)
        
        values = set(mask.flatten().tolist())
        final_masks = []
        label = []
        for v in values:
            final_masks.append((mask[:,:] == v, v))
        np.random.seed(42)
        if len(final_masks) == 0:
            return
        h, w = final_masks[0][0].shape[:2]
        result = np.zeros((h, w, 3), dtype=np.uint8) 
        for m, label in final_masks:
            if label == 0:
                continue
            result[m, :] = color_list[label] 
        image = cv2.imread(self.img_path)
        vis = cv2.addWeighted(image, 0.5, result, 0.5, 0) 
        cv2.imwrite(ensembled_vis_path, vis)
