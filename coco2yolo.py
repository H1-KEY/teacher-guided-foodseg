import os
import json


def convert(org_ann: str, img_path: str, save_folder: str):
    img_file_names = os.listdir(img_path)

    with open(org_ann) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = [_["name"] for _ in coco["categories"]]
    categories_ = {_["id"]: _["name"] for _ in coco["categories"]}
    categories_reverse_dict = {v: k for k, v in enumerate(categories)}

    for ann in annotations:
        image = next(img for img in images if (img["id"] == ann["image_id"]))
        if image["file_name"] not in img_file_names:
            continue
        width, height = image["width"], image["height"]
        category_id = ann["category_id"]
        category = categories_[category_id]
        category_id = categories_reverse_dict[category] # convert category id to aicrowd format
        filename = image["file_name"]
        label_filename = filename.split(".jpg")[0]
        label_path = os.path.join(save_folder, f"{label_filename}.txt")
        with open(label_path, "w") as f:
            segmentation_points_list = []
            for segmentation in ann["segmentation"]:
                # Check if any element in segmentation is a string
                if any(isinstance(point, str) for point in segmentation):
                    continue  # Skip this segmentation if it contains strings
                segmentation_points = [
                    str(
                        float(point) / (width - 1)
                        if i % 2 == 0
                        else float(point) / (height - 1)
                    )
                    for i, point in enumerate(segmentation)
                ]
                segmentation_points_list.append(" ".join(segmentation_points))
                segmentation_points_string = " ".join(segmentation_points_list)
                line = "{} {}\n".format(category_id, segmentation_points_string)
                f.write(line)
                segmentation_points_list.clear()

if __name__ == "__main__":
    train_img_path = "./datasets/train/images"
    train_org_ann = "./datasets/val/new_ann.json"
    train_save_folder = "./datasets/val/labels"

    val_org_ann = "./datasets/val/new_ann.json"
    val_save_folder = "./datasets/val/labels"
    val_img_path = "./datasets/val/images"
    
    convert(train_org_ann, train_img_path, train_save_folder)   # convert train json (coco) to yolo format
    convert(val_org_ann, val_img_path, val_save_folder) # convert val json (coco) to yolo format