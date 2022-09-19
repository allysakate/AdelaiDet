"""
https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py

Balloon Sample:
https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=UkNbUzUOLYf0
balloon sample: {
    "34020010494_e5cb88e1c4_k.jpg1115004":{
        "fileref":"","size":1115004,
        "filename":"34020010494_e5cb88e1c4_k.jpg",
        "base64_img_data":"",
        "file_attributes":{},
        "regions":{"0":{
            "shape_attributes":{
                "name":"polygon",
                "all_points_x":[1020,1000,994,1003,1023,1050, ...],
                "all_points_y":[963,899,841,787,738,700,663, ...]
                },
            "region_attributes":{}
            }
        }
        },

MOTS: https://www.vision.rwth-aachen.de/page/mots
class_id: car 1, pedestrian 2
format: time_frame id class_id img_height img_width rle
mots sample: 1 2002 2 1080 1920 UkU\1`0RQ1>PoN\OVP1X1F=I3oSOTNlg0U2l ...

"""
import PIL.Image as Image
import numpy as np
import pycocotools.mask as rletools
import glob
import torch
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks, save_image
from torchvision.io import read_image
import os
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

class SegmentedObject:
    def __init__(self, mask, class_id, track_id, bbox, mask_bool):
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id
        self.bbox = bbox
        self.mask_bool = mask_bool

def load_image(mots_dir, group, filename, is_visualize= False, id_divisor=1000):
    img = np.array(Image.open(f"{mots_dir}/instances/{group}/{filename}.png"))
    obj_ids = np.unique(img)
    objects = []
    mask = np.zeros(img.shape, dtype=np.uint8, order="F")  # Fortran order needed for pycocos RLE tools
    for idx, obj_id in enumerate(obj_ids):
        if obj_id == 0:  # background
            continue

        mask.fill(0)
        pixels_of_elem = np.where(img == obj_id)
        mask[pixels_of_elem] = 1
        mask_bool = torch.from_numpy(mask.astype(bool))
        mask_t = torch.from_numpy(mask)
        mask_torch = mask_t.unsqueeze(0)
        bbox = masks_to_boxes(mask_torch).tolist()[0]
        bbox = list(map(int, masks_to_boxes(mask_torch)[0]))
        seg_obj = SegmentedObject(
            rletools.encode(mask),
            obj_id // id_divisor,
            obj_id,
            bbox,
            mask_bool
        )
        objects.append(seg_obj)
    return objects


def get_mots(folder_path, group, data_split):
    # Check whether the specified path exists or not
    images_path = os.path.join(folder_path, "new_images", group)
    if not os.path.exists(images_path):
        # Create a new directory because it does not exist 
        os.makedirs(images_path)
        print(f"Directory {images_path} is created!")

    dataset_dicts = []
    mask_imgs = glob.glob(os.path.join(folder_path, "instances", group, "*.png"))
    num_imgs = len(mask_imgs)
    num_train = int(num_imgs * 0.9)
    data_dict = {}
    for m_img in mask_imgs:
        basename, _ = os.path.splitext(os.path.basename(m_img))
        image_tensor = read_image(os.path.join(folder_path, "images", group, f"{basename}.jpg"))
        frame_id = int(basename)
        if (data_split=="train" and frame_id <= num_train) or (data_split=="val" and frame_id > num_train):
            objects = load_image(folder_path, group, basename)
            for seg_obj in objects:
                image_name = basename + ".jpg"
                # file_name = os.path.join(
                #     folder_path,
                #     "images",
                #     group,
                #     "img1",
                #     image_name
                # )
                class_id = seg_obj.class_id
                image_id = group + "_" + basename
                bbox = seg_obj.bbox
                mask_rle = seg_obj.mask["counts"].decode(encoding='UTF-8')
                if class_id in [1,2]:
                    if image_id in data_dict:
                        data_dict[image_id]["annotations"].append({
                            "bbox": bbox,
                            "segmentation": mask_rle,
                            "category_id": class_id,
                        })
                    else:
                        data_dict[image_id] = {}
                        data_dict[image_id]["image_name"] = image_name
                        data_dict[image_id]["group"] = group
                        data_dict[image_id]["frame_id"] = frame_id
                        data_dict[image_id]["height"] = seg_obj.mask["size"][0] 
                        data_dict[image_id]["width"] = seg_obj.mask["size"][1]
                        data_dict[image_id]["annotations"] = [
                            {
                                "bbox": bbox,
                                "segmentation": mask_rle,
                                "category_id": class_id
                            }
                        ]
                if class_id == 10:
                    image_tensor = draw_segmentation_masks(image_tensor, seg_obj.mask_bool, alpha=1, colors="gray")
            # save image
            image_seg = image_tensor.detach()
            image_seg = F.to_pil_image(image_seg)
            image_seg.save(os.path.join(images_path, f"{basename}.jpg"))

    for _, value in data_dict.items():
        dataset_dicts.append(value)
    print(f"group: {group}, last: {image_id}")
    return dataset_dicts

if __name__ == "__main__":
    data_split = "train"
    folder_path = "/media/catchall/starplan/Dissertation/Dataset/MOTSChallenge/train/"
    group_idx = {
        1: [1, "0002", "train"],
        2: [541, "0005", "train"],
        3: [1294, "0009", "train"],
        4: [1766, "0011", "train"],
        5: [2576, "0002", "val"],
        6: [2636, "0005", "val"],
        7: [2720, "0009", "val"],
        8: [2773, "0011", "val"],
    }
    for key in [1,2,3,4]:
        last_idx = group_idx[key][0]
        group = group_idx[key][1]
        data_split = group_idx[key][2]
        lines = []
        data_list = get_mots(folder_path, group, data_split)
        for idx, data in enumerate(data_list):
            image_id = idx + last_idx
            for ann in data["annotations"]:
                bbox_str = list(map(str, ann["bbox"]))
                new_line = [
                    data["group"],
                    str(image_id),
                    data["image_name"],
                    str(data["frame_id"]),
                    str(ann["category_id"]),
                    str(data["height"]),
                    str(data["width"])
                ]
                new_line.extend(bbox_str)
                new_line.append(ann["segmentation"])
                lines.append(new_line)
        print(lines[0][:-1])
        print(new_line[:-1])

        txt_file = os.path.join(folder_path, f"{data_split}_{group}.txt")
        with open(txt_file, 'w') as fp:
            for line in lines:
                STRING_TEXT = " ".join(line)
                fp.write(f"{STRING_TEXT}\n")

    # with open(f"{folder_path}/train.txt") as f:
    #     CNT = 1
    #     for line in f:
    #         if CNT == 1:
    #             flds = line.strip().split(" ")
    #             for field in flds:
    #                 print(field)
    #         CNT+=1
    out = os.path.join(folder_path, f"{data_split}.txt")
    filenames = ["0002", "0005", "0009", "0011"]
    with open(out, 'w') as outfile:
        for fname in filenames:
            txt_file = os.path.join(folder_path, f"{data_split}_{fname}.txt")
            with open(txt_file) as infile:
                for line in infile:
                    outfile.write(line)