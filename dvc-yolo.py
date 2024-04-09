#!/bin/env python

# (c) Charlie Wartnaby 2024

import numpy as np
import os
import PIL
import pandas as pd
import random
import re
import shutil
import sys
from ultralytics import YOLO

# Supplied data layout
data_folder    = "data"
images_folder  = os.path.join(data_folder, "Images/Images")
labels_folder  = os.path.join(data_folder, "Images/labels")
train_filename = "Train.csv"
test_filename  = "Test.csv"
train_path     = os.path.join(data_folder, train_filename)
test_path      = os.path.join(data_folder, test_filename)

# Specific layout required for YOLO to work
yolo_folder          = os.path.join(data_folder, "yolo")
yolo_images_folder   = os.path.join(yolo_folder, "images")
yolo_labels_folder   = os.path.join(yolo_folder, "labels")
# then 'train' and 'val' beneath each of those; also:
yolo_test_folder     = os.path.join(yolo_folder, "test")

runs_folder          = "runs" # relative to ~/.config/Ultralytics/settings.yaml path
detect_output_folder = os.path.join(runs_folder, "detect")
submission_file      = "submission.csv"

do_create_label_files     = True
do_copy_train_val_to_yolo = True
do_copy_test_to_yolo      = True
do_train                  = True
do_inference_test         = True
do_save_annotated_images  = False

TYPE_NONE   = 0
TYPE_OTHER  = 1
TYPE_TIN    = 2
TYPE_THATCH = 3

train_proportion    = 0.99 # maximised now for competition test not training validation
train_epochs        = 30
debug_max_test_imgs = 0 # zero to do all


def main():
    train_df, test_df = load_clean_metadata()

    train_image_dict = collate_image_labels(train_df)

    if do_create_label_files: 
        create_training_label_files(train_image_dict)

    train_ids, val_ids = train_val_split(train_image_dict, train_proportion)

    if do_copy_train_val_to_yolo:
        create_copy_train_val_yolo(train_ids, val_ids)

    test_ids = list(test_df["image_id"])

    if do_copy_test_to_yolo:
        copy_test_yolo(test_ids)

    if do_train:
        run_training(train_df, train_epochs)

    if do_inference_test:
        run_prediction(test_ids)


def load_clean_metadata():
    print("Loading metadata...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # category_id as supplied is blank for images with no dwellings
    train_df["category_id"].fillna(TYPE_NONE, inplace=True)

    return train_df, test_df


def collate_image_labels(train_df):
    """The dataset has zero or more rows with the same image ID, each corresponding
    to one bounding box. Collate the bounding boxes for each training image here."""
    image_dict = {}
    for row in train_df.itertuples():
        if row.image_id not in image_dict:
            image_dict[row.image_id] = []
        image_dict[row.image_id].append(row)
    return image_dict


def create_training_label_files(image_dict):
    """Create a text file for each training image listing the known
    classes and bounding boxes of the labelled objects. See
    https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format """

    print("Creating label files...")
    if os.path.exists(labels_folder):
        # Start with clean slate
        shutil.rmtree(labels_folder)
    os.makedirs(labels_folder, exist_ok=True)

    for image_id, row_list in image_dict.items():
        imagefile_name = image_id + ".tif"
        imagefile_path = os.path.join(images_folder, imagefile_name)
        image = PIL.Image.open(imagefile_path)
        width, height = image.size
        textfile_name = image_id + ".txt"
        textfile_path = os.path.join(labels_folder, textfile_name)
        with open(textfile_path, "w") as fd:
            for row in row_list:
                if row.category_id != TYPE_NONE:
                    # Supplied bounding boxes in x, y, width, height; but in pixels,
                    # and we need centre coordinates not top left corner. In Yolo,
                    # top left of image is still (0, 0) but bottom right is (1, 1).
                    x, y, dx, dy = eval(row.bbox)
                    x = (x + dx / 2) / width
                    y = (y + dy / 2) / height
                    dx /= width
                    dy /= height
                    # YOLO docs say classes should be zero-based. Does it matter if
                    # we have an unused class 0? TODO but not worrying for now.
                    fd.write("%d %f %f %f %f\n" % (row.category_id, x, y, dx, dy))


def train_val_split(train_image_dict, train_proportion):
    random.seed(6119) # Competition requires deterministic output
    all_ids = list(train_image_dict.keys())
    random.shuffle(all_ids)
    split_idx = int(len(all_ids) * train_proportion)
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]
    return train_ids, val_ids


def create_copy_train_val_yolo(train_ids, val_ids):
    print("Copying image and label files into YOLO directory layout...")
    if os.path.exists(yolo_folder):
        # Start with clean slate
        shutil.rmtree(yolo_folder)
    for parent in [yolo_images_folder, yolo_labels_folder]:
        for category in ['train', 'val']:
            directory = os.path.join(parent, category)
            os.makedirs(directory)
    for id in train_ids:
        copy_image_and_label_files(id, 'train')
    for id in val_ids:
        copy_image_and_label_files(id, 'val')


def copy_test_yolo(test_ids):
    print("Copying test images to YOLO directory for inference...")
    os.makedirs(yolo_test_folder, exist_ok=True)
    for id in test_ids:
        image_filename = id + ".tif"
        image_src_path = os.path.join(images_folder, image_filename)
        image_dest_path = os.path.join(yolo_test_folder, image_filename)
        shutil.copy(image_src_path, image_dest_path)

        
def copy_image_and_label_files(id, category):
    image_filename = id + ".tif"
    label_filename = id + ".txt"
    image_src_path = os.path.join(images_folder, image_filename)
    label_src_path = os.path.join(labels_folder, label_filename)
    image_dest_path = os.path.join(yolo_images_folder, category, image_filename)
    label_dest_path = os.path.join(yolo_labels_folder, category, label_filename)
    shutil.copy(image_src_path, image_dest_path)
    shutil.copy(label_src_path, label_dest_path)


def run_training(train_df, epochs):
    print("Running training on supplied labelled data...")
    model = YOLO('yolov8n.pt')
    results = model.train(data='dvc-dataset.yaml', epochs=epochs, imgsz=640)


def run_prediction(test_ids):
    # id_0b0pzumg4rbl.tif is good first example
    print("Running prediction on test images...")

    if debug_max_test_imgs > 0:
        test_ids = test_ids[:debug_max_test_imgs]

    # Try opening file before doing anything heavy in case it fails
    with open(submission_file, "w") as fd:
        fd.write("image_id,Target\n")
        img_paths = [os.path.join(yolo_test_folder, test_id + ".tif") for test_id in test_ids]
        latest_train_dir = get_latest_dir(detect_output_folder, "train")
        model_filename = os.path.join(latest_train_dir, "weights/best.pt")
        model = YOLO(model_filename)
        results = model.predict(img_paths, save=do_save_annotated_images)
        for i, result in enumerate(results):
            classes = result.boxes.cls
            np_classes = classes.numpy()
            for target_class in [TYPE_OTHER, TYPE_TIN, TYPE_THATCH]:
                num_instances = np.count_nonzero(np_classes == target_class)
                fd.write(f"{test_ids[i]}_{target_class},{num_instances}\n")


def get_latest_dir(parent_dir, subdir_base_name):
    """Find latest folder of train, train2, train3 etc"""

    parent_dir_contents = os.listdir(parent_dir)
    matcher = re.compile(subdir_base_name + r"([0-9]+)?")
    highest_suffix = -1
    best_name = ""
    for name in parent_dir_contents:
        match_obj = matcher.match(name)
        if match_obj:
            suffix_str = match_obj.group(1)
            if suffix_str:
                suffix = int(suffix_str)
            else:
                suffix = 0
            if suffix > highest_suffix:
                highest_suffix = suffix
                best_name = name
    if not best_name:
        print(f"Error: can't find latest folder starting {subdir_base_name} in {parent_dir}")
        sys.exit(-1)

    return os.path.join(parent_dir, best_name)


if __name__ == "__main__":
    main()
