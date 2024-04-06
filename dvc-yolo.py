
import os
import PIL
import pandas as pd
import re
from ultralytics import YOLO


data_folder    = "data"
images_folder  = "data/Images/Images"
labels_folder  = "data/Images/labels"
train_filename = "Train.csv"
test_filename  = "Test.csv"
train_path     = os.path.join(data_folder, train_filename)
test_path      = os.path.join(data_folder, test_filename)

create_label_files = False

TYPE_NONE   = 0
TYPE_OTHER  = 1
TYPE_TIN    = 2
TYPE_THATCH = 3


def main():
    train_df, test_df = load_clean_metadata()
    if create_label_files: 
        create_training_label_files(train_df)
    train(train_df)

def load_clean_metadata():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # category_id as supplied is blank for images with no dwellings
    train_df["category_id"].fillna(TYPE_NONE, inplace=True)

    return train_df, test_df


def create_training_label_files(train_df):
    """Create a text file for each training image listing the known
    classes and bounding boxes of the labelled objects. See
    https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format """

    image_dict = {}
    for row in train_df.itertuples():
        if row.image_id not in image_dict:
            image_dict[row.image_id] = []
        image_dict[row.image_id].append(row)
    
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
                    x = (x + width / 2) / width
                    y = (y + height / 2) / height
                    dx /= width
                    dy /= height
                    # YOLO docs say classes should be zero-based. Does it matter if
                    # we have an unused class 0? TODO but not worrying for now.
                    fd.write("%d %f %f %f %f\n" % (row.category_id, x, y, dx, dy))


def train(train_df):
    model = YOLO('yolov8n.pt')
    results = model.train(data='dvc-dataset.yaml', epochs=1, imgsz=640)


if __name__ == "__main__":
    main()
