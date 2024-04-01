
import os
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

TYPE_NONE   = 0
TYPE_OTHER  = 1
TYPE_TIN    = 2
TYPE_THATCH = 3


def main():
    train_df, test_df = load_clean_metadata()
    create_training_label_files(train_df)


def load_clean_metadata():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # category_id as supplied is blank for images with no dwellings
    train_df["category_id"].fillna(TYPE_NONE, inplace=True)

    return train_df, test_df


def create_training_label_files(train_df):
    image_dict = {}
    for row in train_df.itertuples():
        if row.image_id not in image_dict:
            image_dict[row.image_id] = []
        image_dict[row.image_id].append(row)
    
    re_brackets_commas = re.compile("(\[|\]|,)")
    os.makedirs(labels_folder, exist_ok=True)

    for image_id, row_list in image_dict.items():
        textfile_name = image_id + ".txt"
        textfile_path = os.path.join(labels_folder, textfile_name)
        with open(textfile_path, "w") as fd:
            for row in row_list:
                if row.category_id != TYPE_NONE:
                    # Supplied bounding boxes in correct x, y, width, height
                    # order already in Python list format
                    bbox_str = str(row.bbox)
                    bbox_str = re_brackets_commas.sub("", bbox_str)
                    fd.write("%d %s\n" % (row.category_id, bbox_str))


if __name__ == "__main__":
    main()
