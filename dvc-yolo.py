
import os
import pandas as pd
from ultralytics import YOLO


data_folder    = "data"
images_folder  = "data/Images/Images"
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


def load_clean_metadata():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # category_id as supplied is blank for images with no dwellings
    train_df["category_id"].fillna(TYPE_NONE, inplace=True)

    return train_df, test_df


if __name__ == "__main__":
    main()
