import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

def load_data():
    return None

if __name__ == "__main__":
    # Load Data, set configuration variables
    item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

    num_item_features = item_train.shape[1] - 1  # remove movie id at train timent and ave rating during training
    num_item_features = item_train.shape[1] - 1  # remove movie id at train time
