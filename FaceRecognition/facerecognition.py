import numpy as np
import tensorflow as tf

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss function

    Arguments:
    y_true -- true labels
    y_pred -- list containing three objects:
                anchor -- the encodings for the anchor images
                positive -- the encodings for the positive images
                negative -- the encodings for the negative images
    
    Returns:
    loss -- real number, value of the loss
    """

    # Retrieve the anchor, positive and negative encodings from y_pred
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis = -1)
    # Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis = -1)
    # Subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

def load_model():
    json_file = open('keras-facenet-h5/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights('keras-facenet-h5/model.h5')

    return model

def img_to_encoding(image_path, model):
    # The pre-trained model expects inputs with the shape (160, 160, 3) and we got (96, 96, 3)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)

    return embedding / np.linalg.norm(embedding, ord=2)

def build_database(model):
    database = {}
    database["danielle"] = img_to_encoding("images/danielle.png", model)
    database["younes"] = img_to_encoding("images/younes.jpg", model)
    database["tian"] = img_to_encoding("images/tian.jpg", model)
    database["kian"] = img_to_encoding("images/kian.jpg", model)
    database["felix"] = img_to_encoding("images/felix.jpg", model)
    database["kevin"] = img_to_encoding("images/kevin.jpg", model)

    return database

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras
    
    Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
    """
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding - database[identity])
    
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
    
    return dist, door_open

def who_is_it(image_path, database, model):
    """
    Implements face recognition by finding who is the person on the image_path image.
    
    Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras
    
    Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
    """
    
    encoding = img_to_encoding(image_path, model)
    
    min_dist = 100
    threshold = 0.7
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - db_enc)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > threshold:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

if __name__ == '__main__':
    model = load_model()

    database = build_database(model)

    # Check if verification is working
    _, _ = verify("images/camera_0.jpg", "younes", database, model)
