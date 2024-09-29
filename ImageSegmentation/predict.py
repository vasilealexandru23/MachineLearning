import os
import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import unet 

def visualize_data(N, image_list, mask_list):
    img = imageio.imread(image_list[N])
    mask = imageio.imread(mask_list[N])

    print(image_list.shape)
    print(mask_list.shape)

    fig, arr = plt.subplots(1, 2, figsize=(14,10))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[1].imshow(mask[:,:,0])
    arr[1].set_title('Segmentation')
    plt.show()

def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')

    return input_image, input_mask

def load_data():
    path = ''
    image_path = os.path.join(path, './dataA/dataA/CameraRGB/')
    mask_path = os.path.join(path, './dataA/dataA/CameraSeg/')

    image_list_orig = os.listdir(image_path)
    image_list = [image_path+i for i in image_list_orig]
    mask_list = [mask_path+i for i in image_list_orig]

    visualize_data(0, image_list, mask_list)

    # Split data
    image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
    mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)

    image_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

    # Preprocess Data

    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)

    return processed_image_ds

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def main():
    processed_image_ds = load_data()

    myunet = unet.unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23)
    myunet.summary()

    # Compile
    myunet.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

    train_dataset = processed_image_ds.cache().shuffle(500).batch(32)
    history = myunet.fit(train_dataset, epochs=5)

if __name__ == '__main__':
    main()
