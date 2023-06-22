import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.image import flip_left_right, flip_up_down, adjust_brightness
import matplotlib.pyplot as plt
import os
import cv2

def sort_by_name(arr, split_at):
    return sorted([f for f in arr if f.split(split_at)[0].isdigit()], key=lambda x: int(x.split(split_at)[0]))


def load_data(dir_path):
    # dir_path = '../satellite-roads/train/'
    image_path = os.path.join(dir_path, 'Image')
    mask_path = os.path.join(dir_path, 'Mask')

    images = os.listdir(image_path)
    masks = os.listdir(mask_path)

    sorted_images = sort_by_name(images, '.')
    sorted_masks = sort_by_name(masks, '.')

    return np.array(sorted_images), np.array(sorted_masks)


def preprocess_data(root_path, sorted_images, sorted_masks, input_size, augmented=False):

    images = []
    masks = []
    for img_file, mask_file in zip(sorted_images, sorted_masks):
        img = load_img(root_path + 'Image/' + img_file, target_size=input_size, color_mode='rgb')
        mask = load_img(root_path + 'Mask/' + mask_file, target_size=input_size, color_mode='grayscale')

        # Convert image and mask to arrays
        img_array = img_to_array(img)
        img_array = img_array / 255.0

        mask_array = img_to_array(mask, dtype=np.bool_)

        # Append images and masks to the lists
        images.append(img_array)
        masks.append(mask_array)

        if augmented:
            images.append(flip_left_right(img_array))
            masks.append(flip_left_right(mask_array))

            # images.append(flip_up_down(img_array))
            # masks.append(flip_up_down(mask_array))

    # Convert lists to numpy arrays
    images = np.array(images)
    masks = np.array(masks)

    return images, masks


def display_data(dir_path, image_paths, mask_paths, resize):

    fig, axes = plt.subplots(5, 3, figsize=(5, 10))

    # Iterate over the image and mask pairs and display them in subplots
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load the image and mask using your preferred method
        image = cv2.imread(dir_path + 'Image/' + image_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, resize)
        mask = cv2.imread(dir_path + 'Mask/' + mask_path)
        mask = cv2.resize(mask, resize)
        inverted_mask = cv2.bitwise_not(mask)


        # Plot the image and mask in the corresponding subplot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(image)
        axes[i, 2].imshow(tf.squeeze(inverted_mask), cmap='gray', alpha=0.4)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    return

