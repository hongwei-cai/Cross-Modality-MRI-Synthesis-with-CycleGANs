# visualization.py
import matplotlib.pyplot as plt
import numpy as np


def show_images(image_t1, image_t2, converted=None):
    """
    Visualizes a batch of images (T1, T2) with optional converted image.

    Args:
        image_t1 (np.ndarray): A numpy array representing the T1 image data.
        image_t2 (np.ndarray): A numpy array representing the T2 image data.
        converted (np.ndarray, optional): A numpy array representing the converted image 
                                          (e.g., after some processing).
    """

    # Prepare images and titles for plotting
    imgs = [image_t1, image_t2]
    titles = ['Image T1', 'Image T2']
    if converted is not None:
        imgs.append(converted)
        titles.append('Converted Image')

    # Create a figure and subplots for each image
    fig, axs = plt.subplots(1, len(imgs), figsize=(15, 5))  # 1 row, len(imgs) columns

    # Loop through images, formatting and plotting them
    for i, img in enumerate(imgs):
        # Remove channel dimension if it exists (assuming grayscale images)
        img = img.squeeze()
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(titles[i])  # Set the title for each subplot
        axs[i].axis('off')  # Turn off axes for cleaner visualization

    # Display the plot
    plt.show()
