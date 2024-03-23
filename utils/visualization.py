# visualization.py
import matplotlib.pyplot as plt
import numpy as np

def show_images(image_t1, image_t2, converted=None):
    """Show image with landmarks for a batch of samples."""
    imgs = [image_t1, image_t2]
    titles = ['Image T1', 'Image T2']
    if converted is not None:
        imgs.append(converted)
        titles.append('Converted Image')
        
    fig, axs = plt.subplots(1, len(imgs), figsize=(15, 5))
    for i, img in enumerate(imgs):
        img = img.squeeze()  # Remove channel dimension if present
        axs[i].imshow(img, cmap='gray')
        axs[i].title.set_text(titles[i])
        axs[i].axis('off')
    plt.show()
