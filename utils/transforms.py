# transforms.py
from torchvision import transforms

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_t1, image_t2 = sample['image_t1'], sample['image_t2']

        # Swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_t1 = image_t1.transpose((2, 0, 1))
        image_t2 = image_t2.transpose((2, 0, 1))
        return {'image_t1': torch.from_numpy(image_t1),
                'image_t2': torch.from_numpy(image_t2)}

# Add more transformations as needed
