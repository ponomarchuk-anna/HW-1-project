import numpy as np

from io import BytesIO
from PIL import Image, ImageEnhance
from torchvision import transforms


def apply_brightness(image, b):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(b + 1)

def apply_rotation(image, angle):
    return image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

def apply_gaussian_noise(image, sigma):
    array = np.asarray(image, dtype=np.float32) / 255
    noise = np.random.normal(loc=0.0, scale=sigma, size=array.shape)
    array = np.clip(array + noise, 0.0, 1.0)
    return Image.fromarray((array * 255).astype(np.uint8))

def apply_jpeg_compression(image, quality):
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=int(quality))
    buffer.seek(0)
    compressed = Image.open(buffer).convert('RGB')
    return compressed.copy()

all_transformations = {
    'brightness': {
        'alpha_name': 'b',
        'alphas': [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        'transform_fn': apply_brightness,
    },
    'rotation': {
        'alpha_name': 'angle',
        'alphas': [-30, -15, -5, 0, 5, 15, 30],
        'transform_fn': apply_rotation,
    },
    'gaussian_noise': {
        'alpha_name': 'sigma',
        'alphas': [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.10],
        'transform_fn': apply_gaussian_noise,
    },
    'jpeg_compression': {
        'alpha_name': 'quality',
        'alphas': [100, 90, 80, 50, 30, 20],
        'transform_fn': apply_jpeg_compression,
    },
}

class StressTransformations:
    def __init__(self, image_size, mean, std, transformation_name, alpha):
        self.image_size = tuple(image_size)
        self.mean = mean
        self.std = std
        self.transformation_name = transformation_name
        self.alpha = alpha

        # стандартные преобразования исходного изображения
        self.resize = transforms.Resize(self.image_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def __call__(self, image):
        image = image.convert('RGB')
        image = self.resize(image)

        if self.transformation_name is not None and self.alpha is not None:
            transformation = all_transformations[self.transformation_name]
            image = transformation['transform_fn'](image, self.alpha)

        image = self.to_tensor(image)
        image = self.normalize(image)
        return image
