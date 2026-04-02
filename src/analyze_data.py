from pathlib import Path
import json
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

root_directory = Path(r'data/raw_data')
path_to_split_json = Path(r'splits/v1.json') # первая версия разбиения на выборки

# работаем со стандартными форматами изображений
EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def average_hash(image, hash_size=8):
    image = image.convert('L').resize((hash_size, hash_size))
    arr = np.asarray(image)
    return arr > arr.mean()


def hamming_distance(hash1, hash2):
    return int(np.sum(hash1 != hash2))


with open(path_to_split_json, 'r', encoding='utf-8') as f:
    split = json.load(f)

images = []
class_directories = sorted([label for label in root_directory.iterdir() if label.is_dir()])

for class_directory in class_directories:
    for image_path in sorted(class_directory.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in EXTENSIONS:
            images.append(image_path)

print('Всего изображений:', len(images))

# посчитаем распределение форматов, размеров и классов
format_counter = Counter()
size_counter = Counter()
class_counter = Counter()

# посчитаем также статистики яркости, контраста и цветовых каналов
brightness_values = []
contrast_values = []
channel_means = []

# для оценки разнообразия внутри классов посчитаем хэши изображений и расстояния между ними
class_to_hashes = defaultdict(list)

for current_image in images:
    image = Image.open(current_image).convert('RGB')
    class_name = current_image.parent.name

    class_counter[class_name] += 1
    format_counter[current_image.suffix.lower()] += 1
    size_counter[image.size] += 1

    arr = np.asarray(image) / 255
    grayscale = np.asarray(image.convert('L')) / 255

    brightness_values.append(grayscale.mean())
    contrast_values.append(grayscale.std())
    channel_means.append(arr.mean(axis=(0, 1)))

    image_average_hash = average_hash(image)
    class_to_hashes[class_name].append(image_average_hash)

print('Формат изображений:')
for format, amount in sorted(format_counter.items()):
    print(format, '\t', amount)

print('Размер изображений:')
for size, amount in sorted(size_counter.items()):
    print(size, '\t', amount)

print('Распределение изображений по классам:')
for class_name, amount in sorted(class_counter.items()):
    print(class_name, '\t', amount)

for split_name in ['train', 'validation', 'test']:
    split_counter = Counter(Path(relative_path).parent.name for relative_path in split[split_name])
    print(f'В {split_name} выборке {len(split[split_name])} изображений')
    for split_name, amount in sorted(split_counter.items()):
        print(split_name, '\t', amount)

brightness_values = np.array(brightness_values)
contrast_values = np.array(contrast_values)
channel_means = np.array(channel_means)

print('Средняя яркость:', brightness_values.mean(), '\n',
      'Средний контраст:', contrast_values.mean(), '\n', 
      'Средние значения цветовых каналов', channel_means.mean(axis=0), '\n',
      'Стандартное отклонение яркости:', brightness_values.std(), '\n',
      'Стандартное отклонение контраста:', contrast_values.std(), '\n',
      'Стандартное отклонение цветовых каналов:', channel_means.std(axis=0)
      )

# чтобы понять, насколько похожи изображения, посчитаем расстояния между хэшами внутри каждого класса
for class_name, hashes in sorted(class_to_hashes.items()):
    distances = []
    max_pairs = min(len(hashes) - 1, 100)
    for i in range(max_pairs):
        distance = hamming_distance(hashes[i], hashes[i + 1])
        distances.append(distance)
    distances = np.array(distances)
    print('Для класса', class_name, ':\n',
          'минимальное расстояние между хэшами изображений:', distances.min(), '\n',
          'максимальное расстояние между хэшами изображений:', distances.max(), '\n',
          'среднее расстояние между хэшами изображений:', distances.mean()
          )
