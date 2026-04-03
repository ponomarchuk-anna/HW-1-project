import json
import re

from pathlib import Path

root_directory = Path(r'data/raw_data')
path_to_split_json = Path(r'splits/v2.json') # вторая версия разбиения на выборки

# фиксируем для воспроизводимости сид
SEED = 42

# работаем со стандартными форматами изображений
EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# фиксируем пропорции для разбиения на выборки 
train_size = 0.7
validation_size = 0.15
test_size = 0.15

def get_number_from_file_name(path_to_file):
    match = re.search(r'(\d+)', path_to_file.stem)
    if match is None:
        return 1000000
    return int(match.group(1))

if not root_directory.exists():
    raise FileNotFoundError(f'Directory {root_directory} not found')
else:
    images_by_class = {}
    class_directories = sorted([label for label in root_directory.iterdir() if label.is_dir()])

    for class_directory in class_directories:
        class_name = class_directory.name
        class_images = [image_path for image_path in class_directory.iterdir() if image_path.is_file() and image_path.suffix.lower() in EXTENSIONS]
        class_images = sorted(class_images, key=get_number_from_file_name)
        images_by_class[class_name] = class_images

    class_names = sorted(images_by_class.keys())
    indices = {class_name: index for index, class_name in enumerate(class_names)}
    
    train_paths = []
    validation_paths = []
    test_paths = []

    for class_name, class_images in images_by_class.items():
        train_last = int(train_size * len(class_images))
        validation_last = int((train_size + validation_size) * len(class_images))
        
        train = class_images[ : train_last]
        validation = class_images[train_last : validation_last]
        test = class_images[validation_last : ]

        train_paths.extend([p.relative_to(root_directory).as_posix() for p in train])
        validation_paths.extend([p.relative_to(root_directory).as_posix() for p in validation])
        test_paths.extend([p.relative_to(root_directory).as_posix() for p in test])

    split_information = {
        'seed': SEED,
        'root_directory': root_directory.as_posix(),
        'ratios': {
            'train': train_size,
            'validation': validation_size,
            'test': test_size,
        },
        'class_names': class_names,
        'class_indices': indices,
        'train': train_paths,
        'validation': validation_paths,
        'test': test_paths,
    }

    path_to_split_json.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_split_json, 'w', encoding='utf-8') as f:
        json.dump(split_information, f)
