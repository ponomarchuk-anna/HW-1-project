import json

from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split

root_directory = Path(r'data/raw_data')
path_to_split_json = Path(r'splits/v1.json') # первая версия разбиения на выборки

# фиксируем для воспроизводимости сид
SEED = 42

# работаем со стандартными форматами изображений
EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# фиксируем пропорции для разбиения на выборки 
train_size = 0.7
validation_size = 0.15
test_size = 0.15

if not root_directory.exists():
    raise FileNotFoundError(f'Directory {root_directory} not found')
else:
    images = []
    labels = []
    class_directories = sorted([label for label in root_directory.iterdir() if label.is_dir()])

    for class_directory in class_directories:
        class_name = class_directory.name
        for image_path in sorted(class_directory.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in EXTENSIONS:
                relative_path = image_path.relative_to(root_directory).as_posix()
                images.append(relative_path)
                labels.append(class_name)

    train_paths, other_paths, train_labels, other_labels = train_test_split(images, labels, test_size=(validation_size + test_size), random_state=SEED, stratify=labels)
    validation_paths, test_paths, validation_labels, test_labels = train_test_split(other_paths, other_labels, test_size=(1.0 - validation_size / (validation_size + test_size)), random_state=SEED, stratify=other_labels)

    class_names = sorted(set(labels))
    indices = {class_name: index for index, class_name in enumerate(class_names)}

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
