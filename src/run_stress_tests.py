import csv
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import models

from dataset import ASLDataset
from seed import set_seed
from stress_transformations import all_transformations, StressTransformations


configuration_path = 'configs/baseline.yaml' # путь до конфигурационного файла
path_to_result = Path(r'runs/stress_tests_v1')

@torch.no_grad() # функция для оценки результатов
def evaluate_stress(model, loader, dataset, device, transformation_name, alpha):
    model.eval()

    targets = []
    predictions = []
    confidences = [] # максимальная вероятность
    margins = [] # разница между вероятностью предсказанного класса и вторым по вероятности классом
    rows = []
    index_to_class = {index: class_name for class_name, index in dataset.class_indices.items()}
    current_position = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)

        confidence, prediction = probabilities.max(dim=1)
        best_probabilities = torch.topk(probabilities, k=2, dim=1).values
        margin = best_probabilities[:, 0] - best_probabilities[:, 1]

        batch_size = y.size(0)
        batch_paths = dataset.images[current_position: current_position + batch_size]
        current_position += batch_size

        targets.extend(y.cpu().tolist())
        predictions.extend(prediction.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
        margins.extend(margin.cpu().tolist())

        for image_path, true_label, predicted_label, current_confidence, current_margin in zip(
            batch_paths,
            y.cpu().tolist(),
            prediction.cpu().tolist(),
            confidence.cpu().tolist(),
            margin.cpu().tolist()
        ):
            rows.append({
                'transformation': transformation_name,
                'alpha': alpha,
                'image_path': image_path,
                'true_label': true_label,
                'true_class': index_to_class[true_label],
                'predicted_index': predicted_label,
                'predicted_class': index_to_class[predicted_label],
                'is_correct': int(true_label == predicted_label),
                'confidence': current_confidence,
                'margin': current_margin,
            })

    accuracy = sum(int(true == predicted) for true, predicted in zip(targets, predictions)) / len(targets)
    macro_f1 = f1_score(targets, predictions, average='macro')
    risk = 1 - accuracy

    # посчитаем также самые частые ошибки
    confusion_counter = Counter()
    for true_label, predicted_label in zip(targets, predictions):
        if true_label != predicted_label:
            confusion_counter[(index_to_class[true_label], index_to_class[predicted_label])] += 1

    # и уверенность с отступом на них
    error_confidences = [
        current_confidence for current_confidence, true_label, predicted_label
        in zip(confidences, targets, predictions)
        if true_label != predicted_label
    ]

    error_margins = [
        current_margin for current_margin, true_label, predicted_label
        in zip(margins, targets, predictions)
        if true_label != predicted_label
    ]

    summary = {
        'transformation': transformation_name,
        'alpha': alpha,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'risk': risk,
        'amount_of_errors': int(sum(true != pred for true, pred in zip(targets, predictions))),
        'mean_confidence': float(sum(confidences) / len(confidences)),
        'mean_margin': float(sum(margins) / len(margins)),
        'mean_confidence_on_errors': float(sum(error_confidences) / len(error_confidences)) if len(error_confidences) > 0 else None,
        'mean_margin_on_errors': float(sum(error_margins) / len(error_margins)) if len(error_margins) > 0 else None,
        'top_confusions': [
            {
                'true_class': true_class,
                'predicted_class': predicted_class,
                'count': amount,
            }
            for (true_class, predicted_class), amount in confusion_counter.most_common(10)
        ],
    }
    return summary, rows

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

cfg = load_config(configuration_path)

data_cfg = cfg['data']
train_cfg = cfg['train']
model_cfg = cfg['model']
eval_cfg = cfg['eval']
output_cfg = cfg['output']

set_seed(train_cfg['seed'])
path_to_result.mkdir(parents=True, exist_ok=True)
checkpoint_path = Path(output_cfg['run_directory']) / 'best.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model(model_cfg['num_classes']).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])

all_results = []
all_rows = []

just_transform = StressTransformations( # случай без стресс-тестов вообще
    image_size=data_cfg['image_size'],
    mean=data_cfg['normalize']['mean'],
    std=data_cfg['normalize']['std'],
    scenario_name=None,
    alpha=None
)

just_dataset = ASLDataset(
    data_directory=data_cfg['root'],
    path_to_split_json=data_cfg['split'],
    set_name='test',
    transform=just_transform
)

just_loader = DataLoader(
    just_dataset,
    batch_size=train_cfg['batch_size'],
    shuffle=False,
    num_workers=0
)

just_summary, just_rows = evaluate_stress(
    model=model,
    loader=just_loader,
    dataset=just_dataset,
    device=device,
    transformation_name='no_transformations',
    alpha='no_transformations'
)

all_results.append(just_summary)
all_rows.extend(just_rows)

print('Результаты, полученные без стресс-тестов\n',
      'accuracy =', just_summary['accuracy'],
      'macro_f1 =', just_summary['macro_f1'],
      'amount of errors =', just_summary['num_errors'],
      sep='\n'
)

# теперь пройдём по всем сценариям и альфам
for transformation_number, (transformation_name, transformation_information) in enumerate(all_transformations.items()):
    print('\nРезультаты для стресс-теста', transformation_name)

    for alpha in transformation_information['alphas']:
        set_seed(train_cfg['seed'])

        current_transform = StressTransformations(
            image_size=data_cfg['image_size'],
            mean=data_cfg['normalize']['mean'],
            std=data_cfg['normalize']['std'],
            scenario_name=transformation_name,
            alpha=alpha
        )

        current_dataset = ASLDataset(
            data_directory=data_cfg['root'],
            path_to_split_json=data_cfg['split'],
            set_name='test',
            transform=current_transform
        )

        current_loader = DataLoader(
            current_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            num_workers=0
        )

        current_summary, current_rows = evaluate_stress(
            model=model,
            loader=current_loader,
            dataset=current_dataset,
            device=device,
            scenario_name=transformation_name,
            alpha=alpha
        )

        all_results.append(current_summary)
        all_rows.extend(current_rows)
        
        print('Параметр равен', alpha,
              'accuracy =', current_summary['accuracy'],
              'macro_f1 =', current_summary['macro_f1'],
              'amount of errors =', current_summary['num_errors'],
              sep='\n'
            )

with open(path_to_result / 'stress_results.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f)

with open(path_to_result / 'predictions_for_stress_tests.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            'transformation',
            'alpha',
            'image_path',
            'true_index',
            'true_class',
            'predicted_index',
            'predicted_class',
            'is_correct',
            'confidence',
            'margin',
        ]
    )
    writer.writeheader()
    writer.writerows(all_rows)
