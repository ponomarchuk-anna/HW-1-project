import json
import torch
import torch.nn as nn
import yaml

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import models

from dataset import ASLDataset, train_transform
from evaluate import evaluate
from seed import set_seed

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    cfg = load_config('configs/baseline.yaml')
    data_cfg = cfg['data']
    train_cfg = cfg['train']
    model_cfg = cfg['model']
    eval_cfg = cfg['eval']
    output_cfg = cfg['output']

    set_seed(train_cfg['seed'])
    run_directory = Path(output_cfg['run_directory'])
    run_directory.mkdir(parents=True, exist_ok=True)

    transform = train_transform(data_cfg['image_size'], data_cfg['normalize']['mean'], data_cfg['normalize']['std'])

    train_dataset = ASLDataset(
        data_directory=data_cfg['root'],
        path_to_split_json=data_cfg['split'],
        set_name='train',
        transform=transform
    )

    validation_dataset = ASLDataset(
        data_directory=data_cfg['root'],
        path_to_split_json=data_cfg['split'],
        set_name='validation',
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'])
    validation_loader = DataLoader(validation_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(model_cfg['num_classes']).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = -1
    best_metrics = -1
    best_epoch = -1

    for epoch in range(train_cfg['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        
        validation_metrics = evaluate(model, validation_loader, device)
        if validation_metrics['accuracy'] > best_accuracy:
            best_accuracy = validation_metrics['accuracy']
            best_metrics = validation_metrics
            best_epoch = epoch
            torch.save({'model': model.state_dict(), 'best_epoch': best_epoch + 1, 'best_metrics': best_metrics, 'cfg': cfg}, run_directory / 'best.pt')

    result = {
        'best_epoch': best_epoch + 1,
        'best_validation_accuracy': best_metrics['accuracy'],
        'best_validation_macro_f1': best_metrics['macro_f1'],
        'operating_point': eval_cfg['operating_point'],
        'model_name': model_cfg['name'],
        'num_classes': model_cfg['num_classes'],
        'seed': train_cfg['seed']
    }

    with open( run_directory / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
