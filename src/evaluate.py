import torch

from sklearn.metrics import f1_score

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    targets = []
    predictions = []
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        prediction = logits.argmax(dim=1)
        
        correct += (prediction == y).sum().item()
        total += y.numel()
        targets.extend(y.cpu().tolist())
        predictions.extend(prediction.cpu().tolist())

    accuracy = correct / total
    macro_f1 = f1_score(targets, predictions, average='macro')
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
    }
