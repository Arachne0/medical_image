import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from tqdm import tqdm
from torchvision import models
from functions.nih_loader import nih_loader
from functions.calculate_metrics import calculate_metrics

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
disease_names = [
    "Pneumonia", "Nodule", "Mass", "Infiltration", "Pneumothorax",
    "Edema", "Pleural_Thickening", "Fibrosis", "Effusion",
    "Consolidation", "Cardiomegaly", "Atelectasis", "Hernia", "Emphysema"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--pretrain', type=bool, default=False)
    args = parser.parse_args()

    wandb.init(project="NIH_Chest_X-ray",
               entity="hails",
               config=args.__dict__,
               name=f"ChestXray_{args.model}_lr:{args.lr}_Batch:{args.batch_size}"
               )

    train_loader, test_loader = nih_loader(batch_size=args.batch_size, num_workers=4, resize=True)

    if args.model == 'resnet50':
        model = models.resnet50(pretrained=args.pretrain)
        model.fc = nn.Linear(model.fc.in_features, 14)
        model = model.to(device)
    elif args.model == 'resnet101':
        model = models.resnet101(pretrained=args.pretrain)
        model.fc = nn.Linear(model.fc.in_features, 14)
        model = model.to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('start training')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.4f}')
        wandb.log({'Train Loss': avg_loss}, step=epoch + 1)

        test_model(model, test_loader, epoch, criterion)


def test_model(model, test_loader, epoch, criterion):
    model.eval()
    true_labels = []
    predicted_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5

            true_labels.append(labels.cpu().numpy())
            predicted_labels.append(preds.cpu().numpy())

    true_labels = np.concatenate(true_labels)
    predicted_labels = np.concatenate(predicted_labels)

    metrics = calculate_metrics(true_labels, predicted_labels)

    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')

    for i, (acc, sens, spec) in enumerate(zip(metrics['accuracy'], metrics['sensitivity'], metrics['specificity'])):
        print(f'{disease_names[i]}: Accuracy: {acc:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}')

    wandb.log({
        'Test Loss': avg_loss,
        **{f'Accuracy_{disease_names[i]}': acc for i, acc in enumerate(metrics['accuracy'])},
        **{f'Sensitivity_{disease_names[i]}': sens for i, sens in enumerate(metrics['sensitivity'])},
        **{f'Specificity_{disease_names[i]}': spec for i, spec in enumerate(metrics['specificity'])}
    }, step=epoch + 1)

if __name__ == '__main__':
    main()