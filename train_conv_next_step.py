import torch.nn as nn
from transformers import ConvNextV2ForImageClassification
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor
import argparse
import wandb
import shutil
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

def load_model(checkpoint_path=None, num_classes=2, device="cpu"):
    if checkpoint_path:
        model = ConvNextV2ForImageClassification.from_pretrained(checkpoint_path)
    else:
        model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-22k-224")
        #model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-large-22k-224")
		
    if model.config.num_labels != num_classes:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        model.config.num_labels = num_classes

    model.to(device)
    return model

def compute_class_weights(dataset, device):
    targets = [label for _, label in dataset.imgs]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets
    )
    print(f"📊 Using class weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float).to(device)

def get_weighted_sampler(dataset):
    targets = [label for _, label in dataset.imgs]
    class_sample_counts = np.bincount(targets)
    weights = 1. / class_sample_counts[targets]
    weights = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

def train(dataset_dir, save_path, resume=None, epochs=5, lr=5e-5, batch_size=32, patience=3,
          balance_strategy="none"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="road-scene-classification",
        name="convnextv2-run",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "resume": resume is not None,
            "balance_strategy": balance_strategy
        }
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    full_dataset = datasets.ImageFolder(dataset_dir, transform=transform)

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Determine criterion and sampler based on balance strategy
    if balance_strategy == "weighted":
        class_weights = compute_class_weights(train_dataset.dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    elif balance_strategy == "oversample":
        criterion = nn.CrossEntropyLoss()
        sampler = get_weighted_sampler(train_dataset.dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    elif balance_strategy == "weighted_oversample":
        class_weights = compute_class_weights(train_dataset.dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        sampler = get_weighted_sampler(train_dataset.dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = load_model(resume, num_classes=2, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Resume optimizer state if available
    if resume and os.path.exists(f"{resume}/optimizer.pt"):
        optimizer.load_state_dict(torch.load(os.path.join(resume, "optimizer.pt")))

    best_f1 = 0.0
    epochs_no_improve = 0
    checkpoint_paths = []

    for epoch in range(epochs):
        #if epoch % 2 == 0:
        #    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        #    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #    val_loader = DataLoader(val_dataset, batch_size=batch_size)
        #    print(f"\n🔄 Resplit train/val at epoch {epoch+1}")

        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            wandb.log({"train_loss": loss.item()})

        # Validation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(pixel_values=images).logits
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"✅ Val Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
        wandb.log({"val_accuracy": acc, "val_f1": f1, "epoch": epoch + 1})

        scheduler.step(f1)

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            best_path = os.path.join(save_path, "best_model")
            shutil.rmtree(best_path, ignore_errors=True)
            model.save_pretrained(best_path)
            torch.save(optimizer.state_dict(), os.path.join(best_path, "optimizer.pt"))

            config_path = os.path.join(best_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                config["id2label"] = {"0": "None", "1": "Roadwork"}
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

            print(f"\n🔥 Best model saved with F1: {f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⏸️ No improvement. {epochs_no_improve} epoch(s) in a row.")
            if epochs_no_improve >= patience:
                print("⏹️ Early stopping triggered.")
                break

        # Save checkpoint
        checkpoint_dir = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}")
        model.save_pretrained(checkpoint_dir)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        checkpoint_paths.append(checkpoint_dir)

        # Keep last 5 checkpoints only
        if len(checkpoint_paths) > 5:
            old = checkpoint_paths.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            print(f"🧹 Removed old checkpoint: {old}")

    wandb.finish()
    print(f"\n🏁 Training finished. Best F1: {best_f1:.4f}")

def predict(image_path, checkpoint_path):
    model = load_model(checkpoint_path, num_classes=2)
    model.eval()

    processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-base-22k-224")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    label = "working" if pred == 1 else "not working"
    print(f"Prediction: {label}")
    return label

def main():
    parser = argparse.ArgumentParser(description="Road Scene Classifier CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset_dir", required=True, default="/workspace/totrain", help="Path to dataset root (with 0/ and 1/)")
    train_parser.add_argument("--save", default="saved_model", help="Path to save model")
    train_parser.add_argument("--resume", help="Resume training from checkpoint")
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=5e-5)
    train_parser.add_argument("--batchsize", type=int, default=192)
    train_parser.add_argument("--balance_strategy", type=str, choices=["none", "weighted", "oversample", "weighted_oversample"], default="none",
                              help="Balance strategy to handle class imbalance")

    pred_parser = subparsers.add_parser("predict")
    pred_parser.add_argument("--img", required=True, help="Path to input image")
    pred_parser.add_argument("--model", default="saved_model", help="Path to saved model")

    args = parser.parse_args()

    if args.command == "train":
        train(args.dataset_dir, args.save, args.resume, args.epochs, args.lr, args.batchsize, balance_strategy=args.balance_strategy)
    elif args.command == "predict":
        predict(args.img, args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
