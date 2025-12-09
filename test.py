import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import wandb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ConDo.loader import TwoCropsTransform
from ConDo.model import ConDo

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from Camelyon import Camelyon16


def evaluate(checkpoint_path, use_wandb=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = [0.6377, 0.4562, 0.6402]
    std = [0.2403, 0.2536, 0.1970]

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    root_dir = "/path/to/dataset"
    test_txt = "/path/to/test.txt"

    testset = Camelyon16(root_dir, test_txt, transform=transform)

    testloader = DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    encoder = models.resnet50(weights=None)
    encoder.fc = torch.nn.Identity()

    model = ConDo(
        encoder,
        dim_in=2048,
        d_cat=2,
        d_inst=256,
        mlp=True,
        temperature=0.07,
        class_weight=20.0,
        eps=1e-8,
        logit_temp=0.07,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    if use_wandb:
        wandb.init(
            project="sdc-contrastive-mnistl",
            name="eval-MNIST-best",
            config={
                "phase": "testing",
                "checkpoint": checkpoint_path,
                "dataset": "CIFAR-10",
            },
        )

    all_preds, all_labels, all_probs = [], [], []
    logit_temp = 0.1

    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            z_c = model.eval_forward(images)
            logits = z_c / logit_temp
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    correct_mask = all_preds == all_labels
    conf_correct = all_probs[
        np.arange(len(all_preds))[correct_mask], all_preds[correct_mask]
    ]
    conf_incorrect = all_probs[
        np.arange(len(all_preds))[~correct_mask], all_preds[~correct_mask]
    ]
    avg_conf_correct = np.mean(conf_correct) if len(conf_correct) > 0 else float("nan")
    avg_conf_incorrect = (
        np.mean(conf_incorrect) if len(conf_incorrect) > 0 else float("nan")
    )

    try:
        auc_score = roc_auc_score(y_true=all_labels, y_score=all_probs[:, 1])
    except ValueError:
        auc_score = float("nan")

    cm = confusion_matrix(all_labels, all_preds)
    print("\n🧩 Confusion Matrix:")
    print(cm)

    print("\n📄 Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    print(
        f"\n✅ Final Metrics:\n"
        f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | "
        f"F1 Score: {f1:.4f} | AUC: {auc_score:.4f}"
    )
    print(f"Avg Confidence (Correct): {avg_conf_correct:.4f}")
    print(f"Avg Confidence (Incorrect): {avg_conf_incorrect:.4f}")

    if use_wandb:
        wandb.log(
            {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "test_avg_conf_correct": avg_conf_correct,
                "test_avg_conf_incorrect": avg_conf_incorrect,
                "test_auc": auc_score,
            }
        )
        wandb_confmat = wandb.plot.confusion_matrix(
            y_true=all_labels,
            preds=all_preds,
        )
        wandb.log({"confusion_matrix": wandb_confmat})
        wandb.finish()


# === Run Evaluation ===
if __name__ == "__main__":
    checkpoint_path = "/path/to/best_model.pth"
    evaluate(checkpoint_path, use_wandb=False)
