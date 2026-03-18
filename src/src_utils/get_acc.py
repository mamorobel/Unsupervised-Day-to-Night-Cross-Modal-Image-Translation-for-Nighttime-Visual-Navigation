import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from segment import ImageDs, SegModel

GREEN_CLASS = 2

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, lbl in dataloader:
            img = img.to(device)
            lbl = lbl.to(device)

            output = model(img)
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(lbl.cpu().numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    ious = []
    for i in range(3):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious.append(iou)

    mean_iou = np.mean(ious)

    tp = cm[GREEN_CLASS, GREEN_CLASS]
    fp = cm[:, GREEN_CLASS].sum() - tp
    fn = cm[GREEN_CLASS, :].sum() - tp

    precision_green = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_green = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return accuracy, mean_iou, precision_green, recall_green


if __name__ == '__main__':

    config_file = '../configs/config.yaml'

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    base_dir = f'../data/full_night_3fold/set_1/val' # change based on set.

    model_checkpoint = f"../segmentation_checkpoints/{config["experiment_name"]}.pth"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SegModel()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)

    models = {
        f"{config["experiment_name"]}": model
    }

    results = {
        name: {
            "acc": [],
            "miou": [],
            "prec_g": [],
            "rec_g": []
        }
        for name in models
    }

    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')

    dataset = ImageDs(images_dir, labels_dir, mode='val')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for name, model in models.items():
        preds, labels = evaluate_model(model, dataloader, device)
        acc, miou, prec_g, rec_g = compute_metrics(labels, preds)

        results[name]["acc"].append(acc)
        results[name]["miou"].append(miou)
        results[name]["prec_g"].append(prec_g)
        results[name]["rec_g"].append(rec_g)

        print(f"  {name:<6} → "
                f"Acc: {acc:.4f} | "
                f"mIoU: {miou:.4f} | "
                f"Prec(G): {prec_g:.4f} | "
                f"Rec(G): {rec_g:.4f}")