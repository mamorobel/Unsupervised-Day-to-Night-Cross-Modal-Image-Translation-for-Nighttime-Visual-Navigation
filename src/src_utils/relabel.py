import os
import cv2
import yaml
import numpy as np

def generate_mask(img, lbl):
    img = np.array(img)
    lbl = np.array(lbl)
    lbl = cv2.resize(lbl, (img.shape[1], img.shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    mask = np.zeros((h,w), dtype=np.uint8)

    threshold = 20

    for i in range(w):
        locs = np.where(img[:,i] >= threshold)[0]
        if len(locs) > 0:
            mask[locs[0]:, i] = 1

    new_label = lbl * mask[..., None]
    new_label = cv2.cvtColor(new_label, cv2.COLOR_BGR2RGB)

    return new_label

if __name__ == '__main__':
    config_file = '../configs/config.yaml'

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_labels_path = f"../datasets/{config['experiment_name']}_preds_train/labels"
    train_images_path = f"../datasets/{config['experiment_name']}_preds_train/images"
    val_labels_path = f"../datasets/{config['experiment_name']}_preds_val/labels"
    val_images_path = f"../datasets/{config['experiment_name']}_preds_val/images"

    train_labels = os.listdir(train_labels_path)
    train_images = os.listdir(train_images_path)
    val_labels = os.listdir(val_labels_path)
    val_images = os.listdir(val_images_path)

    for image, label in zip(train_images, train_labels):
        image_path = os.path.join(train_images_path, image)
        label_path = os.path.join(train_labels_path, label)

        new_label = generate_mask(cv2.imread(image_path), cv2.imread(label_path))
        torgb = cv2.cvtColor(new_label, cv2.COLOR_BGR2RGB)
        cv2.imwrite(label_path, torgb)

    for image, label in zip(val_images, val_labels):
        image_path = os.path.join(val_images_path, image)
        label_path = os.path.join(val_labels_path, label)

        new_label = generate_mask(cv2.imread(image_path), cv2.imread(label_path))
        torgb = cv2.cvtColor(new_label, cv2.COLOR_BGR2RGB)
        cv2.imwrite(label_path, torgb)

