import os
import yaml
import shutil

if __name__ == '__main__':
    config_file = '../configs/config.yaml'

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    vals = '../datasets/strawberryfield_ds/day_val_set/images'
    train = '../datasets/strawberryfield_ds/day_train_set/images'
    vals_labels = '../datasets/strawberryfield_ds/day_val_set/labels'
    train_labels = '../datasets/strawberryfield_ds/day_train_set/labels'
    preds = f'../datasets/{config["experiment_name"]}'

    val_dest = f'../datasets/{config["experiment_name"]}_preds_val/images'
    train_dest = f'../datasets/{config["experiment_name"]}_preds_train/images'
    val_dest_labels = f'../datasets/{config["experiment_name"]}_preds_val/labels'
    train_dest_labels = f'../datasets/{config["experiment_name"]}_preds_train/labels'

    os.makedirs(val_dest, exist_ok=True)
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest_labels, exist_ok=True)
    os.makedirs(train_dest_labels, exist_ok=True)

    vals_dict = {img: "" for img in os.listdir(vals)}
    train_dict = {img: "" for img in os.listdir(train)}

    for img_name in os.listdir(preds):

        img_path = os.path.join(preds, img_name)

        if img_name in vals_dict:
            shutil.copy(img_path, val_dest)
        elif img_name in train_dict:
            shutil.copy(img_path, train_dest)

    for img_name in os.listdir(val_dest):
        img_path = os.path.join(val_dest, img_name)

        if not os.path.isfile(img_path):
            continue

        src_label_path = os.path.join(vals_labels, img_name)
        dest_label_path = os.path.join(val_dest_labels, img_name)

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dest_label_path)
        else:
            print("Label not found: {}".format(img_name))

    for img_name in os.listdir(train_dest):
        img_path = os.path.join(train_dest, img_name)

        if not os.path.exists(img_path):
            continue

        src_label_path = os.path.join(train_labels, img_name)
        dest_label_path = os.path.join(train_dest_labels, img_name)

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dest_label_path)
        else:
            print("Label not found: {}".format(img_name))