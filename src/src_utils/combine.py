import os
import shutil

data_root_a = "../datasets/strawberryfield_ds/day_train_set/images"
data_root_b = "../datasets/strawberryfield_ds/day_val_set/images"

output_dir = "../datasets/full_day_set"
os.makedirs(output_dir, exist_ok=True)

root_a = [img for img in os.listdir(data_root_a)]
root_b = [img for img in os.listdir(data_root_b)]

for i in range(len(root_a)):
    img_path = os.path.join(data_root_a, root_a[i])
    shutil.copy(img_path, output_dir)

for i in range(len(root_b)):
    img_path = os.path.join(data_root_b, root_b[i])
    shutil.copy(img_path, output_dir)