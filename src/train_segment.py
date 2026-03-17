import os
import cv2
import time
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from PIL import Image
from torch.utils.data import DataLoader, Dataset

class ImageDs(Dataset):

    def __init__(self, imgs, labels, mode='train'):
        super().__init__()
        self.imgs = imgs
        self.labels = labels
        self.mode = mode

        self.img_files = sorted(os.listdir(self.imgs))
        self.label_files = sorted(os.listdir(self.labels))

        self.img_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(3)
            ], p=0.25),
            transforms.ToTensor()
        ])

    def __len__(self):
        return int(len(self.img_files) * 0.6)

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs, self.img_files[idx])
        l_img_path = os.path.join(self.labels, self.label_files[idx])

        img = Image.open(img_path).convert('RGB')
        label_img = Image.open(l_img_path).convert('RGB')

        if self.mode == 'train':
            if random.random() < 0.25:
                i, j, h, w = transforms.RandomCrop.get_params(
                    img, output_size=(410, 620)
                )

                img = TF.crop(img, i, j, h, w)
                label_img = TF.crop(label_img, i, j, h, w)

            img = TF.resize(img, (352, 672))
            label_img = TF.resize(label_img, (352, 672), interpolation=TF.InterpolationMode.NEAREST)

            img = self.img_transform(img)

            label_np = np.array(label_img)
            label_final = np.zeros((label_np.shape[0], label_np.shape[1]), dtype=np.uint8)
            label_final[label_np[:, :, 0] > 128] = 1
            label_final[label_np[:, :, 1] > 128] = 2
            label_img = torch.from_numpy(label_final).long()
        else:
            img = TF.resize(img, (352, 672))
            label_img = TF.resize(label_img, (352, 672), interpolation=TF.InterpolationMode.NEAREST)

            img = transforms.ToTensor()(img)

            label_np = np.array(label_img)
            label_final = np.zeros((label_np.shape[0], label_np.shape[1]), dtype=np.uint8)
            label_final[label_np[:, :, 0] > 128] = 1
            label_final[label_np[:, :, 1] > 128] = 2
            label_img = torch.from_numpy(label_final).long()

        return img, label_img


class SegModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.model.children())

        self.encoder()
        self.decoder()

    def residual_block(self, in_channel, out_channel, kernel, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, padding=padding),
            nn.ReLU()
        )

    def encoder(self):
        self.layer1 = nn.Sequential(*self.base_layers[:3])  # 64
        self.layer2 = nn.Sequential(*self.base_layers[3:5])  # 64
        self.layer3 = nn.Sequential(self.base_layers[5])  # 128
        self.layer4 = nn.Sequential(self.base_layers[6])  # 256
        self.bottleneck = nn.Sequential(self.base_layers[7])  # 512

        self.skip1 = self.residual_block(64, 64, 1, 0)
        self.skip2 = self.residual_block(64, 64, 1, 0)
        self.skip3 = self.residual_block(128, 128, 1, 0)
        self.skip4 = self.residual_block(256, 256, 1, 0)

        self.skip_original = self.residual_block(3, 3, 1, 0)

        self.dropout = nn.Dropout2d(p=0.1)

    def decoder(self):
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up4 = self.residual_block(256 + 512, 256, 1, 0)
        self.conv_up3 = self.residual_block(128 + 256, 128, 1, 0)
        self.conv_up2 = self.residual_block(64 + 128, 64, 1, 0)
        self.conv_up1 = self.residual_block(64 + 64, 32, 1, 0)
        self.out_conv = nn.Conv2d(3 + 32, 3, kernel_size=1, padding=0)

    def forward(self, input):
        x_original = self.skip_original(input)
        '''
        encoder
        '''
        x1 = self.layer1(input)
        x1 = self.dropout(x1)
        x1_skip = self.skip1(x1)

        x2 = self.layer2(x1)
        x2 = self.dropout(x2)
        x2_skip = self.skip2(x2)

        x3 = self.layer3(x2)
        x3 = self.dropout(x3)
        x3_skip = self.skip3(x3)

        x4 = self.layer4(x3)
        x4 = self.dropout(x4)
        x4_skip = self.skip4(x4)

        '''
        bottleneck
        '''
        bottleneck = self.bottleneck(x4)

        '''
        decoder
        '''
        x = self.upsample(bottleneck)
        x = F.interpolate(x, size=x4_skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up4(torch.cat((x, x4_skip), dim=1))

        x = self.upsample(x)
        x = F.interpolate(x, size=x3_skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up3(torch.cat((x, x3_skip), dim=1))

        x = self.upsample(x)
        x = F.interpolate(x, size=x2_skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up2(torch.cat((x, x2_skip), dim=1))

        x = self.upsample(x)
        x = F.interpolate(x, size=x1_skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv_up1(torch.cat((x, x1_skip), dim=1))

        x = self.upsample(x)
        x = F.interpolate(x, size=x_original.shape[2:], mode="bilinear", align_corners=False)
        out = self.out_conv(torch.cat((x, x_original), dim=1))

        return out


def compute_mious(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # print("*"*10)
    # print(y_true.shape, y_pred.shape)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    ious = []
    for i in range(3):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious.append(iou)

    print(cm)
    print(np.mean(ious))

    return np.mean(ious)


def train(model, optimizer, loss_fn, train_dataloader, val_dataloader, device, epochs=10):
    criterion = loss_fn

    train_loss = []
    val_loss = []
    mious = []
    highest = 0

    for epoch in range(epochs):

        training_loss = 0.0
        model.train()

        for batch in train_dataloader:
            optimizer.zero_grad()

            img, lbl = batch

            img = img.to(device)
            lbl = lbl.to(device)

            output = model(img)

            loss = criterion(output, lbl)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss = training_loss / len(train_dataloader)
        train_loss.append(training_loss)

        val_loss_epoch = 0.0
        total_cm = np.zeros((3, 3))
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                img, lbl = batch
                img = img.to(device)
                lbl = lbl.to(device)
                output = model(img)

                loss = criterion(output, lbl)
                val_loss_epoch += loss.item()

                pred = torch.argmax(output, dim=1)

                cm = confusion_matrix(
                    lbl.cpu().numpy().flatten(),
                    pred.cpu().numpy().flatten(),
                    labels=[0, 1, 2]
                )

                total_cm += cm

        val_loss_epoch = val_loss_epoch / len(val_dataloader)
        val_loss.append(val_loss_epoch)
        ious = []
        for i in range(3):
            tp = total_cm[i, i]
            fp = total_cm[:, i].sum() - tp
            fn = total_cm[i, :].sum() - tp
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            ious.append(iou)

        miou_epoch = np.mean(ious)
        mious.append(miou_epoch)

        if miou_epoch > highest:
            highest = miou_epoch
            if not os.path.exists('../segmentation_checkpoints'):
                os.makedirs('../segmentation_checkpoints')
            torch.save(model.state_dict(), f'../segmentation_checkpoints/ours_combined.pth')
            print(f"model saved at epoch: {epoch}")

        if not os.path.exists('../loss_plots'):
            os.makedirs('../loss_plots')

        print(f'epoch: {epoch}, training loss: {training_loss}, miou: {miou_epoch}')

        plt.figure(num=1)
        plt.plot(list(range(len(train_loss))), train_loss, label='training loss')
        plt.plot(list(range(len(val_loss))), val_loss, label='val loss')
        plt.plot(list(range(len(mious))), mious, label="mIOU score", color='#000000')
        plt.legend()
        plt.savefig('../loss_plots/segmodel.png')
        plt.close()


if __name__ == '__main__':
    config_file = '../configs/config.yaml'

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    img_dir = '../data/ours_set_2_16_grayscale_id5_preds_train/images'
    labels_dir = '../data/ours_set_2_16_grayscale_id5_preds_train/labels'
    val_dir = '../data/ours_set_2_16_grayscale_id5_preds_val/images'
    labels_val_dir = '../data/ours_set_2_16_grayscale_id5_preds_val/labels'

    # img_dir = '../data/full_night_3fold/set_3/train/images'
    # labels_dir = '../data/full_night_3fold/set_3/train/labels'
    # val_dir = '../data/full_night_3fold/set_3/val/images'
    # labels_val_dir = '../data/full_night_3fold/set_3/val/labels'

    train_imgs = ImageDs(img_dir, labels_dir, mode='train')
    val_imgs = ImageDs(val_dir, labels_val_dir, mode='val')

    train_dataloader = DataLoader(train_imgs, batch_size=6, shuffle=True)
    val_dataloader = DataLoader(val_imgs, batch_size=6, shuffle=False)

    device = 'cuda'
    model = SegModel()
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # 3e-4

    train(model, optimizer, loss, train_dataloader, val_dataloader, device, epochs=100)
