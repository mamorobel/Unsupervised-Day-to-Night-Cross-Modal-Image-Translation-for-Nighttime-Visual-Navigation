import os
import cv2
import glob
import time
import json
import yaml
import wandb
import torch
import random
import argparse
import itertools
import numpy as np
# import torch_fidelity
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPProcessor, CLIPModel
# from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassJaccardIndex
from scipy import linalg
from torchvision.models import inception_v3
from segment import SegModel
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, dirA, dirB, labelA, labelB, mode="train"):
        self.dirA = dirA
        self.dirB = dirB
        self.labelA = labelA
        self.labelB = labelB
        self.mode = mode
        self.fileA = sorted(os.listdir(self.dirA))
        self.fileB = sorted(os.listdir(self.dirB))

        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.label_resize = transforms.Resize((256, 256))

    def __len__(self):
        return max(len(self.fileA), len(self.fileB))

    def __getitem__(self, idx):
        img_a_path = os.path.join(self.dirA, self.filesA[idx % len(self.filesA)])
        img_b_path = os.path.join(self.dirB, self.filesB[np.random.randint(0, len(self.filesB))])
        label_a_path = os.path.join(self.labelA, self.filesA[idx % len(self.filesA)])
        label_b_path = os.path.join(self.labelB, self.filesB[np.random.randint(0, len(self.filesB))])

        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')
        label_a = Image.open(label_a_path).convert('RGB')
        label_b = Image.open(label_b_path).convert('RGB')

        if self.mode == 'train':
            if random.random() < 0.25:
                i, j, h, w = transforms.RandomCrop.get_params(
                    img_a, output_size=(200, 200)
                )

                img_a = TF.crop(img_a, i, j, h, w)
                img_b = TF.crop(img_b, i, j, h, w)
                label_a = TF.crop(label_a, i, j, h, w)
                label_b = TF.crop(label_b, i, j, h, w)

                img_a = TF.resize(img_a, (256, 256), interpolation=TF.InterpolationMode.NEAREST)
                img_b = TF.resize(img_b, (256, 256), interpolation=TF.InterpolationMode.NEAREST)
                label_a = TF.resize(label_a, (256, 256), interpolation=TF.InterpolationMode.NEAREST)
                label_b = TF.resize(label_b, (256, 256), interpolation=TF.InterpolationMode.NEAREST)

            if random.random() < 0.25:
                img_a = TF.hflip(img_a)
                img_b = TF.hflip(img_b)
                label_a = TF.hflip(label_a)
                label_b = TF.hflip(label_b)

        img_a = self.img_transform(img_a)
        img_b = self.img_transform(img_b)
        label_a = self.label_resize(label_a)
        label_b = self.label_resize(label_b)

        label_np_a = np.array(label_a)
        label_final = np.zeros((label_np_a.shape[0], label_np_a.shape[1]), dtype=np.uint8)
        label_final[label_np_a[:, :, 0] > 128] = 1
        label_final[label_np_a[:, :, 1] > 128] = 2
        label_a = torch.from_numpy(label_final).long()
        label_a = transforms.Resize((256, 256))(label_a)

        label_np_b = np.array(label_b)
        label_final = np.zeros((label_np_b.shape[0], label_np_b.shape[1]), dtype=np.uint8)
        label_final[label_np_b[:, :, 0] > 128] = 1
        label_final[label_np_b[:, :, 1] > 128] = 2
        label_b = torch.from_numpy(label_final).long()

        return {'A': img_a, 'B': img_b, 'C': label_a, 'D': label_b}


class ResidualBlock(nn.Module):
    def __init__(self, in_features, use_dropout=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers += [nn.Dropout(0.5)]

        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=16):
        super().__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features, use_dropout=True)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

def compute_stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = inception_v3(
            weights="IMAGENET1K_V1",
            aux_logits=True   # MUST be True with pretrained weights
        )

        # Disable auxiliary classifier AFTER loading
        model.AuxLogits = None
        model.fc = nn.Identity()

        self.model = model.eval()

    def forward(self, x):
        x = nn.functional.interpolate(
            x, size=299, mode="bilinear", align_corners=False
        )
        x = (x + 1) / 2  # [-1,1] → [0,1]
        return self.model(x)


class DinoFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/dino:main", "dino_vits16"
        ).eval()

    def forward(self, x):
        x = nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = (x + 1) / 2
        return self.model(x)

@torch.no_grad()
def extract_real_features(loader, key, feature_net, device):
    feats = []
    for batch in loader:
        imgs = batch[key].to(device)
        feats.append(feature_net(imgs).cpu().numpy())
    return np.concatenate(feats)


@torch.no_grad()
def extract_fake_features(loader, key, generator, feature_net, device):
    feats = []
    for batch in loader:
        imgs = batch[key].to(device)
        fake = generator(imgs)
        feats.append(feature_net(fake).cpu().numpy())
    return np.concatenate(feats)

@torch.no_grad()
def compute_fid_and_dino(
    val_loader,
    G_AB,
    G_BA,
    device
):
    inception = InceptionFeatureExtractor().to(device)
    dino = DinoFeatureExtractor().to(device)

    real_B = extract_real_features(val_loader, "B", inception, device)
    fake_B = extract_fake_features(val_loader, "A", G_AB, inception, device)

    fid_a2b = calculate_fid(
        *compute_stats(real_B),
        *compute_stats(fake_B)
    )

    real_B_dino = extract_real_features(val_loader, "B", dino, device)
    fake_B_dino = extract_fake_features(val_loader, "A", G_AB, dino, device)

    real_B_dino /= np.linalg.norm(real_B_dino, axis=1, keepdims=True)
    fake_B_dino /= np.linalg.norm(fake_B_dino, axis=1, keepdims=True)

    dino_a2b = np.mean(np.sum(real_B_dino * fake_B_dino, axis=1))

    real_A = extract_real_features(val_loader, "A", inception, device)
    fake_A = extract_fake_features(val_loader, "B", G_BA, inception, device)

    fid_b2a = calculate_fid(
        *compute_stats(real_A),
        *compute_stats(fake_A)
    )

    real_A_dino = extract_real_features(val_loader, "A", dino, device)
    fake_A_dino = extract_fake_features(val_loader, "B", G_BA, dino, device)

    real_A_dino /= np.linalg.norm(real_A_dino, axis=1, keepdims=True)
    fake_A_dino /= np.linalg.norm(fake_A_dino, axis=1, keepdims=True)

    dino_b2a = np.mean(np.sum(real_A_dino * fake_A_dino, axis=1))

    return {
        "fid_a2b": fid_a2b,
        "fid_b2a": fid_b2a,
        "dino_a2b": dino_a2b,
        "dino_b2a": dino_b2a,
    }

class CycleGANTrainer:
    def __init__(self, config, clip_model_idx=0):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model_idx = clip_model_idx
        self.global_step = 0

        self.lr = float(config["lr"])
        self.lambda_id = config["lambdas"]["id"]
        self.lambda_adv = config["lambdas"]["adv"]
        self.lambda_cyc = config["lambdas"]["cyc"]

        self.use_wandb = config["use_wandb"]
        self.log_rate = config["log_rate"]
        self.img_size = config["img_size"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]

        self.data_root_A = config["data_root_A"]
        self.data_root_B = config["data_root_B"]
        self.labels_root_A = config["labels_root_A"]
        self.labels_root_B = config["labels_root_B"]

        self.val_root_A = config["val_root_A"]
        self.val_root_B = config["val_root_B"]
        self.val_labels_root_A = config["val_labels_root_A"]
        self.val_labels_root_B = config["val_labels_root_B"]

        self.use_clip = config["use_clip"]
        self.use_segmodel = config["use_segmodel"]

        self.experiment_name = config["experiment_name"]
        self.project_name = config["project_name"]

        os.makedirs(os.path.join(self.experiment_name, "checkpoints"), exist_ok=True)

        if self.use_clip:
            model_name = config["clip_models"][self.clip_model_idx]

            self.lambda_clip = config["lambdas"]["clip"]
            self.clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device)

            for p in self.clip_model.parameters():
                p.requires_grad = False

            self.clip_model.eval()

            self.clip_processor = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                  antialias=True),
                transforms.Normalize( #https://github.com/openai/CLIP/issues/20 (best normalization for clip models)
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            self.clip_model = None
            self.clip_processor = None

        if self.use_segmodel:
            self.lambda_semantic = config['lambdas']['semantic']

            self.seg_modelA = SegModel().to(self.device)
            self.seg_modelA.load_state_dict(torch.load(config["seg_modelA_root"], map_location=self.device))
            self.seg_modelA.eval()

            for p in self.seg_modelA.parameters():
                p.requires_grad = False

            self.seg_modelB = SegModel().to(self.device)
            self.seg_modelB.load_state_dict(torch.load(config["seg_modelA_root"], map_location=self.device))

            self.jaccard_metric = MulticlassJaccardIndex(
                num_classes=3,
                average="macro"
            ).to(self.device)

        self.G_AB = Generator().to(self.device)
        self.G_BA = Generator().to(self.device)
        self.D_AB = Discriminator().to(self.device)
        self.D_BA = Discriminator().to(self.device)

        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                name=f"cyclegan_{time.strftime('%Y%m%d_%H%M%S')}",
                config=config,
            )

        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        if self.use_clip:
            self.cos_embedding = nn.CosineSimilarity()
        if self.use_segmodel:
            self.criterion_seg = nn.CrossEntropyLoss()

        self.optimizer_G = optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
                                      lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=self.lr, betas=(0.5, 0.999))
        if self.use_segmodel:
            self.optimizer_seg = optim.Adam(self.seg_modelB.parameters(), lr=1e-4)

        self.lr_scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G,
            lr_lambda=self.lr_lambda
        )

        self.lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A,
            lr_lambda=self.lr_lambda
        )

        self.lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B,
            lr_lambda=self.lr_lambda
        )

        self.fake_A_pool = []
        self.fake_B_pool = []
        self.pool_size = 50

        self.losses = defaultdict(list)
        self.best_loss = float("inf")

    def lr_lambda(self, epoch):
        decay_start = self.num_epochs * 0.75
        if epoch < decay_start:
            return 1.0
        else:
            return max(0.0, 1.0 - (epoch - decay_start) / (self.num_epochs - decay_start))

    def get_clip_features(self, x):
        x = self.clip_preprocess(x)
        return self.clip_model.get_image_features(pixel_values=x)

    def load_dataset(self):
        train_dataset = ImageDataset(self.data_root_A, self.data_root_B, self.labels_root_A, self.labels_root_B, mode='train')
        val_dataset = ImageDataset(self.val_root_A, self.val_root_B, self.val_labels_root_A, self.val_labels_root_B, mode='val')

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=4)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    def sample_from_pool(self, pool, data):
        if len(pool) < self.pool_size:
            pool.append(data)
            return data
        else:
            if np.random.random() > 0.5:
                idx = np.random.randint(0, self.pool_size)
                temp = pool[idx].clone()
                pool[idx] = data
                return temp
            else:
                return data

    def torchmetrics_jaccard_loss(self, preds, target):
        preds = torch.argmax(preds, dim=1)

        if target.ndim == 2:
            target = target.unsqueeze(0)

        target = target.long()
        iou = self.jaccard_metric(preds, target)

        return 1.0 - iou

    def generate_mask(self, img_tensor):
        t = img_tensor.clone().squeeze(0)
        t = t.detach().cpu().permute(1, 2, 0).numpy()
        t = ((t + 1) / 2 * 255).astype(np.uint8)
        img = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        threshold = 130

        for i in range(w):
            locs = np.where(img[:, i] >= threshold)[0]
            if len(locs) > 0:
                mask[locs[0]:, i] = 1

        return mask

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

        epoch_losses = defaultdict(float)

        for i, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")):
            real_A = batch['A'].to(self.device)
            real_B = batch['B'].to(self.device)
            label_a = batch['C'].to(self.device)
            label_b = batch['D'].to(self.device)

            self.optimizer_G.zero_grad()

            with torch.no_grad():
                d_output_shape = self.D_A(real_A).shape

            valid = torch.ones(d_output_shape, device=self.device)
            fake = torch.zeros(d_output_shape, device=self.device)

            idt_A = self.G_BA(real_A)
            idt_B = self.G_AB(real_B)
            loss_id_A = self.criterion_identity(idt_A, real_A)
            loss_id_B = self.criterion_identity(idt_B, real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)

            loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
            loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            recov_A = self.G_BA(fake_B)
            recov_B = self.G_AB(fake_A)


            fake_B_mask = torch.from_numpy(self.generate_mask(fake_B)).float().to(self.device)
            masked_real_A = real_A * fake_B_mask
            masked_recov_A = recov_A * fake_B_mask
            real_B_mask = torch.from_numpy(self.generate_mask(real_B)).float().to(self.device)
            masked_recov_B = recov_B * real_B_mask

            loss_cycle_A = self.criterion_cycle(masked_recov_A, masked_real_A)
            loss_cycle_B = self.criterion_cycle(masked_recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            loss_feature = 0
            loss_clip = 0
            loss_seg = 0

            if self.sd_in_use:
                print("Extracting Stable Diffusion features for loss computation...")
                with torch.no_grad():
                    real_A_features = self.get_sd_features(real_A)
                    real_B_features = self.get_sd_features(real_B)
                    fake_A_features = self.get_sd_features(fake_A)
                    fake_B_features = self.get_sd_features(fake_B)

                loss_feature_A = self.criterion_cycle(fake_A_features, real_A_features)
                loss_feature_B = self.criterion_cycle(fake_B_features, real_B_features)
                loss_feature = (loss_feature_A + loss_feature_B) * self.lambda_sd

            if self.use_clip:
                real_A_clip = (real_A + 1.0) / 2.0
                real_B_clip = (real_B + 1.0) / 2.0
                fake_A_clip = (fake_A + 1.0) / 2.0
                fake_B_clip = (fake_B + 1.0) / 2.0
                recov_A_clip = (recov_A + 1.0) / 2.0
                recov_B_clip = (recov_B + 1.0) / 2.0

                assert fake_A_clip.requires_grad, "fake_A has no grad"
                assert fake_B_clip.requires_grad, "fake_B has no grad"
                assert recov_A_clip.requires_grad, "recov_A has no grad"
                assert recov_B_clip.requires_grad, "recov_B has no grad"

                binary_mask = torch.tensor(self.generate_mask(fake_B_clip)).to(self.device)
                masked_real_A_clip = transforms.Grayscale(num_output_channels=3)(real_A_clip * binary_mask)
                masked_recov_A_clip = transforms.Grayscale(num_output_channels=3)(recov_A_clip * binary_mask)
                masked_fake_B_clip = transforms.Grayscale(num_output_channels=3)(fake_B_clip * binary_mask)

                real_B_mask = torch.tensor(self.generate_mask(real_B_clip)).to(self.device)
                masked_fake_A_clip = transforms.Grayscale(num_output_channels=3)(fake_A_clip * real_B_mask)
                masked_recov_B_clip = transforms.Grayscale(num_output_channels=3)(recov_B_clip * real_B_mask)

                masked_real_A_clip, masked_fake_B_clip, masked_recov_A_clip = \
                    self.crop_from_top_nonzero(
                        masked_real_A_clip,
                        [masked_real_A_clip, masked_fake_B_clip, masked_recov_A_clip]
                    )

                real_B_clip, masked_fake_A_clip, masked_recov_B_clip = \
                    self.crop_from_top_nonzero(
                        masked_recov_B_clip,
                        [real_B_clip, masked_fake_A_clip, masked_recov_B_clip]
                    )

                features_real_A = self.get_clip_features(masked_real_A_clip)
                features_real_B = self.get_clip_features(real_B_clip)
                features_fake_A = self.get_clip_features(masked_fake_A_clip)
                features_fake_B = self.get_clip_features(masked_fake_B_clip)
                features_recov_A = self.get_clip_features(masked_recov_A_clip)
                features_recov_B = self.get_clip_features(masked_recov_B_clip)

                embed_target = torch.ones(features_real_A.size(0)).to(self.device)

                realA_fakeB = self.cos_embedding(features_fake_B, features_real_A, embed_target)
                realA_recovA = self.cos_embedding(features_recov_A, features_real_A, embed_target)
                realB_fakeA = self.cos_embedding(features_fake_A, features_real_B, embed_target)
                realB_recovB = self.cos_embedding(features_recov_B, features_real_B, embed_target)

                loss_clip = (realA_fakeB + realA_recovA + realB_fakeA + realB_recovB) * self.lambda_clip

            if self.use_segmodel:
                seg_fake_A = self.seg_modelA(fake_A)
                seg_recov_A = self.seg_modelA(recov_A)

                seg_real_B = self.seg_modelB(real_B)
                seg_recov_B = self.seg_modelB(recov_B)

                realB_vs_fakeA = self.torchmetrics_jaccard_loss(
                    seg_real_B,
                    torch.argmax(seg_fake_A.detach(), dim=1)
                )

                recovB_vs_fakeA = self.torchmetrics_jaccard_loss(
                    seg_recov_B,
                    torch.argmax(seg_fake_A.detach(), dim=1)
                )

                gt_vs_night_hat = self.torchmetrics_jaccard_loss(
                    self.seg_modelB(fake_B.detach()),
                    label_a.long()
                )

                gt_vs_recovA = self.torchmetrics_jaccard_loss(
                    seg_recov_A,
                    label_a.long()
                )

                loss_seg = (
                       gt_vs_night_hat +
                       gt_vs_recovA +
                       realB_vs_fakeA +
                       recovB_vs_fakeA
                ) * self.lambda_semantic
            else:
                loss_seg = 0

            loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity + loss_feature + loss_clip + loss_seg

            loss_G.backward()
            self.optimizer_G.step()

            self.optimizer_D_A.zero_grad()

            fake_A_pooled = self.sample_from_pool(self.fake_A_pool, fake_A.detach())
            loss_real_A = self.criterion_GAN(self.D_A(real_A), valid)
            loss_fake_A = self.criterion_GAN(self.D_A(fake_A_pooled), fake)

            loss_D_A = (loss_real_A + loss_fake_A) / 2

            loss_D_A.backward()
            self.optimizer_D_A.step()

            self.optimizer_D_B.zero_grad()

            fake_B_pooled = self.sample_from_pool(self.fake_B_pool, fake_B.detach())
            loss_real_B = self.criterion_GAN(self.D_B(real_B), valid)
            loss_fake_B = self.criterion_GAN(self.D_B(fake_B_pooled), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2

            loss_D_B.backward()
            self.optimizer_D_B.step()

            if self.use_segmodel:
                self.optimizer_seg.zero_grad()

                fake_B_det = fake_B.detach()
                recov_A_det = recov_A.detach()

                with torch.no_grad():
                    fx_recov_A = self.seg_modelA(recov_A_det)

                fy_fake_B = self.seg_modelB(fake_B_det)

                loss_fy = self.criterion_seg(
                    fy_fake_B,
                    fx_recov_A.detach()
                )

                loss_fy.backward()
                self.optimizer_seg.step()

            if self.use_wandb:
                if i % 200 == 0:
                    grid = make_grid(
                        torch.cat([real_A[:4], fake_B[:4], real_B[:4], fake_A[:4]], dim=0),
                        nrow=4,
                        normalize=True
                    )
                    self.global_step += 1

            epoch_losses['G'] += loss_G.item()
            epoch_losses['G_AB'] += loss_GAN_AB.item()
            epoch_losses['G_BA'] += loss_GAN_BA.item()
            epoch_losses['D_A'] += loss_D_A.item()
            epoch_losses['D_B'] += loss_D_B.item()
            epoch_losses['cycle'] += loss_cycle.item()
            # epoch_losses['identity'] += loss_identity.item()
            if self.sd_in_use:
                epoch_losses['feature'] += loss_feature.item()
            if self.use_clip:
                epoch_losses['clip'] += loss_clip.item()
            if self.use_segmodel:
                epoch_losses['semantic'] += loss_seg.item()

        # Average losses over all batches
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
            self.losses[key].append(epoch_losses[key])

        return epoch_losses

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'G_AB_state_dict': self.G_AB.state_dict(),
            'G_BA_state_dict': self.G_BA.state_dict(),
            'D_A_state_dict': self.D_A.state_dict(),
            'D_B_state_dict': self.D_B.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_A_state_dict': self.optimizer_D_A.state_dict(),
            'optimizer_D_B_state_dict': self.optimizer_D_B.state_dict(),
            'losses': dict(self.losses),
            'best_loss': self.best_loss
        }

        if is_best:
            torch.save(checkpoint,
                       f'{self.experiment_name}/checkpoints/model_best.pth')
            print(f"New best model saved at epoch {epoch}!")
        else:
            torch.save(checkpoint, f'{self.experiment_name}/checkpoints/model_{epoch}.pth')











