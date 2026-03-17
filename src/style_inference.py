import os
import yaml
import glob
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.utils import save_image

from cyclegan import Generator


def load_generator(model_path, device):
    model = Generator()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)['G_AB_state_dict'])
    model.to(device)
    model.eval()
    return model


def convert_images(generator, input_dir, output_dir, img_w, img_h, device):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
    ])

    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.*')))

    print(f"Found {len(image_paths)} images in {input_dir}")

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Converting"):
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)

            output_tensor = generator(input_tensor)

            filename = os.path.basename(img_path)
            save_path = os.path.join(output_dir, filename)
            save_image(output_tensor, save_path)

    print(f"Converted images saved to: {output_dir}")


if __name__ == "__main__":
    config_file = '../configs/config.yaml'

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = f'{config['experiment_name']}/checkpoints/model_best.pth'
    input_dir = "../datasets/full_day_set" # make sure you run the src_utils/combine.py file to create full daytime image set

    output_dir = f"../datasets/{config['experiment_name']}_preds"
    img_w = 1280
    img_h = 720

    generator = load_generator(model_path, device)
    convert_images(generator, input_dir, output_dir, img_w, img_h, device)
