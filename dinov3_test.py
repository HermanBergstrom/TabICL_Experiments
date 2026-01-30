import torch
from PIL import Image
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib import colormaps
from transformers.image_utils import load_image

def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

weigths_path = "model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
REPO_DIR = '../dinov3'  # Path to the local directory containing the dinov3 repository

dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=weigths_path)


url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

img_size = 1024
img = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        batch_img = batch_img
        features = dinov3_vitb16(batch_img)

print(features.shape)





#from transformers import pipeline






#feature_extractor = pipeline(
#    model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
#    task="image-feature-extraction", 
#)
#features = feature_extractor(image)