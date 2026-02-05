from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision.transforms import v2

import torch

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64
        )

    @torch.no_grad()
    def update(self, preds, targets):
        """
        preds:   [B, H, W]  (int)
        targets: [B, H, W]  (int)
        """
        mask = (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]

        inds = self.num_classes * targets + preds
        self.mat += torch.bincount(
            inds,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute_iou(self):
        h = torch.diag(self.mat)
        fp = self.mat.sum(0) - h
        fn = self.mat.sum(1) - h

        denom = h + fp + fn
        iou = h.float() / denom.float().clamp(min=1)

        return iou

    def mean_iou(self):
        return self.compute_iou().mean().item()


class VOCDataset(VOCSegmentation):

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        return image_tf(img), mask_tf(mask).long().squeeze(0)

def extract_features(model, x):
    # Expected output: [B, N+1, D] (cls + patches)
    with torch.no_grad():
        out = model.forward_features(x)
        tokens = out["x_norm_patchtokens"]  # [B, N, D]
    return tokens

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

class LinearSegmentationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, D, H, W]
        return self.proj(x)


class DinoLinearProbe(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes, patch_size=16):
        super().__init__()
        self.backbone = backbone
        self.head = LinearSegmentationHead(embed_dim, num_classes)
        self.patch_size = patch_size

    def forward(self, x):
        B, _, H, W = x.shape

        tokens = extract_features(self.backbone, x)
        # tokens: [B, N, D]
        #print("Input shape:", x.shape)
        #print("Patch size:", self.patch_size)
        #print("Token shape:", tokens.shape)

        h = H // self.patch_size
        w = W // self.patch_size

        feat = tokens.transpose(1, 2).reshape(B, -1, h, w)
        #print("Feature shape:", feat.shape)
        logits = self.head(feat)

        # Upsample to pixel resolution
        logits = nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits

def evaluate_miou(model, loader, num_classes=21):
    model.eval()
    cm = ConfusionMatrix(num_classes)

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.cuda()
            masks = masks.cuda()

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            cm.update(preds.cpu(), masks.cpu())

    return cm.mean_iou(), cm.compute_iou()

#Main function
if __name__ == "__main__":

    image_tf = make_transform(224)

    mask_tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
    ])


    dataset = VOCDataset(
        root="datasets",
        year="2012",
        image_set="train",
        #download=True,
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    val_dataset = VOCDataset(
        root="datasets",
        year="2012",
        image_set="val",
    )

    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )

    print(f"Training size: {len(dataset)}")
    print(f"Validation size: {len(val_dataset)}")


    # Example — adjust based on actual DINOv3 API you use
    #dinov3 = torch.hub.load(
    #    "facebookresearch/dinov3",
    #    "dinov3_vitb14",
    #)

    weigths_path = "model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    REPO_DIR = '../dinov3'  # Path to the local directory containing the dinov3 repository

    dinov3 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=weigths_path)


    dinov3.eval()
    for p in dinov3.parameters():
        p.requires_grad = False


    model = DinoLinearProbe(
        backbone=dinov3,
        embed_dim=768,
        num_classes=21,
        patch_size=16,
    ).cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=5e-3)

    for epoch in range(20):
        model.train()
        for imgs, masks in loader:
            imgs, masks = imgs.cuda(), masks.cuda()

            logits = model(imgs)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        miou, per_class_iou = evaluate_miou(model, val_loader)
        print(f"Validation mIoU: {miou:.4f}")

        


