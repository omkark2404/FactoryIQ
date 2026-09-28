import os
from PIL import Image, ImageDraw

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    Dataset = object
    transforms = None

SUPPORTED_CATEGORIES = ["bottle", "screw", "metal_nut", "tile"]

def get_vision_transforms(image_size: int = 224):
    if not HAS_TORCH:
        return None
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def generate_synthetic_inspection_image(category: str = "bottle", has_defect: bool = False) -> Image.Image:
    img = Image.new("RGB", (224, 224), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    
    if category == "bottle":
        draw.rectangle([60, 40, 164, 200], fill=(180, 210, 240), outline=(50, 50, 50), width=3)
        draw.rectangle([85, 15, 139, 40], fill=(150, 180, 210), outline=(50, 50, 50), width=2)
        if has_defect:
            draw.line([(100, 80), (115, 105), (105, 125), (120, 140)], fill=(220, 30, 30), width=4)
    elif category == "screw":
        draw.rectangle([95, 30, 129, 190], fill=(140, 140, 145), outline=(30, 30, 30), width=2)
        draw.rectangle([80, 15, 144, 35], fill=(100, 100, 105), outline=(30, 30, 30), width=3)
        for y in range(45, 180, 15):
            draw.line([(90, y), (134, y)], fill=(80, 80, 85), width=2)
        if has_defect:
            draw.ellipse([90, 100, 134, 130], fill=(200, 50, 50))
    elif category == "metal_nut":
        draw.polygon([(112, 30), (170, 65), (170, 145), (112, 180), (54, 145), (54, 65)], fill=(160, 165, 170), outline=(40, 40, 40), width=3)
        draw.ellipse([82, 75, 142, 135], fill=(230, 230, 230), outline=(40, 40, 40), width=2)
        if has_defect:
            draw.ellipse([55, 60, 85, 90], fill=(180, 20, 20))
    elif category == "tile":
        draw.rectangle([30, 30, 194, 194], fill=(215, 215, 200), outline=(60, 60, 60), width=3)
        if has_defect:
            draw.line([(50, 50), (170, 170)], fill=(180, 40, 40), width=5)

    return img

class MVTecDataset(Dataset):
    def __init__(self, root_dir: str, category: str = "bottle", split: str = "train", transform=None):
        self.root_dir = root_dir
        self.category = category.lower()
        self.split = split
        self.transform = transform or get_vision_transforms()
        
        self.category_dir = os.path.join(root_dir, self.category, split)
        self.image_paths = []
        self.labels = []

        if os.path.exists(self.category_dir):
            for label_name in os.listdir(self.category_dir):
                label_path = os.path.join(self.category_dir, label_name)
                if os.path.isdir(label_path):
                    is_anomaly = 0 if label_name == "good" else 1
                    for fname in os.listdir(label_path):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(label_path, fname))
                            self.labels.append(is_anomaly)

    def __len__(self):
        return len(self.image_paths) if self.image_paths else 20

    def __getitem__(self, idx: int):
        if self.image_paths:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
        else:
            has_defect = (idx % 2 == 1) if self.split == "test" else False
            image = generate_synthetic_inspection_image(self.category, has_defect=has_defect)
            label = 1 if has_defect else 0

        if self.transform:
            tensor_img = self.transform(image)
            return tensor_img, label
        return image, label
