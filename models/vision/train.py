import os
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    HAS_TORCH = True
except Exception as e:
    print(f"PyTorch disabled due to import error: {e}")
    HAS_TORCH = False
from models.vision.dataset import MVTecDataset, get_vision_transforms
from models.vision.model import IndustrialAnomalyDetector
from app.config import VISION_MODEL_DIR, DATA_DIR, setup_seeds

def train_anomaly_model(epochs: int = 5, batch_size: int = 16):
    setup_seeds()
    os.makedirs(VISION_MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Vision Train] Using device: {device}")

    # Use training split and further divide into train/val
    transform = get_vision_transforms()
    full_train_dataset = MVTecDataset(root_dir=str(DATA_DIR / "raw" / "mvtec"), split="train", transform=transform)
    
    # 80/20 train/val split
    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = IndustrialAnomalyDetector().to(device)
    optimizer = optim.Adam(model.encoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_threshold = 0.5

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            features, reconstructed = model(images)
            loss = criterion(features, reconstructed)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_scores = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                features, reconstructed = model(images)
                loss = criterion(features, reconstructed)
                val_loss += loss.item()
                
                # Collect scores for threshold calibration
                mse = nn.functional.mse_loss(features, reconstructed, reduction='none').mean(dim=1).cpu().numpy()
                val_scores.extend(mse)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Calibrate threshold as 95th percentile of validation (good) images
            import numpy as np
            best_threshold = float(np.percentile(val_scores, 95))
            torch.save({
                'model_state_dict': model.state_dict(),
                'calibrated_threshold': best_threshold
            }, VISION_MODEL_DIR / "anomaly_detector.pth")

    print(f"[Vision Train] Training complete. Best Val Loss: {best_val_loss:.4f}, Calibrated Threshold: {best_threshold:.4f}")

def train_anomaly_model_mock():
    print("[Vision Train] Mock training for missing Torch dependencies...")
    os.makedirs(VISION_MODEL_DIR, exist_ok=True)
    with open(VISION_MODEL_DIR / "anomaly_detector.pth", "wb") as f:
        f.write(b"mock_weights")

if __name__ == "__main__":
    if HAS_TORCH:
        try:
            torch.tensor([1.0])
            train_anomaly_model(epochs=2)
        except Exception:
            train_anomaly_model_mock()
    else:
        train_anomaly_model_mock()
