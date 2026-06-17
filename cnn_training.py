import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Dataset ───────────────────────────────────────────────────────────────────
class FishingDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(float(row["mouse_left"]), dtype=torch.float32)
        return img, label

# ── Model ─────────────────────────────────────────────────────────────────────
# Smaller, faster model — input is now 128x32 instead of 590x89
# After 3x MaxPool2d(2): 128->16 height, 32->4 width
class FishingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),         # -> 32 x 64 x 16
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),         # -> 64 x 32 x 8
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),         # -> 128 x 16 x 4
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
all_rows = []
for csv_path in glob.glob("training_data/*/labels_filtered.csv"):
    session_dir = os.path.dirname(csv_path)
    frames_dir  = os.path.join(session_dir, "frames")
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        fpath = os.path.join(frames_dir, row["frame_file"])
        if os.path.exists(fpath):
            all_rows.append({"path": fpath, "mouse_left": int(row["mouse_left"])})

print(f"Total frames: {len(all_rows)}")

train_rows, val_rows = train_test_split(all_rows, test_size=0.2, random_state=42)
print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")

# ── Transforms ────────────────────────────────────────────────────────────────
# Resize to small — keeps the vertical structure (tall strip) but much faster
# ColorJitter simulates lighting variation between fish types / times of day
train_transform = transforms.Compose([
    transforms.Resize((128, 32)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.05)),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_loader = DataLoader(
    FishingDataset(train_rows, train_transform),
    batch_size=128, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    FishingDataset(val_rows, val_transform),
    batch_size=128, shuffle=False, num_workers=0, pin_memory=True
)

# ── Train ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model     = FishingCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

EPOCHS = 40
best_val_acc = 0.0
patience_counter = 0
EARLY_STOP_PATIENCE = 8

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * len(imgs)
        train_correct += ((outputs > 0.5) == labels.bool()).sum().item()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss    += loss.item() * len(imgs)
            val_correct += ((outputs > 0.5) == labels.bool()).sum().item()

    train_acc = train_correct / len(train_rows) * 100
    val_acc   = val_correct   / len(val_rows)   * 100
    avg_val_loss = val_loss / len(val_rows)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} — "
          f"Train loss: {train_loss/len(train_rows):.4f}  acc: {train_acc:.1f}%  | "
          f"Val loss: {avg_val_loss:.4f}  acc: {val_acc:.1f}%")

    scheduler.step(avg_val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "model.pt")
        print(f"  -> Saved new best model (val acc: {val_acc:.1f}%)")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping after {EARLY_STOP_PATIENCE} epochs without improvement.")
            break

print(f"\nDone. Best val accuracy: {best_val_acc:.1f}%")
print("Model saved to model.pt")