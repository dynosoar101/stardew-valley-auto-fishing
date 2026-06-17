import time
import torch
import torch.nn as nn
import mss
import cv2
import numpy
import pyautogui
from torchvision import transforms
from PIL import Image

# ── Model (must match cnn_training.py exactly) ────────────────────────────────
class FishingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
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

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishingCNN().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()
print(f"Model loaded on {device}")

# ── Capture region (same as training) ────────────────────────────────────────
REGION = {
    "left":   int(2560 * 0.420),
    "top":    int(1440 * 0.250),
    "width":  int(2560 * 0.455) - int(2560 * 0.420),
    "height": int(1440 * 0.66)  - int(1440 * 0.250),
}

transform = transforms.Compose([
    transforms.Resize((128, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ── Minigame loop ─────────────────────────────────────────────────────────────
def minigame():
    print("Minigame started")
    last_held = None
    no_minigame_since = None

    with mss.mss() as sct:
        while True:
            frame = numpy.array(sct.grab(REGION))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                confidence = model(tensor).item()

            hold = confidence > 0.5

            if hold != last_held:
                if hold:
                    pyautogui.mouseDown()
                else:
                    pyautogui.mouseUp()
                last_held = hold

            print(f"conf: {confidence:.2f}  {'HOLD' if hold else 'release'}")

            if confidence < 0.2:
                if no_minigame_since is None:
                    no_minigame_since = time.time()
                elif time.time() - no_minigame_since > 1.0:
                    pyautogui.mouseUp()
                    print("Minigame complete")
                    return
            else:
                no_minigame_since = None