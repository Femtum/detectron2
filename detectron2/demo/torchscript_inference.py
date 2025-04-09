# flake8: noqa

import os
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt

model_path = os.path.join(os.path.dirname(__file__), "output", "model.ts")
image_path = os.path.join(os.path.dirname(__file__), "input.jpg")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found at {model_path}. See export-torchscript-example.md for instructions."
    )

model = torch.jit.load(model_path)

image = Image.open(image_path).convert("RGB")
image_tensor = T.ToTensor()(image) * 255  # Convert to uint8 scale
image_tensor = image_tensor.to(torch.uint8)

# Prepare input dict (NO height/width)
inputs = [{"image": image_tensor}]

with torch.no_grad():
    outputs = model(inputs)[0]


# Convert the original PIL image to a NumPy array (RGB format)
img = np.array(image)


# Draw boxes
for box in outputs["pred_boxes"]:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show image
plt.imshow(img)
plt.axis("off")
plt.show()
