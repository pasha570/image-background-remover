import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from u2net import U2NET  # Your U-2-Net model

# Load the model
model = U2NET()
model.load_state_dict(torch.load('u2net.pth', map_location='cpu'))
model.eval()

def remove_background(input_image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

    image = Image.open(input_image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        mask = model(image_tensor)[0][0]
        mask = mask.squeeze().cpu().numpy()

    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (image.width, image.height))

    image_np = np.array(image)
    b, g, r = cv2.split(image_np)
    alpha = mask
    result = cv2.merge((b, g, r, alpha))

    output = Image.fromarray(result, mode='RGBA')
    return output
