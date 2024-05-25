import torch
import model
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

def inference_base(source_image_path, driving_image_path, Gbase):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load source and driving images
    source_image = load_image(source_image_path, transform)
    driving_image = load_image(driving_image_path, transform)

    # Move images to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_image = source_image.to(device)
    driving_image = driving_image.to(device)

    # Set Gbase to evaluation mode
    Gbase.eval()

    with torch.no_grad():
        # Generate output frame
        output_frame = Gbase(source_image, driving_image)

        # Convert output frame to numpy array
        output_frame = output_frame.squeeze(0).cpu().numpy()
        output_frame = np.transpose(output_frame, (1, 2, 0))
        output_frame = (output_frame + 1) / 2
        output_frame = (output_frame * 255).astype(np.uint8)

        # Convert BGR to RGB
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    return output_frame

def main():
    # Load pretrained base model
    Gbase = model.Gbase()
    Gbase.load_state_dict(torch.load("Gbase.pth"))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Gbase.to(device)

    # Specify paths to source and driving images
    source_image_path = "path/to/source/image.jpg"
    driving_image_path = "path/to/driving/image.jpg"

    # Perform inference
    output_frame = inference_base(source_image_path, driving_image_path, Gbase)

    # Save output frame
    cv2.imwrite("output_base.jpg", output_frame)

if __name__ == "__main__":
    main()