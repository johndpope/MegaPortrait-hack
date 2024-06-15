# performance metrics (L1, LPIPS, PSNR, SSIM, AKD, AED) 
import numpy as np
import cv2
import dlib  # For example, using dlib for facial landmark detection
import os
import cv2
import torch
import numpy as np
import lpips
import skimage.metrics


# Load pre-trained dlib model for facial landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None  # No face detected
    face = faces[0]
    landmarks = predictor(gray, face)
    keypoints = np.array([(p.x, p.y) for p in landmarks.parts()])
    return keypoints

# Average Euclidean Distance
def calculate_aed(pred, target):
    pred_keypoints = extract_keypoints(pred)
    target_keypoints = extract_keypoints(target)
    if pred_keypoints is None or target_keypoints is None:
        return None  # Skip images without detected keypoints
    distance = np.linalg.norm(pred_keypoints - target_keypoints, axis=1)
    return np.mean(distance)

def calculate_l1(pred, target):
    return torch.nn.functional.l1_loss(pred, target).item()

def calculate_lpips(pred, target, lpips_model):
    return lpips_model(pred, target).item()

def calculate_psnr(pred, target):
    return skimage.metrics.peak_signal_noise_ratio(target, pred)

def calculate_ssim(pred, target):
    return skimage.metrics.structural_similarity(target, pred, multichannel=True)

def load_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize to [0, 1]
    return img

def preprocess_image_for_lpips(img):
    # Convert image to PyTorch tensor and normalize to [-1, 1] as required by LPIPS
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC to CHW
    img = img * 2 - 1  # Normalize to [-1, 1]
    return img.unsqueeze(0)  # Add batch dimension


def evaluate_metrics(output_dir, target_dir):
    lpips_model = lpips.LPIPS(net='alex')

    l1_scores = []
    lpips_scores = []
    psnr_scores = []
    ssim_scores = []
    akd_scores = []
    aed_scores = []

    for filename in os.listdir(output_dir):
        if filename.startswith("cross_reenactment_images") or filename.startswith("pred_frame"):
            pred_path = os.path.join(output_dir, filename)
            target_path = os.path.join(target_dir, filename)
            
            if os.path.exists(target_path):
                pred_img = load_image(pred_path)
                target_img = load_image(target_path)
                
                l1 = calculate_l1(torch.tensor(pred_img), torch.tensor(target_img))
                lpips_score = calculate_lpips(preprocess_image_for_lpips(pred_img), preprocess_image_for_lpips(target_img), lpips_model)
                psnr = calculate_psnr(pred_img, target_img)
                ssim = calculate_ssim(pred_img, target_img)
                akd = calculate_akd(pred_img, target_img)
                aed = calculate_aed(pred_img, target_img)
                
                l1_scores.append(l1)
                lpips_scores.append(lpips_score)
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)
                akd_scores.append(akd)
                if aed is not None:
                    aed_scores.append(aed)
    
    return {
        "L1": np.mean(l1_scores),
        "LPIPS": np.mean(lpips_scores),
        "PSNR": np.mean(psnr_scores),
        "SSIM": np.mean(ssim_scores),
        "AKD": np.mean(akd_scores),
        "AED": np.mean(aed_scores) if aed_scores else None
    }


output_directory = "path/to/output_images"
target_directory = "path/to/target_images"

metrics = evaluate_metrics(output_directory, target_directory)

print(f"L1: {metrics['L1']}")
print(f"LPIPS: {metrics['LPIPS']}")
print(f"PSNR: {metrics['PSNR']}")
print(f"SSIM: {metrics['SSIM']}")
print(f"AKD: {metrics['AKD']}")
print(f"AED: {metrics['AED']}")
