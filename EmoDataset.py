from moviepy.editor import VideoFileClip, ImageSequenceClip
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
import os
from typing import List, Tuple, Dict, Any
# from rembg import remove - slow
import io
import numpy as np
import random
from tqdm import tqdm
import cv2
from pathlib import Path
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchaudio.io import StreamReader
import torchvision
# from skimage.transform import PiecewiseAffineTransform, warp
import face_recognition
from rembg import remove
from multiprocessing import Pool, cpu_count
import cupy as cp
# background remover
# Initialize the model with GPU support
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True).to(device)
# model.eval()



class EMODataset(Dataset):
    def __init__(self, use_gpu: False, sample_rate: int, n_sample_frames: int, width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None, remove_background=False, use_greenscreen=False, apply_crop_warping=False, num_videos=100):
        self.sample_rate = sample_rate

        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.video_dir = video_dir
        self.transform = transform
        self.stage = stage
        self.pixel_transform = transform
        self.drop_ratio = drop_ratio
        self.remove_background = remove_background
        self.use_greenscreen = use_greenscreen
        self.apply_crop_warping = apply_crop_warping
        with open(json_file, 'r') as f:
            self.celebvhq_info = json.load(f)

        self.use_gpu = use_gpu



        # self.video_ids = list(self.celebvhq_info['clips'].keys())

        # random_video_id = random.choice(self.video_ids)
        # driving = os.path.join(self.video_dir, f"{random_video_id}.mp4")
        # print("driving:",driving)

        # self.driving_vid_pil_image_list = self.load_and_process_video(driving)
        vids = list(self.celebvhq_info['clips'].keys())
        random_video_id = random.choice(vids)
        driving_star = os.path.join(self.video_dir, f"{random_video_id}.mp4")
        print("driving_star:",driving_star)

            # multicore - this never finishes
        with Pool(8) as pool:
            results = pool.map(self.load_and_process_video, [(video_id, idx) for idx, video_id in enumerate(vids)])
      

        
        # self.driving_vid_pil_image_list_star = self.load_and_process_video(driving_star)



        # driving = os.path.join(self.video_dir, "-2KGPYEFnsU_11.mp4")
        # self.driving_vid_pil_image_list = self.load_and_process_video(driving)
        # self.video_ids = ["M2Ohb0FAaJU_1","-2KGPYEFnsU_11","-1eKufUP5XQ_4","-2KGPYEFnsU_8"]  # list(self.celebvhq_info['clips'].keys())
        # self.video_ids_star = ["M2Ohb0FAaJU_1","-2KGPYEFnsU_11","-1eKufUP5XQ_4","-2KGPYEFnsU_8"]  # list(self.celebvhq_info['clips'].keys())
        # driving_star = os.path.join(self.video_dir, "-2KGPYEFnsU_8.mp4")
        # self.driving_vid_pil_image_list_star = self.load_and_process_video(driving_star)


    def __len__(self) -> int:
        return len(self.video_ids)



    def remove_bg(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        bg_removed_bytes = remove(img_byte_arr)
        bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")  # Use RGBA to keep transparency

        if self.use_greenscreen:
            # Create a green screen background
            green_screen = Image.new("RGBA", bg_removed_image.size, (0, 255, 0, 255))  # Green color

            # Composite the image onto the green screen
            final_image = Image.alpha_composite(green_screen, bg_removed_image)
        else:
            final_image = bg_removed_image

        final_image = final_image.convert("RGB")  # Convert to RGB format
        return final_image
        

    def piecewise_affine_transform(image, src_points, dst_points):
        # Create an empty array to store the transformed image
        transformed_image = cp.zeros_like(image)
        
        # Define the transformation matrix for each triangle
        for i in range(len(src_points) - 1):
            src_tri = np.array([src_points[i], src_points[i + 1], src_points[(i + 2) % len(src_points)]])
            dst_tri = np.array([dst_points[i], dst_points[i + 1], dst_points[(i + 2) % len(dst_points)]])
            
            # Compute the affine transform matrix for the current triangle
            M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
            
            # Warp the corresponding region of the image
            mask = cp.zeros_like(image)
            cv2.fillConvexPoly(mask, np.int32(dst_tri), (1, 1, 1))
            warped = cv2.cuda.warpAffine(image, M, (image.shape[1], image.shape[0]))
            transformed_image += mask * warped

        return transformed_image
        
    def warp_and_crop_face(self, image_tensor, video_name, frame_idx, transform=None, output_dir="output_images", warp_strength=0.01, apply_warp=False):
        # Ensure the output directory exists
        print("frame_idx:",frame_idx)
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the file path
        output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx}.png")
        
        # Check if the file already exists
        if os.path.exists(output_path):
            # Load and return the existing image as a tensor
            existing_image = Image.open(output_path).convert("RGBA")
            return to_tensor(existing_image)
        
        # Check if the input tensor has a batch dimension and handle it
        if image_tensor.ndim == 4:
            # Assuming batch size is the first dimension, process one image at a time
            image_tensor = image_tensor.squeeze(0)
        
        # Convert the single image tensor to a PIL Image
        image = to_pil_image(image_tensor)
        
        # Remove the background from the image using the updated remove_bg method
        bg_removed_image = self.remove_bg(image)
        
        # Convert the image to RGB format to make it compatible with face_recognition
        bg_removed_image_rgb = bg_removed_image.convert("RGB")
        
        # Detect the face in the background-removed RGB image using the numpy array
        face_locations = face_recognition.face_locations(np.array(bg_removed_image_rgb))
        
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            
            # automatically choose sweet spot to crop.
            # https://github.com/tencent-ailab/V-Express/blob/main/assets/crop_example.jpeg
            face_width = right - left
            face_height = bottom - top
            
            # Calculate the padding amount based on face size and output dimensions
            pad_width = int(face_width * 0.5)
            pad_height = int(face_height * 0.5)
            
            # Expand the cropping coordinates with the calculated padding
            left = max(0, left - pad_width)
            top = max(0, top - pad_height)
            right = min(bg_removed_image.width, right + pad_width)
            bottom = min(bg_removed_image.height, bottom + pad_height)

            # Crop the face region from the image
            face_image = bg_removed_image.crop((left, top, right, bottom))
            
            if apply_warp:
                # Convert the face image to a numpy array
                face_array = cp.array(face_image)

                # Generate random control points for thin-plate-spline warping
                rows, cols = face_array.shape[:2]
                src_points = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
                dst_points = src_points + np.random.randn(4, 2) * (rows * warp_strength)

                # Apply the piecewise affine transformation
                warped_face_array = self.piecewise_affine_transform(face_array, src_points, dst_points)
                
                # Convert the warped face array back to a PIL image
                warped_face_image = Image.fromarray((warped_face_array * 255).astype(np.uint8))
            else:
                warped_face_image = face_image
            
            # Apply the transform if provided
            if transform:
                warped_face_image = warped_face_image.convert("RGB")
                warped_face_tensor = transform(warped_face_image)
                return warped_face_tensor
            
            # Convert the warped PIL image back to a tensor
            # Convert the warped PIL image to RGB format before converting to a tensor
            warped_face_image = warped_face_image.convert("RGB")
            return to_tensor(warped_face_image)

        else:
            return None

   
    def load_and_process_video(self, video_id: str) -> List[torch.Tensor]:
        print(f"load_and_process_video... {video_id}")
        # video_id = Path(video_path).stem
        video_path = self.video_dir + "/" + f"{video_id[0]}.mp4"

        output_dir = Path(self.video_dir + "/" + video_id[0])
        output_dir.mkdir(exist_ok=True)
        
 
        tensor_frames = []

        tensor_file_path = output_dir / f"{video_id}_tensors.npz"

        if tensor_file_path.exists():
            print(f"Loading processed tensors from file: {tensor_file_path}")
            with np.load(tensor_file_path) as data:
                tensor_frames = [torch.tensor(data[key]) for key in data]
        else:
            if self.apply_crop_warping:
                print(f"Warping + Processing and saving video frames to directory: {output_dir}")
            else:
                print(f"Processing and saving video frames to directory: {output_dir}")
                
            streamer = StreamReader(src=video_path)

            # - ``"rgb24"``: 8 bits * 3 channels (R, G, B)
            # - ``"bgr24"``: 8 bits * 3 channels (B, G, R)
            # - ``"yuv420p"``: 8 bits * 3 channels (Y, U, V)
            # - ``"gray"``: 8 bits * 1 channels
            
            streamer.add_basic_video_stream(
                frames_per_chunk=16000,
                frame_rate=25,
                width=self.width,
                height=self.height,
                format="rgb24"
            )
            
            all_frames = []
            for video_chunk in streamer.stream():
                if video_chunk is not None:
                    frames = video_chunk[0] # Directly use the tensor from video_chunk
                    all_frames = frames


            tensor_frames = [] #- this succeeds
            for idx, frame in enumerate(all_frames):
                result = self.process_frame((frame, idx, output_dir, video_path))
                if result is not None:
                    tensor_frames.append(result)
            if tensor_frames:
                np.savez_compressed(tensor_file_path, *[tensor_frame.numpy() for tensor_frame in tensor_frames])
                print(f"Processed tensors saved to file: {tensor_file_path}")
            else:
                print(f"No valid tensor frames processed for video {video_path}")
            return tensor_frames
                
            np.savez_compressed(tensor_file_path, *[tensor_frame.numpy() for tensor_frame in tensor_frames])
            print(f"Processed tensors saved to file: {tensor_file_path}")

        return tensor_frames
        

    def process_frame(self, args):
        frame, frame_idx, output_dir,video_path = args
        print("idx:",frame_idx)
        try:

            frame = to_pil_image(frame)  # Convert to PIL image
            state = torch.get_rng_state()
            tensor_frame, image_frame = self.augmentation(frame, self.pixel_transform, state)

    
            if self.apply_crop_warping:
                transform = transforms.Compose([
                    transforms.Resize((self.width, self.height)),
                    transforms.ToTensor(),
                ])
                video_name = Path(video_path).stem
                print("frame_idx:",frame_idx)
                tensor_frame1 = self.warp_and_crop_face(tensor_frame, video_name, frame_idx, transform, apply_warp=False)
                
                if tensor_frame1 is not None:
                    img = to_pil_image(tensor_frame1)
                    img.save(output_dir / f"{frame_idx:06d}.png")
                    return tensor_frame1

                    # tensor_frame2 = self.warp_and_crop_face(tensor_frame, video_name, frame_idx, transform, apply_warp=True)
                    # img = to_pil_image(tensor_frame2)
                    # img.save(output_dir / f"w_{frame_idx:06d}.png")
                    # tensor_frames.append(tensor_frame2)
                else:
                    print("we shouldn't be here...returning original tensor_frame")
                    if tensor_frame is not None:
                        print("tensor_frame is not None")
                    img = to_pil_image(tensor_frame)
                    img.save(output_dir / f"{frame_idx:06d}.png")
                    return tensor_frame
            else:
                image_frame.save(output_dir / f"{frame_idx:06d}.png")
                return tensor_frame
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            return None


    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)

        if isinstance(images, list):
            if self.remove_background:
                images = [self.remove_bg(img) for img in images]
            transformed_images = [transform(img) for img in tqdm(images, desc="Augmenting Images")]
            ret_tensor = torch.stack(transformed_images, dim=0)
        else:
            if self.remove_background:
                try:
                    images = self.remove_bg(images)
                except Exception as e:
                    pass
            ret_tensor = transform(images)

        return ret_tensor, images



    def save_video(self, frames, output_path, fps=30):
        print(f"Saving video with {len(frames)} frames to {output_path}")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to other codecs if needed
        height, width, _ = np.array(frames[0]).shape
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            frame = np.array(frame)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert to BGR format

        out.release()
        print(f"Video saved to {output_path}")




    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_id = self.video_ids[index]
        # Use next item in the list for video_id_star, wrap around if at the end
        video_id_star = self.video_ids_star[(index + 1) % len(self.video_ids_star)]
        vid_pil_image_list = self.load_and_process_video(os.path.join(self.video_dir, f"{video_id}.mp4"))
        vid_pil_image_list_star = self.load_and_process_video(os.path.join(self.video_dir, f"{video_id_star}.mp4"))

        sample = {
            "video_id": video_id,
            "source_frames": vid_pil_image_list,
            "driving_frames": self.driving_vid_pil_image_list,
            "video_id_star": video_id_star,
            "source_frames_star": vid_pil_image_list_star,
            "driving_frames_star": self.driving_vid_pil_image_list_star,
        }
        return sample
