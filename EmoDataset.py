from moviepy.editor import VideoFileClip
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import json
import os
import decord
from typing import List, Tuple, Dict, Any
from decord import VideoReader,AVReader
from rembg import remove
import io 
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np

class EMODataset(Dataset):
    def __init__(self, use_gpu:False, cycle_consistency:False,sample_rate: int, n_sample_frames: int, width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None,remove_background=False):
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

        with open(json_file, 'r') as f:
            self.celebvhq_info = json.load(f)

        self.use_gpu = use_gpu

        decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        self.ctx = decord.cpu()

        if cycle_consistency:
            self.video_ids = list(self.celebvhq_info['clips'].keys())
            video_drv_reader = VideoReader("./junk/-2KGPYEFnsU_11.mp4", ctx=self.ctx)
        else:    
            self.video_ids = ["M2Ohb0FAaJU_1"]
            video_drv_reader = VideoReader("./junk/-2KGPYEFnsU_8.mp4", ctx=self.ctx)

        
        video_length = len(video_drv_reader)

        self.driving_vid_pil_image_list = []
        # keypoints_list = []
        
        for frame_idx in range(video_length):
            # Read frame and convert to PIL Image
            frame = Image.fromarray(video_drv_reader[frame_idx].numpy())
                        # Transform the frame
            state = torch.get_rng_state()
            pixel_values_frame = self.augmentation(frame, self.pixel_transform, state)
            self.driving_vid_pil_image_list.append(pixel_values_frame)
        print("driving video frames:",len(self.driving_vid_pil_image_list))

    def __len__(self) -> int:
        n = len(self.video_ids)
        print("n:",n)
        return n



    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        
        if isinstance(images, list):
            if self.remove_background:
                images = [self.remove_bg(img) for img in images]
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)
        else:
            if self.remove_background:
                images = self.remove_bg(images)
            ret_tensor = transform(images)
        
        return ret_tensor

    def remove_bg(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        bg_removed_bytes = remove(img_byte_arr)
        bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGB")
        return bg_removed_image

    def save_video(self, frames, output_path, fps=30):
        clip = ImageSequenceClip([np.array(frame) for frame in frames], fps=fps)
        clip.write_videofile(output_path, codec='libx264')

    def process_video(self, video_path):
        video_reader = VideoReader(video_path, ctx=self.ctx)
        video_length = len(video_reader)
        processed_frames = []
        
        for frame_idx in range(video_length):
            frame = Image.fromarray(video_reader[frame_idx].numpy())
            state = torch.get_rng_state()
            pixel_values_frame = self.augmentation(frame, self.pixel_transform, state)
            processed_frames.append(pixel_values_frame)
        
        return processed_frames
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        print("__getitem__")
        video_id = self.video_ids[index]
        mp4_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        processed_video_path = os.path.join(self.video_dir, f"{video_id}_nobg.mp4")

        # Check if processed video file exists
        if os.path.exists(processed_video_path):
            video_reader = VideoReader(processed_video_path, ctx=self.ctx)
        else:
            video_reader = VideoReader(mp4_path, ctx=self.ctx)
            processed_frames = []

            for frame_idx in range(len(video_reader)):
                frame = Image.fromarray(video_reader[frame_idx].numpy())
                if self.remove_background:
                    frame = self.remove_bg(frame)
                state = torch.get_rng_state()
                pixel_values_frame = self.augmentation(frame, self.pixel_transform, state)
                processed_frames.append(pixel_values_frame)

            # Save the processed video
            # if self.remove_background:
            #     self.save_video(processed_frames, processed_video_path) - broken
            
            vid_pil_image_list = processed_frames

        # Convert list of lists to a tensor
        sample = {
            "video_id": video_id,
            "source_frames": vid_pil_image_list,
            "driving_frames": self.driving_vid_pil_image_list,
        }
        return sample
