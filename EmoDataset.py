from moviepy.editor import VideoFileClip, ImageSequenceClip
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
import os
from typing import List, Tuple, Dict, Any
from decord import VideoReader, cpu
from rembg import remove
import io
import numpy as np
import decord
import subprocess
from tqdm import tqdm
import cv2


class EMODataset(Dataset):
    def __init__(self, use_gpu: False, sample_rate: int, n_sample_frames: int, width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None, remove_background=False,use_greenscreen=False):
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

        with open(json_file, 'r') as f:
            self.celebvhq_info = json.load(f)

        self.use_gpu = use_gpu

        decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        self.ctx = cpu()


        # TODO - make this more dynamic
        self.driving_vid_pil_image_list = self.load_and_process_video("./junk/-2KGPYEFnsU_11.mp4")
        self.video_ids = ["M2Ohb0FAaJU_1"] # list(self.celebvhq_info['clips'].keys())
        self.video_ids_star = ["-1eKufUP5XQ_4"] #list(self.celebvhq_info['clips'].keys())
        self.driving_vid_pil_image_list_star = self.load_and_process_video("./junk/-2KGPYEFnsU_8.mp4")

    def __len__(self) -> int:
        return len(self.video_ids)

    def load_and_process_video(self, video_path: str) -> List[torch.Tensor]:
        processed_video_path = video_path.replace(".mp4", "_nobg.mp4")
        processed_frames = []
        tensor_frames = []
        if os.path.exists(processed_video_path):
            print(f"Loading processed video: {processed_video_path}")
            video_reader = VideoReader(processed_video_path, ctx=self.ctx)
            for frame_idx in tqdm(range(len(video_reader)), desc="Loading Processed Video Frames"):
                frame = Image.fromarray(video_reader[frame_idx].numpy())
                state = torch.get_rng_state()
                tensor_frame,image_frame = self.augmentation(frame, self.pixel_transform, state)
                tensor_frames.append(tensor_frame)
                
        else:
            print(f"Processing and saving video: {video_path}")
            video_reader = VideoReader(video_path, ctx=self.ctx)
            processed_frames = []
            for frame_idx in tqdm(range(len(video_reader)), desc="Processing Video Frames"):
                frame = Image.fromarray(video_reader[frame_idx].numpy())
                state = torch.get_rng_state()
                tensor_frame,image_frame = self.augmentation(frame, self.pixel_transform, state)
                processed_frames.append(image_frame)
                tensor_frames.append(tensor_frame)
            
            self.save_video(processed_frames, processed_video_path)

        return tensor_frames

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
                images = self.remove_bg(images)
            ret_tensor = transform(images)

        return ret_tensor,images


   

    # this will be easiest putting the background back in
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
            final_image = Image.alpha_composite(green_screen, bg_removed_image).convert("RGB")
        else:
            final_image = bg_removed_image.convert("RGB")

        return final_image

        

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

    def process_video(self, video_path):
        video_reader = VideoReader(video_path, ctx=self.ctx)
        processed_frames = []
        for frame_idx in range(len(video_reader)):
            frame = Image.fromarray(video_reader[frame_idx].numpy())
            state = torch.get_rng_state()
            pixel_values_frame = self.augmentation(frame, self.pixel_transform, state)
            processed_frames.append(pixel_values_frame)
        return processed_frames

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
