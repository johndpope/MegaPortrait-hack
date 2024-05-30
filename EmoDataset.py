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
# import face_alignment

class EMODataset(Dataset):
    def __init__(self, use_gpu:False, sample_rate: int, n_sample_frames: int, width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None):
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.video_dir = video_dir
        self.transform = transform
        self.stage = stage

        # Reduce 512 images -> 256
        self.pixel_transform = transforms.Compose(
            [
                # transforms.Resize((256, 256)), - just go HQ
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]), - this makes picture go red
            ]
        )

        self.drop_ratio = drop_ratio
        with open(json_file, 'r') as f:
            self.celebvhq_info = json.load(f)

        # self.video_ids = list(self.celebvhq_info['clips'].keys())
        self.video_ids = ["M2Ohb0FAaJU_1"]
        self.use_gpu = use_gpu

        decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        self.ctx = decord.cpu()

        # DRIVING VIDEO
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
        
        return len(self.video_ids)

    def augmentation(self, images, transform, state=None):
            if state is not None:
                torch.set_rng_state(state)
            if isinstance(images, List):
                transformed_images = [transform(img) for img in images]
                ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
            else:
                ret_tensor = transform(images)  # (c, h, w)
            return ret_tensor
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        print("__getitem__")
        video_id = self.video_ids[index]
        mp4_path = os.path.join(self.video_dir, f"{video_id}.mp4")


        video_reader = VideoReader(mp4_path, ctx=self.ctx)
        video_length = len(video_reader)
        

        vid_pil_image_list = []
        keypoints_list = []
        
        for frame_idx in range(video_length):

            # Read frame and convert to PIL Image
            frame = Image.fromarray(video_reader[frame_idx].numpy())

            # Transform the frame
            state = torch.get_rng_state()
            pixel_values_frame = self.augmentation(frame, self.pixel_transform, state)
            vid_pil_image_list.append(pixel_values_frame)

        # Convert list of lists to a tensor
        sample = {
            "video_id": video_id,
            "source_frames": vid_pil_image_list,
            "driving_frames": self.driving_vid_pil_image_list,
            "keypoints": keypoints_list
        }
        return sample