from moviepy.editor import VideoFileClip, ImageSequenceClip
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
import os
from typing import List, Tuple, Dict, Any
import io
import numpy as np
import random
from tqdm import tqdm
import cv2
from pathlib import Path
from torchvision.transforms.functional import to_tensor
from torchaudio.io import StreamReader
import face_recognition
from rembg import remove
from multiprocessing import Pool
import cupy as cp
from u2net import U2NET



# https://drive.usercontent.google.com/download?id=1IG3HdpcRiDoWNookbncQjeaPN28t90yW&export=download&authuser=1

class EMODataset(Dataset):
    def __init__(self, use_gpu: bool, sample_rate: int, n_sample_frames: int, width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None, remove_background=False, use_greenscreen=False, apply_crop_warping=False, num_videos=100):
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

        # vids = list(self.celebvhq_info['clips'].keys())
        # random_video_id = random.choice(vids)
        # driving_star = os.path.join(self.video_dir, f"{random_video_id}.mp4")
        # print("driving_star:", driving_star)

        # TODO - make this more dynamic
        driving = os.path.join(self.video_dir, "-2KGPYEFnsU_11.mp4")
        self.driving_vid_pil_image_list = self.load_and_process_video(driving)
        self.video_ids = ["M2Ohb0FAaJU_1"]  # list(self.celebvhq_info['clips'].keys())
        self.video_ids_star = ["-1eKufUP5XQ_4"]  # list(self.celebvhq_info['clips'].keys())
        driving_star = os.path.join(self.video_dir, "-2KGPYEFnsU_8.mp4")
        self.driving_vid_pil_image_list_star = self.load_and_process_video(driving_star)

    def __len__(self) -> int:
        return len(self.video_ids)

    def remove_bg(self, image):
        model = U2NET(model_path="./u2net_portrait.pth", cuda=self.use_gpu)
        bg_removed_image = model.remove_bg(cp.asnumpy(image))
        bg_removed_image = cp.array(bg_removed_image)

        if self.use_greenscreen:
            green_screen = cp.zeros_like(bg_removed_image)
            green_screen[..., 1] = 255  # Green color
            final_image = cp.where(bg_removed_image[..., 3:] > 0, bg_removed_image, green_screen)
        else:
            final_image = bg_removed_image

        final_image = final_image[..., :3]  # Convert to RGB format
        return final_image

    def piecewise_affine_transform(self, image, src_points, dst_points):
        transformed_image = cp.zeros_like(image)
        for i in range(len(src_points) - 1):
            src_tri = np.array([src_points[i], src_points[i + 1], src_points[(i + 2) % len(src_points)]])
            dst_tri = np.array([dst_points[i], dst_points[i + 1], dst_points[(i + 2) % len(dst_points)]])
            M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
            mask = cp.zeros_like(image)
            cv2.fillConvexPoly(mask, np.int32(dst_tri), (1, 1, 1))
            warped = cv2.warpAffine(cp.asnumpy(image), M, (image.shape[1], image.shape[0]))
            transformed_image += mask * cp.array(warped)
        return transformed_image

    def warp_and_crop_face(self, image_tensor, video_name, frame_idx, transform=None, output_dir="output_images", warp_strength=0.01, apply_warp=False):
        print("frame_idx:", frame_idx)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx}.png")

        if os.path.exists(output_path):
            existing_image = cp.array(cv2.imread(output_path, cv2.IMREAD_UNCHANGED))
            return to_tensor(existing_image)

        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)

        bg_removed_image = self.remove_bg(image_tensor)
        bg_removed_image_rgb = cp.asnumpy(bg_removed_image[..., :3])
        face_locations = face_recognition.face_locations(bg_removed_image_rgb)

        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            face_width = right - left
            face_height = bottom - top
            pad_width = int(face_width * 0.5)
            pad_height = int(face_height * 0.5)
            left = max(0, left - pad_width)
            top = max(0, top - pad_height)
            right = min(bg_removed_image.shape[1], right + pad_width)
            bottom = min(bg_removed_image.shape[0], bottom + pad_height)
            face_image = bg_removed_image[top:bottom, left:right]

            if apply_warp:
                rows, cols = face_image.shape[:2]
                src_points = np.array([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
                dst_points = src_points + np.random.randn(4, 2) * (rows * warp_strength)
                warped_face_array = self.piecewise_affine_transform(face_image, src_points, dst_points)
                warped_face_image = warped_face_array
            else:
                warped_face_image = face_image

            if transform:
                warped_face_image_rgb = cp.asnumpy(warped_face_image[..., :3])
                warped_face_tensor = transform(warped_face_image_rgb)
                return warped_face_tensor

            return to_tensor(warped_face_image)

        else:
            return None

    def load_and_process_video(self, video_path: str) -> List[torch.Tensor]:
        # Extract video ID from the path
        video_id = Path(video_path).stem
        output_dir =  Path(self.video_dir + "/" + video_id)
        output_dir.mkdir(exist_ok=True)

        processed_frames = []
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
                    frames = video_chunk[0]
                    all_frames = frames

            for idx, frame in enumerate(all_frames):
                result = self.process_frame((frame, idx, output_dir, video_path))
                if result is not None:
                    tensor_frames.append(result)
            if tensor_frames:
                np.savez_compressed(tensor_file_path, *[tensor_frame.cpu().numpy() for tensor_frame in tensor_frames])
                print(f"Processed tensors saved to file: {tensor_file_path}")
            else:
                print(f"No valid tensor frames processed for video {video_path}")

        return tensor_frames

    def process_frame(self, args):
        frame, frame_idx, output_dir, video_path = args
        print("idx:", frame_idx)
        try:
            state = torch.get_rng_state()
            tensor_frame, _ = self.augmentation(frame, self.pixel_transform, state)

            if self.apply_crop_warping:
                transform = transforms.Compose([
                    transforms.Resize((self.width, self.height)),
                    transforms.ToTensor(),
                ])
                video_name = Path(video_path).stem
                tensor_frame1 = self.warp_and_crop_face(tensor_frame, video_name, frame_idx, transform, apply_warp=False)

                if tensor_frame1 is not None:
                    img = cp.asnumpy(tensor_frame1.permute(1, 2, 0).numpy())
                    cv2.imwrite(output_dir / f"{frame_idx:06d}.png", img)
                    return tensor_frame1
                else:
                    img = cp.asnumpy(tensor_frame.permute(1, 2, 0).numpy())
                    cv2.imwrite(output_dir / f"{frame_idx:06d}.png", img)
                    return tensor_frame
            else:
                img = cp.asnumpy(tensor_frame.permute(1, 2, 0).numpy())
                cv2.imwrite(output_dir / f"{frame_idx:06d}.png", img)
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
            transformed_images = [transform(cp.asnumpy(img)) for img in tqdm(images, desc="Augmenting Images")]
            ret_tensor = torch.stack(transformed_images, dim=0)
        else:
            if self.remove_background:
                images = self.remove_bg(images)
            ret_tensor = transform(cp.asnumpy(images))

        return ret_tensor, images

    def save_video(self, frames, output_path, fps=30):
        print(f"Saving video with {len(frames)} frames to {output_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = np.array(frames[0]).shape
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            frame = np.array(frame)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
        print(f"Video saved to {output_path}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_id = self.video_ids[index]
        video_id_star = self.video_ids_star[(index + 1) % len(self.video_ids_star)]
        vid_cp_image_list = self.load_and_process_video((video_id, index))
        vid_cp_image_list_star = self.load_and_process_video((video_id_star, index))

        sample = {
            "video_id": video_id,
            "source_frames": vid_cp_image_list,
            "driving_frames": self.driving_vid_cp_image_list,
            "video_id_star": video_id_star,
            "source_frames_star": vid_cp_image_list_star,
            "driving_frames_star": self.driving_vid_cp_image_list_star,
        }
        return sample
