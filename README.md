## IMPORTANT
My VASA hack project https://github.com/johndpope/vasa-1-hack has running /training code stage 1 (megaportraits) - with hot fixes
[https://github.com/johndpope/VASA-1-hack/blob/main/train_stage_1.py
](https://github.com/johndpope/VASA-1-hack/commit/430947d9707777d2ed9d38183e523d31c13054eb)





# MegaPortrait - SamsungLabs AI - Russia 
Implementation of Megaportrait using Claude Opus


All models / code is in model.py


![Image](diagram.jpeg)


memory debug
```shell
    mprof run train.py
```
or just
```shell
    python train.py
```



###  UPDATES

- Save / restore checkpoint) specify in config ./configs/training/stage10base.yaml to restore checkpoint
- auto crop video frames to sweet spot 
- tensorboard losses
- LPIPS
- additional imagepyramide from one shot view code for loss - (this broke things..)




### EmoDataset

[warp / crop / spline / remove background / transforms](EmoDataset.md)

## Training Data (☢️ dont need this yet.)

- **Total Videos:** 35,000 facial videos
- **Total Size:** 40GB


### Training Strategy
for now - to simplify problem - use the 4 videos in junk folder. 
once models are validated - can point the video_dir to above torrent
```yaml
 # video_dir:  '/Downloads/CelebV-HQ/celebvhq/35666'  
  video_dir: './junk'
```
the preprocessing is taking 1-2 mins for each video - I add some saving to npz format for faster reloading.


### Torrent Download

You can download the dataset via the provided magnet link or by visiting [Academic Torrents](https://academictorrents.com/details/843b5adb0358124d388c4e9836654c246b988ff4).

```plaintext
magnet:?xt=urn:btih:843b5adb0358124d388c4e9836654c246b988ff4&dn=CelebV-HQ&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=https%3A%2F%2Fipv6.academictorrents.com%2Fannounce.php
```



### Implemented Functionality / Descriptions

#### Base Model (`Gbase`)
- **Description**: Responsible for creating the foundational neural head avatar at a medium resolution of \(512 x 512\). Uses volumetric features to encode appearance and latent descriptors to encode motion.
- **Components**:
  - **Appearance Encoder (`Eapp`)**: Encodes the appearance of the source frame into volumetric features and a global descriptor.
    ```python
    class Eapp(nn.Module):
        # Architecture details omitted for brevity
    ```
  - **Motion Encoder (`Emtn`)**: Encodes the motion from both source and driving images into head rotations, translations, and latent expression descriptors.
    ```python
    class Emtn(nn.Module):
        # Architecture details omitted for brevity
    ```
  - **Warping Generators (`Wsrc_to_can` and `Wcan_to_drv`)**: Removes motion from the source and imposes driver motion onto canonical features.
    ```python
    class WarpGenerator(nn.Module):
        # Architecture details omitted for brevity
    ```
  - **3D Convolutional Network (`G3D`)**: Processes canonical volumetric features.
    ```python
    class G3D(nn.Module):
        # Architecture details omitted for brevity
    ```
  - **2D Convolutional Network (`G2D`)**: Projects 3D features into 2D and generates the output image.
    ```python
    class G2D(nn.Module):
        # Architecture details omitted for brevity
    ```

#### High-Resolution Model (`GHR`)
- **Description**: Enhances the resolution of the base model output from \(512 \times 512\) to \(1024 \times 1024\) using a high-resolution dataset of photographs.
- **Components**:
  - **Encoder**: Takes the base model output and produces a 3D feature tensor.
    ```python
    class EncoderHR(nn.Module):
        # Architecture details omitted for brevity
    ```
  - **Decoder**: Converts the 3D feature tensor to a high-resolution image.
    ```python
    class DecoderHR(nn.Module):
        # Architecture details omitted for brevity
    ```

#### Student Model (`Student`)
- **Description**: A distilled version of the high-resolution model for real-time applications. Trained to mimic the full model’s predictions but runs faster and is limited to a predefined number of avatars.
- **Components**:
  - **ResNet18 Encoder**: Encodes the input image.
    ```python
    class ResNet18(nn.Module):
        # Architecture details omitted for brevity
    ```
  - **Generator with SPADE Normalization Layers**: Generates the final output image. Each SPADE block uses tensors specific to an avatar.
    ```python
    class SPADEGenerator(nn.Module):
        # Architecture details omitted for brevity
    ```

#### Gaze and Blink Loss Model
- **Description**: Computes the gaze and blink loss using a pretrained face mesh from MediaPipe and a custom network. The gaze loss uses MAE and MSE, while the blink loss uses binary cross-entropy.
- **Components**:
  - **Backbone (VGG16)**: Extracts features from the eye images.
    ```python
    class VGG16Backbone(nn.Module):
        # Architecture details omitted for brevity
    ```
  - **Keypoint Network**: Processes 2D keypoints.
    ```python
    class KeypointNet(nn.Module):
        # Architecture details omitted for brevity
    ```
  - **Gaze Head**: Predicts gaze direction.
    ```python
    class GazeHead(nn.Module):
        # Architecture details omitted for brevity
    ```
  - **Blink Head**: Predicts blink probability.
    ```python
    class BlinkHead(nn.Module):
        # Architecture details omitted for brevity
    ```

#### Training Functions
- **`train_base(cfg, Gbase, Dbase, dataloader)`**: Trains the base model using perceptual, adversarial, and cycle consistency losses.
  ```python
  def train_base(cfg, Gbase, Dbase, dataloader):
      # Training code omitted for brevity
  ```
- **`train_hr(cfg, GHR, Dhr, dataloader)`**: Trains the high-resolution model using super-resolution objectives and adversarial losses.
  ```python
  def train_hr(cfg, GHR, Dhr, dataloader):
      # Training code omitted for brevity
  ```
- **`train_student(cfg, Student, GHR, dataloader)`**: Distills the high-resolution model into a student model for faster inference.
  ```python
  def train_student(cfg, Student, GHR, dataloader):
      # Training code omitted for brevity
  ```

#### Training Pipeline
- **Data Augmentation**: Applies random horizontal flips, color jitter, and other augmentations to the input images.
- **Optimizers**: Uses AdamW optimizer with cosine learning rate scheduling for both base and high-resolution models.
- **Losses**:
  - **Perceptual Loss**: Matches the content and facial appearance between predicted and ground-truth images.
  - **Adversarial Loss**: Ensures the realism of predicted images using a multi-scale patch discriminator.
  - **Cycle Consistency Loss**: Prevents appearance leakage through the motion descriptor.

#### Main Function
- **Description**: Sets up the dataset and data loaders, initializes the models, and calls the training functions for base, high-resolution, and student models.
- **Implementation**:
  ```python
  def main(cfg: OmegaConf) -> None:
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")

      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.5], [0.5]),
          transforms.RandomHorizontalFlip(),
          transforms.ColorJitter()
      ])

      dataset = EMODataset(
          use_gpu=use_cuda,
          width=cfg.data.train_width,
          height=cfg.data.train_height,
          n_sample_frames=cfg.training.n_sample_frames,
          sample_rate=cfg.training.sample_rate,
          img_scale=(1.0, 1.0),
          video_dir=cfg.training.video_dir,
          json_file=cfg.training.json_file,
          transform=transform
      )

      dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

      Gbase = model.Gbase()
      Dbase = model.Discriminator()
      train_base(cfg, Gbase, Dbase, dataloader)

      GHR = model.GHR()
      GHR.Gbase.load_state_dict(Gbase.state_dict())
      Dhr = model.Discriminator()
      train_hr(cfg, GHR, Dhr, dataloader)

      Student = model.Student(num_avatars=100)
      train_student(cfg, Student, GHR, dataloader)

      torch.save(Gbase.state_dict(), 'Gbase.pth')
      torch.save(GHR.state_dict(), 'GHR.pth')
      torch.save(Student.state_dict(), 'Student.pth')

  if __name__ == "__main__":
      config = OmegaConf.load("./configs/training/stage1-base.yaml")
      main(config)
  ```


rome/losses - cherry picked from 
https://github.com/SamsungLabs/rome





wget 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
extract to state_dicts


# RT-GENE (Real-Time Gaze Estimation) - couldn't get working
```python
git clone https://github.com/Tobias-Fischer/rt_gene.git
cd rt_gene/rt_gene
pip install .
```
