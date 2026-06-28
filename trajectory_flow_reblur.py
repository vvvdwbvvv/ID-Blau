import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


"""
Trajectory Flow Reblur
----------------------

This file is intentionally standalone.  It does not import the old
rectified_flow_train.py / rectified_flow_inference.py implementation.

The code follows the module-like design in the pasted note:

    S -> trajectory flow matching -> trajectory field T
      -> trajectory-to-kernel projection -> physical renderer -> reblur B

The default training path is video-free: it uses only paired sharp/blur images,
procedurally sampled synthetic trajectories for flow-matching pretraining, and a
real-pair reblur consistency loss.  The older condition-map-supervised path is
kept behind --trajectory_supervision condition for prototype/debug runs.

When that prototype path is used, ID-Blau's 3-channel blur condition map is
converted to a straight exposure path from:

    flow[0:2] = direction, flow[2] = normalized magnitude.

If a future dataset stores true curved trajectories, replace only
ConditionMapToTrajectoryTarget; the flow model and renderer can stay the same.
"""


def set_seed(seed):
    # Keep experiments reproducible across Python, NumPy, and PyTorch.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_train_logging(output_dir):
    log_path = Path(output_dir) / "train.log"
    logging.basicConfig(
        filename=str(log_path),
        format="%(asctime)s %(levelname)s:%(message)s",
        encoding="utf-8",
        level=logging.INFO,
        force=True,
    )
    return log_path


def require_cv2():
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required for dataset/image I/O. Install opencv-python "
            "or run this file inside the project environment that already has cv2."
        )


def image_to_tensor(image):
    # Convert RGB uint8/float image to CHW tensor in the repo's usual [-0.5, 0.5] range.
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return tensor / 255.0 - 0.5


def tensor_to_uint8(tensor):
    # Convert a [-0.5, 0.5] BCHW or CHW tensor to an RGB uint8 HWC image.
    if tensor.ndim == 4:
        tensor = tensor[0]
    image = (tensor.detach().cpu().clamp(-0.5, 0.5) + 0.5) * 255.0
    image = image.permute(1, 2, 0).numpy()
    return image.round().astype(np.uint8)


def save_rgb(path, tensor):
    require_cv2()
    ensure_dir(Path(path).parent)
    image = tensor_to_uint8(tensor)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def rotate_vectors_90(flow_hw3, k):
    # Rotate the x/y direction channels consistently with np.rot90(image, k).
    if k == 0:
        return flow_hw3

    vectors = flow_hw3[..., :2].copy()
    angle = math.radians(90 * k)
    rot = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    flat = vectors.reshape(-1, 2)
    flow_hw3[..., :2] = (rot @ flat.T).T.reshape(vectors.shape)
    return flow_hw3


class SharpBlurDataset(Dataset):
    """
    Video-free paired dataset:

        data_path/mode/video/sharp/*.png
        data_path/mode/video/blur/*.png

    It intentionally does not load flow maps or trajectory labels.
    """

    def __init__(self, data_path, mode="train", crop_size=None, augment=True):
        self.data_path = Path(data_path)
        self.mode = mode
        self.crop_size = crop_size
        self.augment = augment and crop_size is not None and mode == "train"
        self.samples = []

        mode_dir = self.data_path / mode
        if not mode_dir.exists():
            raise FileNotFoundError(f"Missing data directory: {mode_dir}")

        for video_dir in sorted(mode_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            sharp_dir = video_dir / "sharp"
            blur_dir = video_dir / "blur"
            if not sharp_dir.exists() or not blur_dir.exists():
                continue
            for blur_file in sorted(blur_dir.glob("*.png")):
                sharp_file = sharp_dir / blur_file.name
                if sharp_file.exists():
                    self.samples.append((sharp_file, blur_file))

        if not self.samples:
            raise RuntimeError(f"No sharp/blur pairs found under data_path={data_path}")

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        require_cv2()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    def _random_crop(self, sharp, blur):
        h, w = sharp.shape[:2]
        crop = self.crop_size
        if crop is None:
            return sharp, blur
        if h < crop or w < crop:
            raise ValueError(f"crop_size={crop} is larger than image size {(h, w)}")

        top = random.randint(0, h - crop)
        left = random.randint(0, w - crop)
        sharp = sharp[top : top + crop, left : left + crop]
        blur = blur[top : top + crop, left : left + crop]
        return sharp, blur

    def _augment(self, sharp, blur):
        if random.randint(0, 1):
            sharp = np.fliplr(sharp).copy()
            blur = np.fliplr(blur).copy()

        if random.randint(0, 1):
            sharp = np.flipud(sharp).copy()
            blur = np.flipud(blur).copy()

        k = random.randint(0, 3)
        if k:
            sharp = np.rot90(sharp, k).copy()
            blur = np.rot90(blur, k).copy()

        return sharp, blur

    def __getitem__(self, idx):
        sharp_path, blur_path = self.samples[idx]
        sharp = self._load_rgb(sharp_path)
        blur = self._load_rgb(blur_path)

        sharp, blur = self._random_crop(sharp, blur)
        if self.augment:
            sharp, blur = self._augment(sharp, blur)

        return {
            "sharp": image_to_tensor(sharp),
            "blur": image_to_tensor(blur),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class SharpFlowDataset(Dataset):
    """
    Prototype dataset for sharp images plus blur-condition maps.

    The folder contract matches the existing repo:

        data_path/mode/video/sharp/*.png
        data_path/mode/video/blur/*.png
        flow_path/mode/video/*.npy

    The loaded blur image is used for preview/reblur loss.  The flow map is only
    for the legacy condition-supervised trajectory target.
    """

    def __init__(
        self,
        data_path,
        flow_path,
        mode="train",
        crop_size=None,
        augment=True,
        flow_norm=True,
        flow_norm_num=147.0,
    ):
        self.data_path = Path(data_path)
        self.flow_path = Path(flow_path)
        self.mode = mode
        self.crop_size = crop_size
        self.augment = augment and crop_size is not None and mode == "train"
        self.flow_norm = flow_norm
        self.flow_norm_num = flow_norm_num
        self.samples = []

        mode_flow_dir = self.flow_path / mode
        if not mode_flow_dir.exists():
            raise FileNotFoundError(f"Missing flow directory: {mode_flow_dir}")

        for video_dir in sorted(mode_flow_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video = video_dir.name
            for flow_file in sorted(video_dir.glob("*.npy")):
                image_name = flow_file.with_suffix(".png").name
                sharp_file = self.data_path / mode / video / "sharp" / image_name
                blur_file = self.data_path / mode / video / "blur" / image_name
                if sharp_file.exists() and blur_file.exists():
                    self.samples.append((sharp_file, blur_file, flow_file))

        if not self.samples:
            raise RuntimeError(
                f"No samples found under data_path={data_path} and flow_path={flow_path}"
            )

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        require_cv2()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    def _load_condition_map(self, path):
        raw = np.load(path).astype(np.float32)
        if raw.shape[0] != 3:
            raise ValueError(f"Expected flow map shape (3,H,W), got {raw.shape} at {path}")

        direction = raw[:2]
        norm = np.sqrt(np.sum(direction**2, axis=0, keepdims=True))
        direction = direction / np.maximum(norm, 1e-6)

        magnitude = raw[2:3]
        if self.flow_norm:
            # Match the original loader's convention: divide by 147 and clip to [0, 1].
            magnitude = np.clip(magnitude / self.flow_norm_num, 0.0, 1.0)

        return np.concatenate([direction, magnitude], axis=0)

    def _random_crop(self, sharp, blur, condition):
        h, w = sharp.shape[:2]
        crop = self.crop_size
        if crop is None:
            return sharp, blur, condition
        if h < crop or w < crop:
            raise ValueError(f"crop_size={crop} is larger than image size {(h, w)}")

        top = random.randint(0, h - crop)
        left = random.randint(0, w - crop)
        sharp = sharp[top : top + crop, left : left + crop]
        blur = blur[top : top + crop, left : left + crop]
        condition = condition[:, top : top + crop, left : left + crop]
        return sharp, blur, condition

    def _augment(self, sharp, blur, condition):
        condition_hw3 = condition.transpose(1, 2, 0)

        if random.randint(0, 1):
            # Horizontal flip changes the sign of the x-direction channel.
            sharp = np.fliplr(sharp).copy()
            blur = np.fliplr(blur).copy()
            condition_hw3 = np.fliplr(condition_hw3).copy()
            condition_hw3[..., 0] *= -1.0

        if random.randint(0, 1):
            # Vertical flip changes the sign of the y-direction channel.
            sharp = np.flipud(sharp).copy()
            blur = np.flipud(blur).copy()
            condition_hw3 = np.flipud(condition_hw3).copy()
            condition_hw3[..., 1] *= -1.0

        k = random.randint(0, 3)
        if k:
            # Rotate images and rotate direction vectors by the same angle.
            sharp = np.rot90(sharp, k).copy()
            blur = np.rot90(blur, k).copy()
            condition_hw3 = np.rot90(condition_hw3, k).copy()
            condition_hw3 = rotate_vectors_90(condition_hw3, k)

        return sharp, blur, condition_hw3.transpose(2, 0, 1).copy()

    def __getitem__(self, idx):
        sharp_path, blur_path, flow_path = self.samples[idx]
        sharp = self._load_rgb(sharp_path)
        blur = self._load_rgb(blur_path)
        condition = self._load_condition_map(flow_path)

        sharp, blur, condition = self._random_crop(sharp, blur, condition)
        if self.augment:
            sharp, blur, condition = self._augment(sharp, blur, condition)

        return {
            "sharp": image_to_tensor(sharp),
            "blur": image_to_tensor(blur),
            "condition": torch.from_numpy(condition).float(),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class ConditionMapToTrajectoryTarget(nn.Module):
    """
    Module adapter: current ID-Blau condition map -> trajectory field T.

    The theory wants T_i = {Delta_i(tau), tau in [0, 1]}.
    Current data gives one direction and one magnitude per pixel, so this module
    constructs a straight-line trajectory sampled at trajectory_steps exposure
    times.  The output is normalized motion, not pixels:

        T in shape (B, 2 * trajectory_steps, H, W)

    PhysicalReblurRenderer later multiplies by max_motion_pixels.
    """

    def __init__(self, trajectory_steps=7):
        super().__init__()
        if trajectory_steps < 2:
            raise ValueError("trajectory_steps must be >= 2")
        self.trajectory_steps = trajectory_steps

    def forward(self, condition_map):
        direction = condition_map[:, 0:2]
        magnitude = condition_map[:, 2:3].clamp(min=0.0)

        # Centered exposure samples avoid a constant image shift in the rendered blur.
        exposure = torch.linspace(
            -0.5,
            0.5,
            self.trajectory_steps,
            device=condition_map.device,
            dtype=condition_map.dtype,
        )
        trajectory = direction.unsqueeze(2) * magnitude.unsqueeze(2)
        trajectory = trajectory * exposure.view(1, 1, self.trajectory_steps, 1, 1)
        b, _, steps, h, w = trajectory.shape
        return trajectory.reshape(b, 2 * steps, h, w)


class SyntheticTrajectorySampler(nn.Module):
    """
    Video-free synthetic trajectory prior for Stage-1 flow matching.

    It samples a mostly global exposure direction with low-frequency spatial
    jitter.  The output matches ConditionMapToTrajectoryTarget:

        T in shape (B, 2 * trajectory_steps, H, W)

    Values are normalized motion; TrajectoryToKernelProjection later converts
    them to pixels using max_motion_pixels.
    """

    def __init__(
        self,
        trajectory_steps=7,
        min_magnitude=0.05,
        max_magnitude=1.0,
        local_jitter=0.15,
        lowres_grid=16,
    ):
        super().__init__()
        if trajectory_steps < 2:
            raise ValueError("trajectory_steps must be >= 2")
        if min_magnitude < 0 or max_magnitude <= 0 or min_magnitude > max_magnitude:
            raise ValueError("invalid synthetic trajectory magnitude range")
        self.trajectory_steps = trajectory_steps
        self.min_magnitude = float(min_magnitude)
        self.max_magnitude = float(max_magnitude)
        self.local_jitter = float(local_jitter)
        self.lowres_grid = int(lowres_grid)

    def forward(self, sharp):
        b, _, h, w = sharp.shape
        device, dtype = sharp.device, sharp.dtype

        angle = torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * (2.0 * math.pi)
        direction = torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)
        direction = direction.expand(b, 2, h, w)

        magnitude = torch.empty(b, 1, 1, 1, device=device, dtype=dtype).uniform_(
            self.min_magnitude, self.max_magnitude
        )
        magnitude = magnitude.expand(b, 1, h, w)

        if self.local_jitter > 0:
            grid = max(2, min(self.lowres_grid, h, w))
            jitter = torch.randn(b, 2, grid, grid, device=device, dtype=dtype)
            jitter = F.interpolate(jitter, size=(h, w), mode="bilinear", align_corners=False)
            direction = F.normalize(direction + self.local_jitter * jitter, dim=1, eps=1e-6)

            mag_jitter = torch.randn(b, 1, grid, grid, device=device, dtype=dtype)
            mag_jitter = F.interpolate(
                mag_jitter, size=(h, w), mode="bilinear", align_corners=False
            )
            magnitude = magnitude * (1.0 + 0.25 * self.local_jitter * torch.tanh(mag_jitter))
            magnitude = magnitude.clamp(self.min_magnitude, self.max_magnitude)

        exposure = torch.linspace(
            -0.5,
            0.5,
            self.trajectory_steps,
            device=device,
            dtype=dtype,
        )
        trajectory = direction.unsqueeze(2) * magnitude.unsqueeze(2)
        trajectory = trajectory * exposure.view(1, 1, self.trajectory_steps, 1, 1)
        return trajectory.reshape(b, 2 * self.trajectory_steps, h, w)


def sinusoidal_time_embedding(t, dim):
    # Standard transformer/DDPM-style embedding for continuous t in [0, 1].
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def group_count(channels):
    # Pick a GroupNorm group count that divides channels.
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class TimeConvBlock(nn.Module):
    """
    Small time-conditioned convolution block.

    This is the building block for Module 1's velocity field v_theta(x_t,t|S).
    """

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Inject the scalar ODE time into every spatial location.
        h = h + self.time_proj(time_emb).view(time_emb.shape[0], -1, 1, 1)

        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(h)


class TrajectoryVelocityUNet(nn.Module):
    """
    Module 1: conditional trajectory flow model.

    Input:
        x_t   - noisy/interpolated trajectory field
        sharp - sharp image condition S
        t     - flow-matching time

    Output:
        predicted velocity v_theta(x_t, t | S), same shape as x_t.
    """

    def __init__(
        self,
        trajectory_channels,
        sharp_channels=3,
        base_channels=64,
        time_dim=128,
    ):
        super().__init__()
        self.time_dim = time_dim
        in_channels = trajectory_channels + sharp_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.enc1 = TimeConvBlock(in_channels, base_channels, time_dim)
        self.enc2 = TimeConvBlock(base_channels, base_channels * 2, time_dim)
        self.enc3 = TimeConvBlock(base_channels * 2, base_channels * 4, time_dim)
        self.mid = TimeConvBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = TimeConvBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = TimeConvBlock(base_channels * 2, base_channels, time_dim)
        self.out = nn.Conv2d(base_channels, trajectory_channels, 3, padding=1)

    def forward(self, trajectory_t, sharp, t):
        time_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        x = torch.cat([trajectory_t, sharp], dim=1)

        e1 = self.enc1(x, time_emb)
        e2 = self.enc2(F.avg_pool2d(e1, 2), time_emb)
        e3 = self.enc3(F.avg_pool2d(e2, 2), time_emb)
        mid = self.mid(e3, time_emb)

        u2 = self.up2(mid)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), time_emb)

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), time_emb)
        return self.out(d1)


class TrajectoryToKernelProjection(nn.Module):
    """
    Module 2: trajectory -> implicit pixel-wise kernel representation.

    Materializing K in R^{H x W x k x k} is expensive.  This module therefore
    represents each per-pixel kernel by the exposure-time sampling grids used by
    the renderer.  Averaging samples along those grids is equivalent to splatting
    the trajectory into a local motion-blur kernel and applying it to S.
    """

    def __init__(self, trajectory_steps, max_motion_pixels=32.0, clamp_trajectory=True):
        super().__init__()
        self.trajectory_steps = trajectory_steps
        self.max_motion_pixels = float(max_motion_pixels)
        self.clamp_trajectory = clamp_trajectory

    def forward(self, trajectory, height, width):
        b, channels, h, w = trajectory.shape
        expected_channels = 2 * self.trajectory_steps
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} trajectory channels, got {channels}")
        if h != height or w != width:
            raise ValueError("Trajectory and image sizes must match")

        trajectory = trajectory.view(b, 2, self.trajectory_steps, h, w)
        if self.clamp_trajectory:
            # Keep generated trajectories inside the normalized motion range used in training.
            trajectory = trajectory.clamp(-1.0, 1.0)

        offsets = trajectory * self.max_motion_pixels

        ys = torch.linspace(-1.0, 1.0, h, device=trajectory.device, dtype=trajectory.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=trajectory.device, dtype=trajectory.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([xx, yy], dim=-1).view(1, 1, h, w, 2)

        # grid_sample expects normalized source coordinates.  A positive dx means
        # "sample S at x + dx" for the output pixel.
        dx = offsets[:, 0] * (2.0 / max(width - 1, 1))
        dy = offsets[:, 1] * (2.0 / max(height - 1, 1))
        offset_grid = torch.stack([dx, dy], dim=-1)
        return base_grid + offset_grid


class PhysicalReblurRenderer(nn.Module):
    """
    Module 3: physical reblur rendering.

    Given S and the implicit kernel grids from Module 2, render:

        B_i = average_tau S(i + Delta_i(tau)).

    This is an exposure integration approximation of local non-uniform motion
    blur.  It is differentiable because it uses grid_sample.
    """

    def forward(self, sharp, kernel_grids):
        b, c, h, w = sharp.shape
        steps = kernel_grids.shape[1]
        rendered = []
        for step in range(steps):
            rendered.append(
                F.grid_sample(
                    sharp,
                    kernel_grids[:, step],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
            )
        return torch.stack(rendered, dim=0).mean(dim=0)


class IdentityImageRefinement(nn.Module):
    """
    Module 4 placeholder: optional image-space realism correction.

    The pasted design correctly treats this as secondary.  This implementation
    keeps it as an identity map so the main novelty remains Module 1-3.  A later
    image-space flow model can replace this class without changing the pipeline.
    """

    def forward(self, coarse_blur):
        return coarse_blur


class TrajectoryFlowReblur(nn.Module):
    """
    Full modular framework:

        Module 1: flow-match trajectory distribution p_theta(T | S)
        Module 2: project T to implicit pixel-wise kernels
        Module 3: render physical reblur from S and kernels
        Module 4: optional image-space refinement, identity by default
    """

    def __init__(
        self,
        velocity_model,
        target_builder,
        synthetic_sampler,
        projector,
        renderer,
        refiner=None,
        noise_scale=1.0,
        t_eps=1e-3,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.target_builder = target_builder
        self.synthetic_sampler = synthetic_sampler
        self.projector = projector
        self.renderer = renderer
        self.refiner = refiner or IdentityImageRefinement()
        self.noise_scale = noise_scale
        self.t_eps = t_eps

    @staticmethod
    def _expand_time(t, x):
        return t.view(-1, *([1] * (x.ndim - 1)))

    def build_target_trajectory(self, condition_map):
        return self.target_builder(condition_map)

    def sample_synthetic_trajectory(self, sharp):
        return self.synthetic_sampler(sharp)

    def compute_target_flow_matching_loss(self, sharp, target):
        # Draw x_0 from the simple base distribution.
        source = torch.randn_like(target) * self.noise_scale

        # Linear probability path: x_t = (1 - t) x_0 + t x_1.
        t = torch.rand(target.shape[0], device=target.device) * (1.0 - self.t_eps) + self.t_eps
        t_img = self._expand_time(t, target)
        trajectory_t = (1.0 - t_img) * source + t_img * target

        # Flow matching target velocity: u_t = x_1 - x_0.
        target_velocity = target - source
        pred_velocity = self.velocity_model(trajectory_t, sharp, t)
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss, {
            "target_trajectory": target,
            "trajectory_t": trajectory_t,
            "pred_velocity": pred_velocity,
            "t": t,
        }

    def compute_flow_matching_loss(self, sharp, condition_map):
        # Legacy prototype path: build x_1 = T from the existing condition map.
        target = self.build_target_trajectory(condition_map)
        return self.compute_target_flow_matching_loss(sharp, target)

    def compute_synthetic_flow_matching_loss(self, sharp):
        # Video-free path: sample x_1 = T from a procedural motion prior.
        target = self.sample_synthetic_trajectory(sharp)
        return self.compute_target_flow_matching_loss(sharp, target)

    def render_from_trajectory(self, sharp, trajectory):
        _, _, h, w = sharp.shape
        grids = self.projector(trajectory, h, w)
        coarse = self.renderer(sharp, grids)
        return self.refiner(coarse)

    def _solve_trajectory(self, sharp, sample_steps=50, sampler="heun"):
        # Start from Gaussian trajectory noise and solve dT_t/dt = v_theta(T_t,t|S).
        b, _, h, w = sharp.shape
        channels = 2 * self.projector.trajectory_steps
        x = (
            torch.randn((b, channels, h, w), device=sharp.device, dtype=sharp.dtype)
            * self.noise_scale
        )
        dt = (1.0 - self.t_eps) / sample_steps

        for i in range(sample_steps):
            t_value = self.t_eps + i * dt
            t_next = min(t_value + dt, 1.0)
            t = torch.full((b,), t_value, device=sharp.device, dtype=sharp.dtype)
            velocity = self.velocity_model(x, sharp, t)

            if sampler == "euler":
                x = x + dt * velocity
            elif sampler == "heun":
                x_pred = x + dt * velocity
                t2 = torch.full((b,), t_next, device=sharp.device, dtype=sharp.dtype)
                velocity_next = self.velocity_model(x_pred, sharp, t2)
                x = x + 0.5 * dt * (velocity + velocity_next)
            else:
                raise ValueError("sampler must be 'euler' or 'heun'")

        return x

    def sample_trajectory_trainable(self, sharp, sample_steps=50, sampler="heun"):
        return self._solve_trajectory(sharp, sample_steps=sample_steps, sampler=sampler)

    @torch.no_grad()
    def sample_trajectory(self, sharp, sample_steps=50, sampler="heun"):
        return self._solve_trajectory(sharp, sample_steps=sample_steps, sampler=sampler)

    @torch.no_grad()
    def sample_reblur(self, sharp, sample_steps=50, sampler="heun"):
        trajectory = self.sample_trajectory(sharp, sample_steps=sample_steps, sampler=sampler)
        return self.render_from_trajectory(sharp, trajectory), trajectory


def build_pipeline(config, device):
    trajectory_channels = 2 * config["trajectory_steps"]
    velocity_model = TrajectoryVelocityUNet(
        trajectory_channels=trajectory_channels,
        base_channels=config["base_channels"],
        time_dim=config["time_dim"],
    )
    pipeline = TrajectoryFlowReblur(
        velocity_model=velocity_model,
        target_builder=ConditionMapToTrajectoryTarget(config["trajectory_steps"]),
        synthetic_sampler=SyntheticTrajectorySampler(
            trajectory_steps=config["trajectory_steps"],
            min_magnitude=config.get("synthetic_min_magnitude", 0.05),
            max_magnitude=config.get("synthetic_max_magnitude", 1.0),
            local_jitter=config.get("synthetic_local_jitter", 0.15),
            lowres_grid=config.get("synthetic_lowres_grid", 16),
        ),
        projector=TrajectoryToKernelProjection(
            trajectory_steps=config["trajectory_steps"],
            max_motion_pixels=config["max_motion_pixels"],
            clamp_trajectory=config["clamp_trajectory"],
        ),
        renderer=PhysicalReblurRenderer(),
        refiner=IdentityImageRefinement(),
        noise_scale=config["noise_scale"],
        t_eps=config["t_eps"],
    )
    return pipeline.to(device)


def config_from_args(args):
    return {
        "trajectory_steps": args.trajectory_steps,
        "max_motion_pixels": args.max_motion_pixels,
        "clamp_trajectory": args.clamp_trajectory,
        "base_channels": args.base_channels,
        "time_dim": args.time_dim,
        "noise_scale": args.noise_scale,
        "t_eps": args.t_eps,
        "synthetic_min_magnitude": args.synthetic_min_magnitude,
        "synthetic_max_magnitude": args.synthetic_max_magnitude,
        "synthetic_local_jitter": args.synthetic_local_jitter,
        "synthetic_lowres_grid": args.synthetic_lowres_grid,
    }


def save_checkpoint(path, pipeline, optimizer, epoch, global_step, config, args):
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "model_state": pipeline.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "args": vars(args),
        },
        path,
    )


def load_pipeline_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pipeline = build_pipeline(checkpoint["config"], device)
    pipeline.load_state_dict(checkpoint["model_state"])
    return pipeline, checkpoint


def make_dataset(args, mode, augment):
    crop_size = args.crop_size if mode == "train" else args.val_crop_size
    if args.trajectory_supervision == "condition":
        return SharpFlowDataset(
            data_path=args.data_path,
            flow_path=args.flow_data_path,
            mode=mode,
            crop_size=crop_size,
            augment=augment,
            flow_norm=args.flow_norm,
            flow_norm_num=args.flow_norm_num,
        )
    return SharpBlurDataset(
        data_path=args.data_path,
        mode=mode,
        crop_size=crop_size,
        augment=augment,
    )


@torch.no_grad()
def save_preview(pipeline, batch, output_dir, epoch, sample_steps, sampler):
    # Save side-by-side evidence: sharp, real blur, synthetic/condition render, sampled reblur.
    pipeline.eval()
    sharp = batch["sharp"][:1]
    blur = batch["blur"][:1]
    device = next(pipeline.parameters()).device
    sharp = sharp.to(device)
    blur = blur.to(device)

    if "condition" in batch:
        condition = batch["condition"][:1].to(device)
        target_trajectory = pipeline.build_target_trajectory(condition)
        target_name = "rendered_condition_trajectory.png"
    else:
        target_trajectory = pipeline.sample_synthetic_trajectory(sharp)
        target_name = "rendered_synthetic_trajectory.png"
    target_render = pipeline.render_from_trajectory(sharp, target_trajectory)
    sampled_reblur, _ = pipeline.sample_reblur(
        sharp,
        sample_steps=sample_steps,
        sampler=sampler,
    )

    preview_dir = Path(output_dir) / "previews" / f"epoch_{epoch:05d}"
    save_rgb(preview_dir / "sharp.png", sharp)
    save_rgb(preview_dir / "real_blur.png", blur)
    save_rgb(preview_dir / target_name, target_render)
    save_rgb(preview_dir / "sampled_reblur.png", sampled_reblur)


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ensure_dir(args.output_dir)
    log_path = setup_train_logging(args.output_dir)
    if args.fm_loss_weight <= 0 and args.reblur_loss_weight <= 0:
        raise ValueError("At least one of --fm_loss_weight or --reblur_loss_weight must be > 0")

    train_set = make_dataset(args, mode="train", augment=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    config = config_from_args(args)
    pipeline = build_pipeline(config, device)
    optimizer = torch.optim.AdamW(
        pipeline.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        pipeline.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)

    with open(Path(args.output_dir) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logging.info("train_log: %s", log_path)
    logging.info("device: %s", device)
    logging.info("args: %s", vars(args))
    logging.info("config: %s", config)
    logging.info("train_samples: %d", len(train_set))
    logging.info("start_epoch: %d end_epoch: %d global_step: %d", start_epoch, args.epochs, global_step)

    for epoch in range(start_epoch, args.epochs + 1):
        pipeline.train()
        meter = 0.0
        metric_totals = {}
        num_batches = 0
        tq = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch_idx, batch in enumerate(tq, start=1):
            sharp = batch["sharp"].to(device, non_blocking=True)
            blur = batch["blur"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = sharp.new_tensor(0.0)
            log_values = {}

            if args.fm_loss_weight > 0:
                if args.trajectory_supervision == "condition":
                    condition = batch["condition"].to(device, non_blocking=True)
                    fm_loss, _ = pipeline.compute_flow_matching_loss(sharp, condition)
                else:
                    fm_loss, _ = pipeline.compute_synthetic_flow_matching_loss(sharp)
                loss = loss + args.fm_loss_weight * fm_loss
                log_values["fm"] = fm_loss.item()

            if args.reblur_loss_weight > 0:
                trajectory = pipeline.sample_trajectory_trainable(
                    sharp,
                    sample_steps=args.train_sample_steps,
                    sampler=args.sampler,
                )
                rendered_blur = pipeline.render_from_trajectory(sharp, trajectory)
                reblur_loss = F.l1_loss(rendered_blur, blur)
                loss = loss + args.reblur_loss_weight * reblur_loss
                log_values["reblur"] = reblur_loss.item()

                if args.trajectory_reg_weight > 0:
                    reg_loss = trajectory.abs().mean()
                    loss = loss + args.trajectory_reg_weight * reg_loss
                    log_values["reg"] = reg_loss.item()

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(pipeline.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            meter += loss.item()
            num_batches = batch_idx
            for name, value in log_values.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + value
            log_values["loss"] = meter / batch_idx
            log_values["step"] = global_step
            tq.set_postfix(log_values)

        if num_batches == 0:
            raise RuntimeError(
                "No training batches were produced. Lower --batch_size or add more training pairs."
            )

        epoch_metrics = {
            name: total / num_batches for name, total in metric_totals.items()
        }
        logging.info(
            "epoch=%d/%d step=%d loss=%.6f metrics=%s",
            epoch,
            args.epochs,
            global_step,
            meter / num_batches,
            epoch_metrics,
        )

        if args.preview_every > 0 and epoch % args.preview_every == 0:
            save_preview(
                pipeline,
                next(iter(train_loader)),
                args.output_dir,
                epoch,
                sample_steps=args.sample_steps,
                sampler=args.sampler,
            )

        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_checkpoint(
                Path(args.output_dir) / f"epoch_{epoch:05d}_trajectory_flow_reblur.pth",
                pipeline,
                optimizer,
                epoch,
                global_step,
                config,
                args,
            )
            save_checkpoint(
                Path(args.output_dir) / "last_trajectory_flow_reblur.pth",
                pipeline,
                optimizer,
                epoch,
                global_step,
                config,
                args,
            )
            logging.info("saved checkpoint for epoch=%d", epoch)


@torch.no_grad()
def sample(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    pipeline, _ = load_pipeline_from_checkpoint(args.checkpoint, device)
    pipeline.eval()

    dataset = make_dataset(args, mode=args.dataset, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "sharp")
    ensure_dir(output_dir / "sampled_reblur")
    if args.save_trajectory_npy:
        ensure_dir(output_dir / "trajectory")

    max_items = min(args.max_samples, len(dataset)) if args.max_samples else len(dataset)
    for item_idx, batch in enumerate(tqdm.tqdm(loader, total=max_items, desc="Sampling")):
        if item_idx >= max_items:
            break

        sharp = batch["sharp"].to(device)
        reblur, trajectory = pipeline.sample_reblur(
            sharp,
            sample_steps=args.sample_steps,
            sampler=args.sampler,
        )

        save_rgb(output_dir / "sharp" / f"{item_idx:05d}.png", sharp)
        save_rgb(output_dir / "sampled_reblur" / f"{item_idx:05d}.png", reblur)
        if args.save_trajectory_npy:
            np.save(
                output_dir / "trajectory" / f"{item_idx:05d}.npy",
                trajectory.squeeze(0).cpu().numpy(),
            )


def self_test(args):
    # Fast shape/gradient check that does not require the GoPro dataset.
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = config_from_args(args)
    pipeline = build_pipeline(config, device)

    sharp = torch.rand(2, 3, 64, 64, device=device) - 0.5
    blur = F.avg_pool2d(sharp, kernel_size=3, stride=1, padding=1)
    condition = torch.zeros(2, 3, 64, 64, device=device)
    condition[:, 0] = 1.0
    condition[:, 2] = 0.5

    fm_loss, pieces = pipeline.compute_synthetic_flow_matching_loss(sharp)
    trajectory_for_reblur = pipeline.sample_trajectory_trainable(
        sharp, sample_steps=2, sampler="euler"
    )
    rendered = pipeline.render_from_trajectory(sharp, trajectory_for_reblur)
    reblur_loss = F.l1_loss(rendered, blur)
    condition_loss, _ = pipeline.compute_flow_matching_loss(sharp, condition)
    loss = fm_loss + reblur_loss + 0.0 * condition_loss
    loss.backward()
    reblur, trajectory = pipeline.sample_reblur(sharp[:1], sample_steps=2, sampler="euler")

    print("self_test ok")
    print(f"loss={loss.item():.6f}")
    print(f"synthetic_fm_loss={fm_loss.item():.6f}")
    print(f"reblur_loss={reblur_loss.item():.6f}")
    print(f"target_trajectory={tuple(pieces['target_trajectory'].shape)}")
    print(f"sampled_trajectory={tuple(trajectory.shape)}")
    print(f"reblur={tuple(reblur.shape)}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Standalone Trajectory Flow Reblur framework")
    parser.add_argument("--mode", choices=["train", "sample", "self_test"], default="self_test")

    # Dataset paths.
    parser.add_argument("--data_path", default="./dataset/GOPRO_Large")
    parser.add_argument("--flow_data_path", default="./dataset/GOPRO_flow")
    parser.add_argument("--dataset", choices=["train", "test"], default="test")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--val_crop_size", type=int, default=None)
    parser.add_argument("--flow_norm", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--flow_norm_num", type=float, default=147.0)

    # Module sizes.
    parser.add_argument("--trajectory_steps", type=int, default=7)
    parser.add_argument("--max_motion_pixels", type=float, default=32.0)
    parser.add_argument("--clamp_trajectory", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=128)

    # Flow matching.
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--t_eps", type=float, default=1e-3)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["euler", "heun"], default="heun")
    parser.add_argument(
        "--trajectory_supervision",
        choices=["synthetic", "condition"],
        default="synthetic",
        help="synthetic is video-free; condition uses GOPRO_flow maps for prototype runs.",
    )
    parser.add_argument("--synthetic_min_magnitude", type=float, default=0.05)
    parser.add_argument("--synthetic_max_magnitude", type=float, default=1.0)
    parser.add_argument("--synthetic_local_jitter", type=float, default=0.15)
    parser.add_argument("--synthetic_lowres_grid", type=int, default=16)

    # Training.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fm_loss_weight", type=float, default=1.0)
    parser.add_argument("--reblur_loss_weight", type=float, default=1.0)
    parser.add_argument("--trajectory_reg_weight", type=float, default=1e-4)
    parser.add_argument("--train_sample_steps", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--preview_every", type=int, default=10)
    parser.add_argument("--resume", default=None)

    # Sampling/checkpoint output.
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="./experiments/trajectory_flow_reblur")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_trajectory_npy", action="store_true")

    # Runtime.
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "sample":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in sample mode")
        sample(args)
    elif args.mode == "self_test":
        self_test(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
