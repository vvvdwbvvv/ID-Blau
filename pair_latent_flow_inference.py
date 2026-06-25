import argparse
import os
import random
from types import SimpleNamespace

import cv2
import torch
import tqdm
from torchvision.utils import save_image

from dataloader import Multi_GoPro_Loader
from pair_latent_flow_model import PairLatentRectifiedFlow, build_pair_latent_rf_config
from rectified_flow_train import NCSNpp
from utils.utils import count_parameters, judge_and_remove_module_dict, same_seed, tensor2cv


def get_arg(saved_args, name, default=None):
    if hasattr(saved_args, name):
        return getattr(saved_args, name)
    if isinstance(saved_args, dict):
        return saved_args.get(name, default)
    return default


def set_arg_default(saved_args, name, value):
    if isinstance(saved_args, dict):
        saved_args.setdefault(name, value)
    elif not hasattr(saved_args, name):
        setattr(saved_args, name, value)


def apply_checkpoint_defaults(model_args):
    defaults = {
        "criterion": "l2",
        "crop_size": 256,
        "rf_image_size": None,
        "rf_noise_scale": 1.0,
        "rf_t_eps": 1e-3,
        "rf_t_scale": 999.0,
        "rf_sampler": "euler",
        "sample_timesteps": 50,
        "base_channels": 128,
        "channel_mults": (1, 1, 2, 2, 2, 2, 2),
        "num_res_blocks": 2,
        "dropout": 0.0,
        "attn_resolutions": (16,),
        "rf_resblock_type": "biggan",
        "rf_progressive": "output_skip",
        "rf_progressive_input": "none",
        "latent_dim": 64,
        "latent_map_channels": 8,
        "latent_encoder_channels": 32,
        "train_latent_perturb_std": 0.0,
        "path_type": "linear",
        "path_gamma": 0.25,
        "path_code_dim": 0,
        "train_path_code_std": 1.0,
        "path_bend_channels": 32,
        "path_bend_scale": 1.0,
        "pad_multiple": None,
        "pad_mode": "reflect",
    }
    for name, value in defaults.items():
        set_arg_default(model_args, name, value)


def build_model_from_checkpoint(
    checkpoint,
    device,
    sample_timesteps=None,
    sampler=None,
    pad_multiple=None,
    pad_mode=None,
):
    model_args = checkpoint.get("args")
    if model_args is None:
        raise ValueError("Pair-latent checkpoint must contain saved training args.")
    if isinstance(model_args, dict):
        model_args = SimpleNamespace(**model_args)

    apply_checkpoint_defaults(model_args)
    model_args.device = device
    if sample_timesteps is not None:
        model_args.sample_timesteps = sample_timesteps
    if sampler is not None:
        model_args.rf_sampler = sampler
    if pad_multiple is not None:
        model_args.pad_multiple = pad_multiple
    elif get_arg(model_args, "pad_multiple") is None:
        model_args.pad_multiple = get_arg(model_args, "crop_size", 256)
    if pad_mode is not None:
        model_args.pad_mode = pad_mode

    rf_config = build_pair_latent_rf_config(model_args)
    net = NCSNpp(rf_config).to(device)
    model = PairLatentRectifiedFlow(
        net,
        img_channels=3,
        criterion=get_arg(model_args, "criterion", "l2"),
        device=device,
        sample_steps=get_arg(model_args, "sample_timesteps", sample_timesteps or 50),
        init_type="gaussian",
        noise_scale=get_arg(model_args, "rf_noise_scale", 1.0),
        t_eps=get_arg(model_args, "rf_t_eps", 1e-3),
        t_scale=get_arg(model_args, "rf_t_scale", 999.0),
        pad_multiple=get_arg(model_args, "pad_multiple", 128),
        pad_mode=get_arg(model_args, "pad_mode", "reflect"),
        latent_dim=get_arg(model_args, "latent_dim", 64),
        latent_map_channels=get_arg(model_args, "latent_map_channels", 8),
        latent_encoder_channels=get_arg(model_args, "latent_encoder_channels", 32),
        train_latent_perturb_std=get_arg(model_args, "train_latent_perturb_std", 0.0),
        path_type=get_arg(model_args, "path_type", "linear"),
        path_gamma=get_arg(model_args, "path_gamma", 0.25),
        path_code_dim=get_arg(model_args, "path_code_dim", 0),
        train_path_code_std=get_arg(model_args, "train_path_code_std", 1.0),
        path_bend_channels=get_arg(model_args, "path_bend_channels", 32),
        path_bend_scale=get_arg(model_args, "path_bend_scale", 1.0),
    ).to(device)

    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(judge_and_remove_module_dict(state))
    return model, model_args


@torch.no_grad()
def make_variant_condition(
    model,
    sharp,
    blur,
    latent_perturb_std=0.0,
    path_code_std=1.0,
):
    base_latent = model.encode_latent(sharp, blur)
    if latent_perturb_std > 0:
        latent = base_latent + torch.randn_like(base_latent) * latent_perturb_std
    else:
        latent = base_latent
    path_code = model.sample_path_code(
        sharp.shape[0],
        sharp.device,
        std=path_code_std if model.path_type == "quadratic" else 0.0,
    )
    condition, _ = model.build_condition(
        sharp=sharp,
        latent=latent,
        path_code=path_code,
    )
    return condition


@torch.no_grad()
def generate_dataset(
    model,
    dir_path,
    dataset,
    sample_timesteps,
    device,
    sampler,
    generate_num=5,
    latent_perturb_std=0.1,
    path_code_std=1.0,
):
    sharp_path = os.path.join(dir_path, "sharp")
    blur_path = os.path.join(dir_path, "blur")
    real_blur_path = os.path.join(dir_path, "real_blur")
    os.makedirs(sharp_path, exist_ok=True)
    os.makedirs(blur_path, exist_ok=True)
    os.makedirs(real_blur_path, exist_ok=True)

    model.eval()
    tq = tqdm.tqdm(range(len(dataset)))
    tq.set_description("Generate pair-latent reblur images")
    for idx in tq:
        sample = dataset[idx]
        sharp_idx_path = os.path.join(sharp_path, f"{idx:05d}")
        blur_idx_path = os.path.join(blur_path, f"{idx:05d}")
        real_blur_idx_path = os.path.join(real_blur_path, f"{idx:05d}")
        os.makedirs(sharp_idx_path, exist_ok=True)
        os.makedirs(blur_idx_path, exist_ok=True)
        os.makedirs(real_blur_idx_path, exist_ok=True)

        save_image(
            sample["sharp"].cpu() + 0.5,
            os.path.join(sharp_idx_path, "sharp.png"),
        )
        save_image(
            sample["blur"].cpu() + 0.5,
            os.path.join(real_blur_idx_path, "blur.png"),
        )

        sharp = sample["sharp"].unsqueeze(0).to(device)
        blur = sample["blur"].unsqueeze(0).to(device)
        for variant_idx in range(generate_num):
            condition = make_variant_condition(
                model,
                sharp,
                blur,
                latent_perturb_std=latent_perturb_std,
                path_code_std=path_code_std,
            )
            output = model.sample(
                condition=condition,
                sample_timesteps=sample_timesteps,
                device=device,
                method=sampler,
            ).clamp(-0.5, 0.5)
            cv2.imwrite(
                os.path.join(blur_idx_path, f"{variant_idx:05d}.png"),
                tensor2cv(output + 0.5),
            )


@torch.no_grad()
def val_save_image(
    model,
    dir_path,
    dataset,
    sample_timesteps,
    device,
    sampler,
    val_num=5,
    generate_num=5,
    latent_perturb_std=0.1,
    path_code_std=1.0,
):
    dir_path = os.path.join(dir_path, "images")
    os.makedirs(dir_path, exist_ok=True)
    model.eval()

    val_idxs = random.sample(range(0, len(dataset)), min(val_num, len(dataset)))
    for idx in val_idxs:
        sample = dataset[idx]
        sample_dir = os.path.join(dir_path, f"{idx:05d}")
        output_dir = os.path.join(sample_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        save_image(sample["sharp"].cpu() + 0.5, os.path.join(sample_dir, "sharp.png"))
        save_image(sample["blur"].cpu() + 0.5, os.path.join(sample_dir, "blur.png"))

        sharp = sample["sharp"].unsqueeze(0).to(device)
        blur = sample["blur"].unsqueeze(0).to(device)
        for variant_idx in range(generate_num):
            condition = make_variant_condition(
                model,
                sharp,
                blur,
                latent_perturb_std=latent_perturb_std,
                path_code_std=path_code_std,
            )
            output = model.sample(
                condition=condition,
                sample_timesteps=sample_timesteps,
                device=device,
                tqdm_visible=True,
                method=sampler,
            ).clamp(-0.5, 0.5)
            cv2.imwrite(
                os.path.join(output_dir, f"{variant_idx:05d}.png"),
                tensor2cv(output + 0.5),
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default="./dataset/GOPRO_Large", type=str)
    parser.add_argument("--dir_path", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument(
        "--type",
        default="generate_dataset",
        choices=["generate_dataset", "image"],
        type=str,
    )
    parser.add_argument("--dataset", default="train", choices=["train", "test"])
    parser.add_argument("--val_num", default=5, type=int)
    parser.add_argument("--sample_timesteps", default=50, type=int)
    parser.add_argument("--generate_num", default=5, type=int)
    parser.add_argument("--crop_size", default=None, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--latent_perturb_std", default=0.1, type=float)
    parser.add_argument("--path_code_std", default=1.0, type=float)
    parser.add_argument("--rf_sampler", default=None, choices=["euler", "heun", "rk45"])
    parser.add_argument("--pad_multiple", default=None, type=int)
    parser.add_argument(
        "--pad_mode", default=None, type=str, choices=["reflect", "replicate"]
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dir_path is None:
        raise ValueError("--dir_path is required")
    if args.model_path is None:
        raise ValueError("--model_path is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    same_seed(args.seed)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model, model_args = build_model_from_checkpoint(
        checkpoint,
        device,
        sample_timesteps=args.sample_timesteps,
        sampler=args.rf_sampler,
        pad_multiple=args.pad_multiple,
        pad_mode=args.pad_mode,
    )
    sampler = args.rf_sampler or get_arg(model_args, "rf_sampler", "euler")

    os.makedirs(args.dir_path, exist_ok=True)
    dataset = Multi_GoPro_Loader(
        data_path=args.data_path,
        mode=args.dataset,
        crop_size=args.crop_size,
    )

    print("device:", device)
    print(f"args: {args}")
    print(f"checkpoint args: {model_args}")
    print(f"model parameters: {count_parameters(model)}")

    if args.type == "generate_dataset":
        generate_dataset(
            model,
            args.dir_path,
            dataset,
            sample_timesteps=args.sample_timesteps,
            device=device,
            sampler=sampler,
            generate_num=args.generate_num,
            latent_perturb_std=args.latent_perturb_std,
            path_code_std=args.path_code_std,
        )
    elif args.type == "image":
        val_save_image(
            model,
            args.dir_path,
            dataset,
            sample_timesteps=args.sample_timesteps,
            val_num=args.val_num,
            generate_num=args.generate_num,
            device=device,
            sampler=sampler,
            latent_perturb_std=args.latent_perturb_std,
            path_code_std=args.path_code_std,
        )
