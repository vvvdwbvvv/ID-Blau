import argparse
import datetime
import json
import logging
import os
import random
import time
from itertools import islice

import cv2
import numpy as np
import pyiqa
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataloader import Flow_Loader
from rectified_flow_train import ConditionalRectifiedFlow, build_rf_config
from rectified_flow_train import NCSNpp
from utils.flow_viz import flow_to_image
from utils.set_condition import select_condition_strategy
from utils.utils import (
    AverageMeter,
    count_parameters,
    judge_and_remove_module_dict,
    same_seed,
    tensor2cv,
)


def get_arg(saved_args, name, default=None):
    if hasattr(saved_args, name):
        return getattr(saved_args, name)
    if isinstance(saved_args, dict):
        return saved_args.get(name, default)
    return default


def set_arg_default(saved_args, name, value):
    if not hasattr(saved_args, name):
        setattr(saved_args, name, value)


def pad_to_multiple(input_tensor, multiple, mode="reflect"):
    _, _, h, w = input_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode=mode)
    return input_tensor, h, w


def build_model_from_checkpoint(checkpoint, device, sample_timesteps=None, sampler=None):
    model_args = checkpoint.get("args")
    if model_args is None:
        raise ValueError("Rectified-flow checkpoint must contain saved training args.")

    model_args.device = device
    if sample_timesteps is not None:
        model_args.sample_timesteps = sample_timesteps

    # Older checkpoints may not contain every RF-specific CLI option.
    set_arg_default(model_args, "rf_noise_scale", 1.0)
    set_arg_default(model_args, "rf_t_eps", 1e-3)
    set_arg_default(model_args, "rf_t_scale", 999.0)
    set_arg_default(model_args, "criterion", "l2")
    set_arg_default(model_args, "rf_sampler", "euler")
    if sampler is not None:
        model_args.rf_sampler = sampler

    rf_config = build_rf_config(model_args)
    net = NCSNpp(rf_config).to(device)
    model = ConditionalRectifiedFlow(
        net,
        img_channels=3,
        criterion=get_arg(model_args, "criterion", "l2"),
        device=device,
        sample_steps=get_arg(model_args, "sample_timesteps", sample_timesteps or 50),
        init_type="gaussian",
        noise_scale=get_arg(model_args, "rf_noise_scale", 1.0),
        t_eps=get_arg(model_args, "rf_t_eps", 1e-3),
        t_scale=get_arg(model_args, "rf_t_scale", 999.0),
    ).to(device)

    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(judge_and_remove_module_dict(state))
    return model, model_args


@torch.no_grad()
def sample_rectified_flow(
    model,
    condition,
    device,
    sample_timesteps=None,
    tqdm_visible=False,
    method="euler",
    pad_multiple=128,
    pad_mode="reflect",
):
    condition, h, w = pad_to_multiple(condition, pad_multiple, mode=pad_mode)
    output = model.sample(
        condition=condition,
        sample_timesteps=sample_timesteps,
        device=device,
        tqdm_visible=tqdm_visible,
        method=method,
    )
    return output[:, :, :h, :w]


@torch.no_grad()
def valid(model, dataloader_val, sample_timesteps, device, sampler, valid_iters=None, title=None):
    model.eval()
    psnr_func = pyiqa.create_metric("psnr", device=device)
    lpips_func = pyiqa.create_metric("lpips", device=device)
    niqe_func = pyiqa.create_metric("niqe", device=device)
    total_val_psnr = AverageMeter()
    total_val_lpips = AverageMeter()
    total_val_niqe = AverageMeter()

    if valid_iters:
        tq = tqdm.tqdm(islice(dataloader_val, valid_iters), total=valid_iters)
    else:
        tq = tqdm.tqdm(dataloader_val, total=len(dataloader_val))
    tq.set_description("Validation")

    start_time = time.time()
    for sample in tq:
        blur, sharp = sample["blur"].to(device), sample["sharp"].to(device)
        flow = sample["flow"].to(device)
        condition = torch.cat([sharp, flow], dim=1)
        output = sample_rectified_flow(
            model,
            condition=condition,
            sample_timesteps=sample_timesteps,
            device=device,
            tqdm_visible=False,
            method=sampler,
            pad_multiple=args.pad_multiple,
            pad_mode=args.pad_mode,
        ).clamp(-0.5, 0.5)

        output_metric = output.detach() + 0.5
        blur_metric = blur.detach() + 0.5
        psnr = torch.mean(psnr_func(output_metric, blur_metric)).item()
        lpips = torch.mean(lpips_func(output_metric, blur_metric)).item()
        niqe = torch.mean(niqe_func(output_metric)).item()
        total_val_psnr.update(psnr)
        total_val_lpips.update(lpips)
        total_val_niqe.update(niqe)
        tq.set_postfix(
            LPIPS=total_val_lpips.avg,
            PSNR=total_val_psnr.avg,
            NIQE=total_val_niqe.avg,
        )

    elapsed_time = time.time() - start_time
    time_str = str(datetime.timedelta(seconds=elapsed_time)).split(".")[0]
    logging.info("-----------EVAL------------")
    logging.info(f"Title : {title}")
    logging.info(f"sample_timesteps : {sample_timesteps}")
    logging.info(f"sampler : {sampler}")
    logging.info(f"The program's running time is (h:m:s) : {time_str}")
    logging.info(
        f"PSNR : {total_val_psnr.avg:.4f}, "
        f"LPIPS : {total_val_lpips.avg:.4f}, "
        f"NIQE : {total_val_niqe.avg:.4f}"
    )


@torch.no_grad()
def val_save_image(model, dir_path, dataset, sample_timesteps, device, sampler, val_num=3, val_idxs=None):
    dir_path = os.path.join(dir_path, "images")
    os.makedirs(dir_path, exist_ok=True)
    model.eval()

    if val_idxs is None:
        val_idxs = random.sample(range(0, len(dataset)), val_num)
    for idx in val_idxs:
        sample = dataset[idx]
        save_sharp_path = os.path.join(dir_path, "sharp")
        os.makedirs(save_sharp_path, exist_ok=True)
        save_image(
            sample["sharp"].squeeze(0).cpu() + 0.5,
            os.path.join(save_sharp_path, f"{idx:05d}.png"),
        )

        save_blur_path = os.path.join(dir_path, "blur")
        os.makedirs(save_blur_path, exist_ok=True)
        save_image(
            sample["blur"].squeeze(0).cpu() + 0.5,
            os.path.join(save_blur_path, f"{idx:05d}.png"),
        )

        sharp = sample["sharp"].unsqueeze(0).to(device)
        flow = sample["flow"].unsqueeze(0).to(device)
        condition = torch.cat([sharp, flow], dim=1)
        output = sample_rectified_flow(
            model,
            condition=condition,
            sample_timesteps=sample_timesteps,
            device=device,
            tqdm_visible=True,
            method=sampler,
            pad_multiple=args.pad_multiple,
            pad_mode=args.pad_mode,
        ).clamp(-0.5, 0.5)

        save_dir_path = os.path.join(dir_path, "output")
        os.makedirs(save_dir_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir_path, f"{idx:05d}.png"), tensor2cv(output + 0.5))

        flow_np = flow.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        flow_x = flow_np[:, :, 0] * flow_np[:, :, 2]
        flow_y = flow_np[:, :, 1] * flow_np[:, :, 2]
        optical_flow = np.stack((flow_x, flow_y), axis=-1)
        flo = flow_to_image(optical_flow, norm=1)

        flow_dir_path = os.path.join(dir_path, "flow")
        os.makedirs(flow_dir_path, exist_ok=True)
        cv2.imwrite(os.path.join(flow_dir_path, f"{idx:05d}.png"), flo[:, :, [2, 1, 0]])


def detect_last_completed_idx(dir_path, generate_num):
    """
    Auto-detect the last completed index by checking the output directory structure.
    Looks for completed folders with all generated files.
    """
    blur_path = os.path.join(dir_path, "blur")
    
    if not os.path.exists(blur_path):
        return -1
    
    # Get all subdirectories in blur folder (format: 00000, 00001, etc.)
    completed_idxs = []
    for folder_name in os.listdir(blur_path):
        folder_path = os.path.join(blur_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        try:
            idx = int(folder_name)
            # Check if all generated images exist
            generated_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
            if len(generated_files) == generate_num:
                completed_idxs.append(idx)
        except (ValueError, OSError):
            continue
    
    if completed_idxs:
        return max(completed_idxs)
    return -1


def load_checkpoint(checkpoint_path):
    """Load checkpoint file with progress information"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"processed_indices": [], "last_idx": -1}


def save_checkpoint(checkpoint_path, data):
    """Save checkpoint file with progress information"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)


@torch.no_grad()
def generate_dataset(
    model,
    dir_path,
    dataset,
    sample_timesteps,
    strategy_setting,
    device,
    sampler,
    generate_num=5,
    save_npy=False,
    resume_idx=None,
):
    sharp_path = os.path.join(dir_path, "sharp")
    blur_path = os.path.join(dir_path, "blur")
    condition_path = os.path.join(dir_path, "condition")
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(sharp_path, exist_ok=True)
    os.makedirs(blur_path, exist_ok=True)
    os.makedirs(condition_path, exist_ok=True)

    # Setup checkpoint
    checkpoint_path = os.path.join(dir_path, ".generation_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)

    if "TURN" not in strategy_setting:
        strategy = strategy_setting[:]
    else:
        strategy_list = strategy_setting[:]
        strategy_list.remove("TURN")
        if "FIXED" in strategy_setting:
            strategy_list.remove("FIXED")

    model.eval()
    
    # Determine starting index
    if resume_idx is not None:
        start_idx = resume_idx
        print(f"📍 Resuming from user-specified index: {resume_idx}")
    else:
        # Auto-detect last completed index
        last_completed = detect_last_completed_idx(dir_path, generate_num)
        start_idx = last_completed + 1
        if last_completed >= 0:
            print(f"✅ Auto-detected last completed index: {last_completed}")
            print(f"📍 Resuming from index: {start_idx}")
        else:
            print(f"📍 Starting from beginning (index 0)")
    
    total_count = len(dataset)
    remaining = total_count - start_idx
    print(f"📊 Progress: {start_idx}/{total_count} items completed | {remaining} remaining")
    print("-" * 70)
    
    tq = tqdm.tqdm(range(start_idx, len(dataset)), initial=start_idx, total=len(dataset))
    tq.set_description("Generate images")
    
    for idx in tq:
        try:
            sample = dataset[idx]
            sharp_idx_path = os.path.join(sharp_path, f"{idx:05d}")
            os.makedirs(sharp_idx_path, exist_ok=True)
            save_image(
                sample["sharp"].squeeze(0).cpu() + 0.5,
                os.path.join(sharp_idx_path, "sharp.png"),
            )

            blur_idx_path = os.path.join(blur_path, f"{idx:05d}")
            os.makedirs(blur_idx_path, exist_ok=True)

            if save_npy:
                condition_idx_path = os.path.join(condition_path, f"{idx:05d}")
                os.makedirs(condition_idx_path, exist_ok=True)

            change_base = random.randint(0, 100) if "FIXED" in strategy_setting else 0
            for index in range(generate_num):
                sharp = sample["sharp"].unsqueeze(0).to(device)
                flow = sample["flow"].clone().unsqueeze(0).to(device)
                choice_num = index if "FIXED" in strategy_setting else None
                if "TURN" in strategy_setting:
                    strategy = [strategy_list[(idx + index) % len(strategy_list)]]
                new_flow = select_condition_strategy(
                    flow,
                    strategy=strategy,
                    choice_num=choice_num,
                    change_base=change_base,
                )
                condition = torch.cat([sharp, new_flow], dim=1)
                output = sample_rectified_flow(
                    model,
                    condition=condition,
                    sample_timesteps=sample_timesteps,
                    device=device,
                    method=sampler,
                    pad_multiple=args.pad_multiple,
                    pad_mode=args.pad_mode,
                ).clamp(-0.5, 0.5)

                cv2.imwrite(
                    os.path.join(blur_idx_path, f"{index:05d}.png"),
                    tensor2cv(output + 0.5),
                )

                if save_npy:
                    condition_np = new_flow.squeeze(0).cpu().numpy()
                    np.save(os.path.join(condition_idx_path, f"{index:05d}.npy"), condition_np)

            # Update checkpoint after successful completion of idx
            if idx not in checkpoint["processed_indices"]:
                checkpoint["processed_indices"].append(idx)
            checkpoint["last_idx"] = idx
            save_checkpoint(checkpoint_path, checkpoint)
            
            # Update progress bar description
            completed = idx + 1 - start_idx
            tq.set_postfix({"Completed": f"{completed}/{remaining}"})
            
        except Exception as e:
            print(f"\n❌ Error processing index {idx}: {str(e)}")
            print(f"Checkpoint saved. You can resume from index {idx + 1}")
            save_checkpoint(checkpoint_path, checkpoint)
            raise

    print("-" * 70)
    print(f"✨ Generation complete! All {total_count} items processed.")
    print(f"📁 Output saved to: {dir_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default="./dataset/GOPRO_Large", type=str)
    parser.add_argument("--dir_path", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--flow_data_path", default="./dataset/GOPRO_flow", type=str)
    parser.add_argument("--flow_norm", default=True, type=bool)
    parser.add_argument("--title", default="None", type=str)
    parser.add_argument(
        "--type",
        default="generate_dataset",
        type=str,
        choices=["generate_dataset", "image"] + pyiqa.list_models(),
    )
    parser.add_argument("--dataset", default="train", type=str, choices=["train", "test"])
    parser.add_argument("--val_num", default=5, type=int)
    parser.add_argument(
        "--strategy",
        default=[],
        type=str,
        choices=[
            "O",
            "M10",
            "M20",
            "M30",
            "M40",
            "M60",
            "M80",
            "ALLM",
            "ALLO",
            "RO",
            "30O",
            "60O",
            "FIXED",
            "TURN",
        ],
        nargs="+",
    )
    parser.add_argument("--sample_timesteps", default=50, type=int)
    parser.add_argument("--generate_num", default=5, type=int)
    parser.add_argument("--valid_iters", default=None, type=int)
    parser.add_argument("--crop_size", default=None, type=int)
    parser.add_argument("--save_npy", default=False, type=bool)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--rf_sampler", default=None, choices=["euler", "heun", "rk45"], type=str)
    parser.add_argument("--pad_multiple", default=128, type=int)
    parser.add_argument(
        "--pad_mode", default="reflect", type=str, choices=["reflect", "replicate"]
    )
    parser.add_argument(
        "--resume_idx",
        default=None,
        type=int,
        help="Resume from specific index (optional; auto-detection used if not specified)"
    )

    args = parser.parse_args()
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
    )
    sampler = args.rf_sampler or get_arg(model_args, "rf_sampler", "euler")

    os.makedirs(args.dir_path, exist_ok=True)

    if args.dataset == "train":
        dataset = Flow_Loader(
            data_path=args.data_path,
            flow_path=args.flow_data_path,
            mode="train",
            crop_size=args.crop_size,
            flow_norm=args.flow_norm,
        )
    elif args.dataset == "test":
        dataset = Flow_Loader(
            data_path=args.data_path,
            flow_path=args.flow_data_path,
            mode="test",
            crop_size=args.crop_size,
            flow_norm=args.flow_norm,
        )
    else:
        raise ValueError("Invalid dataset type (only train and test)")

    print("device:", device)
    print(f"args: {args}")
    print(f"checkpoint args: {model_args}")
    print(f"model parameters: {count_parameters(model)}")

    if args.type == "generate_dataset":
        print(f"strategy: {args.strategy}")
        generate_dataset(
            model,
            args.dir_path,
            dataset,
            sample_timesteps=args.sample_timesteps,
            generate_num=args.generate_num,
            strategy_setting=args.strategy,
            save_npy=args.save_npy,
            device=device,
            sampler=sampler,
            resume_idx=args.resume_idx,
        )
    elif args.type in pyiqa.list_models():
        logging.basicConfig(
            filename=os.path.join(args.dir_path, "eval.log"),
            format="%(asctime)s | %(levelname)s : %(message)s",
            encoding="utf-8",
            level=logging.INFO,
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s : %(message)s"))
        logging.getLogger("").addHandler(console)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )
        valid(
            model,
            dataloader,
            sample_timesteps=args.sample_timesteps,
            device=device,
            sampler=sampler,
            valid_iters=args.valid_iters,
            title=args.title,
        )
    elif args.type == "image":
        val_save_image(
            model,
            args.dir_path,
            dataset,
            sample_timesteps=args.sample_timesteps,
            val_num=args.val_num,
            val_idxs=None,
            device=device,
            sampler=sampler,
        )
