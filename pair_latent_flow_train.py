import argparse
import logging
import os
import random
from itertools import islice

import cv2
import pyiqa
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import Multi_GoPro_Loader
from pair_latent_flow_model import PairLatentRectifiedFlow, build_pair_latent_rf_config
from rectified_flow_train import NCSNpp
from utils.utils import (
    AverageMeter,
    count_parameters,
    judge_and_remove_module_dict,
    tensor2cv,
)


class PairLatentTrainer:
    def __init__(
        self,
        dataloader_train,
        dataloader_val,
        model,
        optimizer,
        scheduler,
        args,
        writer,
    ):
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.writer = writer
        self.epoch = 0
        self.global_step = 0
        self.sample_timesteps = args.sample_timesteps
        self.val_sample_timesteps = args.val_sample_timesteps
        self.device = args.device
        self.psnr_func = pyiqa.create_metric("psnr", device=self.device)
        self.lpips_func = pyiqa.create_metric("lpips", device=self.device)
        self.best_psnr = args.best_psnr if hasattr(args, "best_psnr") else 0
        self.grad_clip = args.grad_clip

    def _apply_warmup_lr(self):
        if self.args.warmup_steps <= 0 or self.global_step >= self.args.warmup_steps:
            return
        warmup_lr = self.args.init_lr * min(
            float(self.global_step + 1) / self.args.warmup_steps, 1.0
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = warmup_lr

    def train(self):
        print("Start_Epoch:", self.args.start_epoch)
        print("End_Epoch:", self.args.end_epoch)
        print("Model:", self.args.model_name)
        print(f"Optimizer:{self.optimizer.__class__.__name__}")
        print(
            f"Scheduler:{self.scheduler.__class__.__name__ if self.scheduler else None}"
        )
        print("Path_Type:", self.args.path_type)
        print("Sample_Timesteps:", self.sample_timesteps)
        print("Val_Sample_Timesteps:", self.val_sample_timesteps)
        print("Valid_Iters:", self.args.valid_iters)
        print("start train")

        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):
            self.epoch = epoch
            self._train_epoch()

            if (
                epoch % self.args.validation_epoch
            ) == 0 or epoch == self.args.end_epoch:
                self.valid(valid_iters=self.args.valid_iters)

            if (
                self.args.val_save_epochs > 0
                and epoch % self.args.val_save_epochs == 0
                or epoch == self.args.end_epoch
            ):
                self.val_save_image(
                    dir_path=self.args.dir_path, dataset=self.dataloader_val.dataset
                )

            self.save_model()

    def _train_epoch(self):
        tq = tqdm.tqdm(self.dataloader_train, total=len(self.dataloader_train))
        tq.set_description(f"Epoch [{self.epoch}/{self.args.end_epoch}] training")
        total_train_loss = AverageMeter()

        for sample in tq:
            self.model.train()
            self.optimizer.zero_grad()

            blur, sharp = (
                sample["blur"].to(self.device),
                sample["sharp"].to(self.device),
            )
            loss = self.model(x1=blur, sharp=sharp, blur=blur)
            loss.backward()

            self._apply_warmup_lr()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.global_step += 1
            total_train_loss.update(loss.detach().item())

            tq.set_postfix(
                {
                    "loss": total_train_loss.avg,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

        if self.scheduler:
            self.scheduler.step()
        self.writer.add_scalar("Loss/Train_loss", total_train_loss.avg, self.epoch)
        logging.info(
            f"Epoch [{self.epoch}/{self.args.end_epoch}]: "
            f"Train_loss: {total_train_loss.avg:.4f}"
        )

    @torch.no_grad()
    def _valid(self, sharp, blur):
        self.model.eval()
        condition, _ = self.model.build_condition(sharp=sharp, blur=blur)
        output = self.model.sample(
            condition=condition,
            sample_timesteps=self.val_sample_timesteps,
            device=self.device,
            method=self.args.rf_sampler,
        )
        output = output.clamp(-0.5, 0.5)
        output_metric = output.detach() + 0.5
        blur_metric = blur.detach() + 0.5
        psnr = torch.mean(self.psnr_func(output_metric, blur_metric)).item()
        lpips = torch.mean(self.lpips_func(output_metric, blur_metric)).item()
        return psnr, lpips

    @torch.no_grad()
    def valid(self, valid_iters=10):
        self.model.eval()
        total_val_psnr = AverageMeter()
        total_val_lpips = AverageMeter()
        tq = tqdm.tqdm(islice(self.dataloader_val, valid_iters), total=valid_iters)
        tq.set_description(f"Epoch [{self.epoch}/{self.args.end_epoch}] Validation")
        for sample in tq:
            blur, sharp = (
                sample["blur"].to(self.device),
                sample["sharp"].to(self.device),
            )
            psnr, lpips = self._valid(sharp, blur)
            total_val_psnr.update(psnr)
            total_val_lpips.update(lpips)
            tq.set_postfix(LPIPS=total_val_lpips.avg, PSNR=total_val_psnr.avg)

        self.writer.add_scalar("Val/Test_lpips", total_val_lpips.avg, self.epoch)
        self.writer.add_scalar("Val/Test_psnr", total_val_psnr.avg, self.epoch)
        logging.info(
            f"Validation Epoch [{self.epoch}/{self.args.end_epoch}]: "
            f"Test lpips: {total_val_lpips.avg:.4f} "
            f"Test psnr:{total_val_psnr.avg:.4f}"
        )

        if self.best_psnr < total_val_psnr.avg:
            self.best_psnr = total_val_psnr.avg
            self.args.best_psnr = self.best_psnr
            best_state = {"model_state": self.model.state_dict(), "args": self.args}
            torch.save(
                best_state,
                os.path.join(
                    self.args.dir_path, "best_{}.pth".format(self.args.model_name)
                ),
            )
            print("Saving model with best PSNR {:.3f}...".format(self.best_psnr))
            logging.info("Saving model with best PSNR {:.3f}...".format(self.best_psnr))

    def save_model(self):
        training_state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_psnr": self.best_psnr,
            "args": self.args,
        }
        should_save_last = (
            self.args.save_last_epoch > 0
            and self.epoch % self.args.save_last_epoch == 0
        )
        if should_save_last or self.epoch == self.args.end_epoch:
            torch.save(
                training_state,
                os.path.join(
                    self.args.dir_path, "last_{}.pth".format(self.args.model_name)
                ),
            )

        if (self.epoch % self.args.check_point_epoch) == 0:
            torch.save(
                training_state,
                os.path.join(
                    self.args.dir_path,
                    "epoch_{}_{}.pth".format(self.epoch, self.args.model_name),
                ),
            )

        if self.epoch == self.args.end_epoch:
            model_state = {"model_state": self.model.state_dict(), "args": self.args}
            torch.save(
                model_state,
                os.path.join(
                    self.args.dir_path, "final_{}.pth".format(self.args.model_name)
                ),
            )

    @torch.no_grad()
    def val_save_image(self, dir_path, dataset, val_num=3):
        os.makedirs(dir_path, exist_ok=True)
        self.model.eval()
        sample_count = min(val_num, len(dataset))
        for idx in random.sample(range(0, len(dataset)), sample_count):
            sample = dataset[idx]
            blur, sharp = (
                sample["blur"].unsqueeze(0).to(self.device),
                sample["sharp"].unsqueeze(0).to(self.device),
            )
            condition, _ = self.model.build_condition(sharp=sharp, blur=blur)
            output = self.model.sample(
                condition=condition,
                sample_timesteps=self.val_sample_timesteps,
                device=self.device,
                method=self.args.rf_sampler,
            )
            output = output.clamp(-0.5, 0.5)

            save_img_dir_path = os.path.join(dir_path, "visualization", "output")
            save_sharp_dir_path = os.path.join(dir_path, "visualization", "sharp")
            save_blur_dir_path = os.path.join(dir_path, "visualization", "blur")
            os.makedirs(save_img_dir_path, exist_ok=True)
            os.makedirs(save_sharp_dir_path, exist_ok=True)
            os.makedirs(save_blur_dir_path, exist_ok=True)

            cv2.imwrite(
                os.path.join(save_img_dir_path, f"{self.epoch:05d}_{idx:05d}.png"),
                tensor2cv(output + 0.5),
            )
            cv2.imwrite(
                os.path.join(save_sharp_dir_path, f"{self.epoch:05d}_{idx:05d}.png"),
                tensor2cv(sharp + 0.5),
            )
            cv2.imwrite(
                os.path.join(save_blur_dir_path, f"{self.epoch:05d}_{idx:05d}.png"),
                tensor2cv(blur + 0.5),
            )


def build_model(args, device):
    rf_config = build_pair_latent_rf_config(args)
    net = NCSNpp(rf_config).to(device)
    return PairLatentRectifiedFlow(
        net,
        img_channels=3,
        criterion=args.criterion,
        device=device,
        sample_steps=args.sample_timesteps,
        init_type="gaussian",
        noise_scale=args.rf_noise_scale,
        t_eps=args.rf_t_eps,
        t_scale=args.rf_t_scale,
        pad_multiple=args.pad_multiple,
        pad_mode=args.pad_mode,
        latent_dim=args.latent_dim,
        latent_map_channels=args.latent_map_channels,
        latent_encoder_channels=args.latent_encoder_channels,
        train_latent_perturb_std=args.train_latent_perturb_std,
        path_type=args.path_type,
        path_gamma=args.path_gamma,
        path_code_dim=args.path_code_dim,
        train_path_code_std=args.train_path_code_std,
        path_bend_channels=args.path_bend_channels,
        path_bend_scale=args.path_bend_scale,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_epoch", default=5000, type=int)
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--val_crop_size", default=None, type=int)
    parser.add_argument("--init_lr", default=2e-4, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--dir_path", default="./experiments/PairLatent_RF", type=str)
    parser.add_argument("--data_path", default="./dataset/GOPRO_Large", type=str)
    parser.add_argument("--model_name", default="PairLatent_RF", type=str)
    parser.add_argument("--model", default="ncsnpp", choices=["ncsnpp"], type=str)
    parser.add_argument(
        "--path_type", default="linear", choices=["linear", "quadratic"], type=str
    )
    parser.add_argument("--path_gamma", default=0.25, type=float)
    parser.add_argument("--latent_dim", default=64, type=int)
    parser.add_argument("--latent_map_channels", default=8, type=int)
    parser.add_argument("--latent_encoder_channels", default=32, type=int)
    parser.add_argument("--train_latent_perturb_std", default=0.0, type=float)
    parser.add_argument("--path_code_dim", default=0, type=int)
    parser.add_argument("--train_path_code_std", default=1.0, type=float)
    parser.add_argument("--path_bend_channels", default=32, type=int)
    parser.add_argument("--path_bend_scale", default=1.0, type=float)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--opt_beta1", default=0.9, type=float)
    parser.add_argument("--scheduler", default=None, type=str)
    parser.add_argument("--sample_timesteps", default=1000, type=int)
    parser.add_argument("--val_sample_timesteps", default=50, type=int)
    parser.add_argument("--valid_iters", default=10, type=int)
    parser.add_argument("--base_channels", default=128, type=int)
    parser.add_argument(
        "--channel_mults", default=(1, 1, 2, 2, 2, 2, 2), type=int, nargs="+"
    )
    parser.add_argument("--num_res_blocks", default=2, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--validation_epoch", default=50, type=int)
    parser.add_argument("--val_save_epochs", default=50, type=int)
    parser.add_argument("--check_point_epoch", default=200, type=int)
    parser.add_argument("--save_last_epoch", default=1, type=int)
    parser.add_argument("--criterion", default="l2", choices=["l1", "l2"], type=str)
    parser.add_argument("--rf_image_size", default=None, type=int)
    parser.add_argument("--attn_resolutions", default=(16,), type=int, nargs="+")
    parser.add_argument(
        "--rf_resblock_type", default="biggan", choices=["biggan", "ddpm"], type=str
    )
    parser.add_argument(
        "--rf_progressive",
        default="output_skip",
        choices=["none", "output_skip", "residual"],
        type=str,
    )
    parser.add_argument(
        "--rf_progressive_input",
        default="none",
        choices=["none", "input_skip", "residual"],
        type=str,
    )
    parser.add_argument("--rf_noise_scale", default=1.0, type=float)
    parser.add_argument("--rf_t_eps", default=1e-3, type=float)
    parser.add_argument("--rf_t_scale", default=999.0, type=float)
    parser.add_argument(
        "--rf_sampler", default="euler", choices=["euler", "heun", "rk45"], type=str
    )
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--pad_multiple", default=None, type=int)
    parser.add_argument(
        "--pad_mode",
        default="reflect",
        choices=["reflect", "replicate"],
        type=str,
    )
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--no_persistent_workers", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.pad_multiple is None:
        args.pad_multiple = args.crop_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("device :", device)
    print(args.__dict__.items())

    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": not args.no_pin_memory,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = not args.no_persistent_workers
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_set = Multi_GoPro_Loader(
        data_path=args.data_path,
        mode="train",
        crop_size=args.crop_size,
    )
    dataloader_train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **dataloader_kwargs,
    )

    val_set = Multi_GoPro_Loader(
        data_path=args.data_path,
        mode="test",
        crop_size=args.val_crop_size,
    )
    dataloader_val = DataLoader(
        val_set,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        **dataloader_kwargs,
    )

    if args.model != "ncsnpp":
        raise ValueError("model error")

    rectified_flow_model = build_model(args, device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(
            rectified_flow_model.parameters(),
            lr=args.init_lr,
            betas=(args.opt_beta1, 0.999),
            eps=args.opt_eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            rectified_flow_model.parameters(),
            lr=args.init_lr,
            betas=(args.opt_beta1, 0.999),
            eps=args.opt_eps,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"optimizer not supported {args.optimizer}")

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.end_epoch, eta_min=args.min_lr
        )
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError(f"scheduler not supported {args.scheduler}")

    last_path = os.path.join(args.dir_path, "last_{}.pth".format(args.model_name))
    if os.path.exists(last_path):
        print("load_last_pretrained")
        training_state = torch.load(last_path, weights_only=False)
        args.start_epoch = training_state["epoch"] + 1
        args.resume_global_step = training_state.get("global_step", 0)
        saved_args = training_state.get("args")
        if hasattr(saved_args, "best_psnr"):
            args.best_psnr = saved_args.best_psnr
        elif isinstance(saved_args, dict) and "best_psnr" in saved_args:
            args.best_psnr = saved_args["best_psnr"]
        training_state["model_state"] = judge_and_remove_module_dict(
            training_state["model_state"]
        )
        new_weight = rectified_flow_model.state_dict()
        new_weight.update(training_state["model_state"])
        rectified_flow_model.load_state_dict(new_weight)
        new_optimizer = optimizer.state_dict()
        new_optimizer.update(training_state["optimizer_state"])
        optimizer.load_state_dict(new_optimizer)
        if scheduler:
            saved_scheduler_state = training_state.get("scheduler_state")
            if saved_scheduler_state:
                new_scheduler = scheduler.state_dict()
                new_scheduler.update(saved_scheduler_state)
                scheduler.load_state_dict(new_scheduler)
    elif args.resume:
        print("load_resume_pretrained")
        model_load = torch.load(args.resume, weights_only=False)
        if "model_state" in model_load.keys():
            rectified_flow_model.load_state_dict(model_load["model_state"])
        else:
            rectified_flow_model.load_state_dict(model_load)
        os.makedirs(args.dir_path, exist_ok=True)
    else:
        os.makedirs(args.dir_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.dir_path, "train.log"),
        format="%(levelname)s:%(message)s",
        encoding="utf-8",
        level=logging.INFO,
    )

    logging.info(f"args: {args}")
    logging.info(f"model: {rectified_flow_model}")
    logging.info(f"model parameters: {count_parameters(rectified_flow_model)}")
    logging.info(f"Optimizer:{optimizer.__class__.__name__}")
    logging.info(f"Scheduler:{scheduler.__class__.__name__ if scheduler else None}")

    writer = SummaryWriter(os.path.join("log", args.model_name))
    writer.add_text("args", str(args))

    trainer = PairLatentTrainer(
        dataloader_train,
        dataloader_val,
        rectified_flow_model,
        optimizer,
        scheduler,
        args,
        writer,
    )
    trainer.global_step = getattr(args, "resume_global_step", 0)
    trainer.train()
