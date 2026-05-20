import argparse
import logging
import os
import random
import sys
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import cv2
import pyiqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import Flow_Loader
from utils.utils import (
    AverageMeter,
    count_parameters,
    judge_and_remove_module_dict,
    tensor2cv,
)

RF_IMAGEGEN_DIR = Path(__file__).resolve().parent / "models" / "RectifiedFlow" / "ImageGeneration"
if str(RF_IMAGEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(RF_IMAGEGEN_DIR))

from models.ncsnpp import NCSNpp  # noqa: E402

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps**2))


class ConditionalRectifiedFlow(nn.Module):
    """
    Conditional rectified-flow wrapper for ID-Blau using the vendored
    RectifiedFlow/ImageGeneration model family.

    The training objective and Gaussian start follow the implementation under
    models/RectifiedFlow/ImageGeneration:
      x_t = t * x_1 + (1 - t) * x_0
      target = x_1 - x_0
    """

    def __init__(
        self,
        model,
        img_channels=3,
        criterion="l2",
        device="cuda",
        sample_steps=50,
        use_dataparallel=True,
        init_type="gaussian",
        noise_scale=1.0,
        t_eps=1e-3,
        t_scale=999.0,
    ):
        super().__init__()
        if use_dataparallel:
            self.model = nn.DataParallel(model).to(device)
        else:
            self.model = model.to(device)
        self.img_channels = img_channels
        self.sample_steps = sample_steps
        self.init_type = init_type
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.t_scale = t_scale
        model_config = getattr(model, "config", None)
        ch_mult = getattr(getattr(model_config, "model", None), "ch_mult", None)
        self.size_divisor = 2 ** (len(ch_mult) - 1) if ch_mult else 1

        if criterion == "l1":
            self.criterion = CharbonnierLoss()
        elif criterion == "l2":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("loss criterion must be l1 or l2")

    def get_x0(self, x1):
        if self.init_type == "gaussian":
            return torch.randn_like(x1) * self.noise_scale
        raise ValueError("init_type must be 'gaussian' for the vendored RF model")

    @staticmethod
    def expand_time(t, x):
        return t.view(-1, *([1] * (x.ndim - 1)))

    def compute_loss(self, x1, condition):
        source = self.get_x0(x1)
        t = (
            torch.rand(x1.shape[0], device=x1.device) * (1.0 - self.t_eps)
            + self.t_eps
        )
        t_img = self.expand_time(t, x1)
        xt = t_img * x1 + (1.0 - t_img) * source
        target_velocity = x1 - source
        model_input = torch.cat([xt, condition], dim=1)
        pred_velocity = self.model(model_input, t * self.t_scale)
        return self.criterion(pred_velocity, target_velocity)

    def forward(self, x1, condition):
        return self.compute_loss(x1=x1, condition=condition)

    @torch.no_grad()
    def sample(
        self,
        condition,
        device="cuda",
        sample_steps=None,
        sample_timesteps=None,
        tqdm_visible=False,
        method="euler",
    ):
        if sample_steps is None and sample_timesteps is not None:
            sample_steps = sample_timesteps
        if sample_steps is None:
            sample_steps = self.sample_steps

        b, _, h, w = condition.shape
        pad_h = (self.size_divisor - h % self.size_divisor) % self.size_divisor
        pad_w = (self.size_divisor - w % self.size_divisor) % self.size_divisor
        if pad_h or pad_w:
            condition = F.pad(condition, (0, pad_w, 0, pad_h), mode="replicate")

        x = torch.randn((b, self.img_channels, h, w), device=device) * self.noise_scale
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        timesteps = torch.linspace(self.t_eps, 1.0, sample_steps + 1, device=device)
        iterator = (
            tqdm.tqdm(range(sample_steps), desc="rectified-flow sampling")
            if tqdm_visible
            else range(sample_steps)
        )
        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_batch = torch.full((b,), t, device=device)
            model_input = torch.cat([x, condition], dim=1)
            velocity = self.model(model_input, t_batch * self.t_scale)
            if method == "euler":
                x = x + dt * velocity
            elif method == "heun":
                x_pred = x + dt * velocity
                t_next_batch = torch.full((b,), t_next, device=device)
                model_input_next = torch.cat([x_pred, condition], dim=1)
                velocity_next = self.model(
                    model_input_next, t_next_batch * self.t_scale
                )
                x = x + 0.5 * dt * (velocity + velocity_next)
            else:
                raise ValueError("method must be 'euler' or 'heun'")

        return x[..., :h, :w].detach()


class Trainer:
    def __init__(
        self,
        dataloader_train,
        dataloader_val,
        model,
        optimizer,
        scheduler,
        args,
        writer,
    ) -> None:
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.writer = writer
        self.epoch = 0
        self.global_step = 0
        self.sample_timesteps = self.args.sample_timesteps
        self.val_sample_timesteps = self.args.val_sample_timesteps
        self.device = self.args.device
        self.psnr_func = pyiqa.create_metric("psnr", device=self.device)
        self.lpips_func = pyiqa.create_metric("lpips", device=self.device)
        self.best_psnr = self.args.best_psnr if hasattr(self.args, "best_psnr") else 0
        self.grad_clip = self.args.grad_clip

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
        total_train_psnr = AverageMeter()
        total_train_lpips = AverageMeter()

        for idx, sample in enumerate(tq):
            self.model.train()
            self.optimizer.zero_grad()

            blur, sharp = (
                sample["blur"].to(self.device),
                sample["sharp"].to(self.device),
            )
            flow = sample["flow"].to(self.device)
            condition = torch.cat([sharp, flow], dim=1)
            loss = self.model(x1=blur, condition=condition)
            loss.backward()

            self._apply_warmup_lr()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.global_step += 1
            total_train_loss.update(loss.detach().item())

            # if idx % 10 == 0:
            #     psnr, lpips = self._valid(blur, gt)
            #     total_train_psnr.update(psnr)
            #     total_train_lpips.update(lpips)

            tq.set_postfix(
                {
                    "loss": total_train_loss.avg,
                    "psnr": total_train_psnr.avg,
                    "lpips": total_train_lpips.avg,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

        if self.scheduler:
            self.scheduler.step()
        self.writer.add_scalar("Loss/Train_loss", total_train_loss.avg, self.epoch)
        self.writer.add_scalar("Loss/Train_psnr", total_train_psnr.avg, self.epoch)
        self.writer.add_scalar("Loss/Train_lpips", total_train_lpips.avg, self.epoch)
        logging.info(
            f"Epoch [{self.epoch}/{self.args.end_epoch}]: Train_loss: {total_train_loss.avg:.4f} Train_psnr:{total_train_psnr.avg:.4f} Train_lpips:{total_train_lpips.avg:.4f}"
        )

    @torch.no_grad()
    def _valid(self, sharp, blur, flow):
        self.model.eval()
        condition = torch.cat([sharp, flow], dim=1)
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
        for idx, sample in enumerate(tq):
            blur, sharp = (
                sample["blur"].to(self.device),
                sample["sharp"].to(self.device),
            )
            flow = sample["flow"].to(self.device)
            psnr, lpips = self._valid(sharp, blur, flow)
            total_val_psnr.update(psnr)
            total_val_lpips.update(lpips)
            tq.set_postfix(LPIPS=total_val_lpips.avg, PSNR=total_val_psnr.avg)

        self.writer.add_scalar("Val/Test_lpips", total_val_lpips.avg, self.epoch)
        self.writer.add_scalar("Val/Test_psnr", total_val_psnr.avg, self.epoch)
        logging.info(
            f"Validation Epoch [{self.epoch}/{self.args.end_epoch}]: Test lpips: {total_val_lpips.avg:.4f} Test psnr:{total_val_psnr.avg:.4f}"
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
        """save model parameters"""
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
        """use train set to val and save image"""
        os.makedirs(dir_path, exist_ok=True)
        self.model.eval()
        for idx in random.sample(range(0, len(dataset)), val_num):
            sample = dataset[idx]
            blur, sharp = (
                sample["blur"].unsqueeze(0).to(self.device),
                sample["sharp"].unsqueeze(0).to(self.device),
            )
            flow = sample["flow"].unsqueeze(0).to(self.device)
            condition = torch.cat([sharp, flow], dim=1)
            output = self.model.sample(
                condition=condition,
                sample_timesteps=self.val_sample_timesteps,
                device=self.device,
                method=self.args.rf_sampler,
            )
            output = output.clamp(-0.5, 0.5)

            save_img_dir_path = os.path.join(dir_path, "visualization", "output")
            os.makedirs(save_img_dir_path, exist_ok=True)
            save_sharp_dir_path = os.path.join(dir_path, "visualization", "sharp")
            os.makedirs(save_sharp_dir_path, exist_ok=True)
            save_blur_dir_path = os.path.join(dir_path, "visualization", "blur")
            os.makedirs(save_blur_dir_path, exist_ok=True)

            save_img_path = os.path.join(
                save_img_dir_path, f"{self.epoch:05d}_{idx:05d}.png"
            )
            output = tensor2cv(output + 0.5)
            cv2.imwrite(save_img_path, output)

            save_sharp_path = os.path.join(
                save_sharp_dir_path, f"{self.epoch:05d}_{idx:05d}.png"
            )
            sharp = tensor2cv(sharp + 0.5)
            cv2.imwrite(save_sharp_path, sharp)

            save_blur_path = os.path.join(
                save_blur_dir_path, f"{self.epoch:05d}_{idx:05d}.png"
            )
            blur = tensor2cv(blur + 0.5)
            cv2.imwrite(save_blur_path, blur)


def generate_linear_schedule(T, beta_1, beta_T):
    return torch.linspace(beta_1, beta_T, T).double()


def build_rf_config(args):
    image_size = args.rf_image_size if args.rf_image_size is not None else args.crop_size
    return SimpleNamespace(
        device=args.device,
        training=SimpleNamespace(
            sde="rectified_flow",
            continuous=False,
            reduce_mean=True,
        ),
        sampling=SimpleNamespace(
            method="rectified_flow",
            init_type="gaussian",
            init_noise_scale=args.rf_noise_scale,
            use_ode_sampler=args.rf_sampler,
            sample_N=args.sample_timesteps,
        ),
        data=SimpleNamespace(
            centered=True,
            image_size=image_size,
            input_channels=9,
            output_channels=3,
            num_channels=3,
        ),
        model=SimpleNamespace(
            name="ncsnpp",
            sigma_max=378,
            sigma_min=0.01,
            num_scales=2000,
            beta_min=0.1,
            beta_max=20.0,
            scale_by_sigma=False,
            ema_rate=0.999,
            normalization="GroupNorm",
            nonlinearity="swish",
            nf=args.base_channels,
            ch_mult=tuple(args.channel_mults),
            num_res_blocks=args.num_res_blocks,
            dropout=args.dropout,
            attn_resolutions=tuple(args.attn_resolutions),
            resamp_with_conv=True,
            conditional=True,
            fir=True,
            fir_kernel=[1, 3, 3, 1],
            skip_rescale=True,
            resblock_type=args.rf_resblock_type,
            progressive=args.rf_progressive,
            progressive_input=args.rf_progressive_input,
            progressive_combine="sum",
            attention_type="ddpm",
            init_scale=0.0,
            embedding_type="fourier",
            fourier_scale=16,
            conv_size=3,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_epoch", default=5000, type=int)
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--val_crop_size", default=None, type=int)
    parser.add_argument("--init_lr", default=2e-4, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--beta_1", default=1e-6, type=float)
    parser.add_argument("--beta_T", default=1e-2, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--num_timesteps", default=2000, type=int)
    parser.add_argument("--dir_path", default="./experiments/ID_Blau", type=str)
    parser.add_argument("--data_path", default="./dataset/GOPRO_Large", type=str)
    parser.add_argument("--flow_data_path", default="./dataset/GOPRO_flow", type=str)
    parser.add_argument("--flow_norm", default=True, type=bool)
    parser.add_argument("--model_name", default="ID_Blau_FM", type=str)
    parser.add_argument("--model", default="ncsnpp", choices=["ncsnpp"], type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--opt_beta1", default=0.9, type=float)
    parser.add_argument("--scheduler", default=None, type=str)
    parser.add_argument("--sample_timesteps", default=1000, type=int)
    parser.add_argument("--val_sample_timesteps", default=50, type=int)
    parser.add_argument("--valid_iters", default=10, type=int)
    parser.add_argument("--base_channels", default=128, type=int)
    parser.add_argument("--time_dim", default=256, type=int)
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
        "--rf_sampler", default="euler", choices=["euler", "heun"], type=str
    )
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--no_persistent_workers", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("device :", device)
    # same_seed(args.seed)
    print(args.__dict__.items())

    # Traning loader
    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": not args.no_pin_memory,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = not args.no_persistent_workers
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    Train_set = Flow_Loader(
        data_path=args.data_path,
        flow_path=args.flow_data_path,
        mode="train",
        crop_size=args.crop_size,
        flow_norm=args.flow_norm,
    )
    dataloader_train = DataLoader(
        Train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **dataloader_kwargs,
    )
    # Valing loader
    Val_set = Flow_Loader(
        data_path=args.data_path,
        flow_path=args.flow_data_path,
        mode="test",
        crop_size=args.val_crop_size,
        flow_norm=args.flow_norm,
    )
    dataloader_val = DataLoader(
        Val_set,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        **dataloader_kwargs,
    )

    if args.model != "ncsnpp":
        raise ValueError("model error")

    rf_config = build_rf_config(args)
    net = NCSNpp(rf_config).to(device)

    rectified_flow_model = ConditionalRectifiedFlow(
        net,
        img_channels=3,
        criterion=args.criterion,
        device=device,
        sample_steps=args.sample_timesteps,
        init_type="gaussian",
        noise_scale=args.rf_noise_scale,
        t_eps=args.rf_t_eps,
        t_scale=args.rf_t_scale,
    )

    if args.optimizer == "adam":
        optimizer = optim.Adam(
            net.parameters(),
            lr=args.init_lr,
            betas=(args.opt_beta1, 0.999),
            eps=args.opt_eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            net.parameters(),
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
    # load pretrained
    if os.path.exists(
        os.path.join(args.dir_path, "last_{}.pth".format(args.model_name))
    ):
        print("load_last_pretrained")
        training_state = torch.load(
            os.path.join(args.dir_path, "last_{}.pth".format(args.model_name)),
            weights_only=False,
        )
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
            new_scheduler = scheduler.state_dict()
            new_scheduler.update(training_state["scheduler_state"])
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

    trainer = Trainer(
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
