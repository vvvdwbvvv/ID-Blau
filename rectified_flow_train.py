import argparse
import logging
import os
import random
from itertools import islice

import cv2
import pyiqa
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import Flow_Loader
from models.diffusion_model import UNet
from models.losses import CharbonnierLoss
from utils.utils import (
    AverageMeter,
    count_parameters,
    judge_and_remove_module_dict,
    tensor2cv,
)

cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True


class ConditionalRectifiedFlow(nn.Module):
    """
    Conditional rectified-flow wrapper for ID-Blau.

    It keeps the existing ID-Blau call surface:
      loss = model(x1=blur, condition=torch.cat([sharp, flow], dim=1), x0=sharp)
      output = model.sample(condition=condition, x0=sharp, ...)

    The training objective follows the Rectified Flow implementation under
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
        init_type="sharp",
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

        if criterion == "l1":
            self.criterion = CharbonnierLoss()
        elif criterion == "l2":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("loss criterion must be l1 or l2")

    def get_x0(self, x1, x0=None):
        if self.init_type == "sharp":
            if x0 is None:
                raise ValueError("init_type='sharp' requires x0=sharp")
            return x0
        if self.init_type == "gaussian":
            return torch.randn_like(x1) * self.noise_scale
        raise ValueError("init_type must be 'sharp' or 'gaussian'")

    @staticmethod
    def expand_time(t, x):
        return t.view(-1, *([1] * (x.ndim - 1)))

    def compute_loss(self, x1, condition, x0=None):
        source = self.get_x0(x1, x0=x0)
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

    def forward(self, x1, condition, x0=None):
        return self.compute_loss(x1=x1, condition=condition, x0=x0)

    @torch.no_grad()
    def sample(
        self,
        condition,
        x0=None,
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
        if x0 is None:
            x = (
                torch.randn((b, self.img_channels, h, w), device=device)
                * self.noise_scale
            )
        else:
            x = x0.to(device)

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

        return x.detach()


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
        self.sample_timesteps = self.args.sample_timesteps
        self.device = self.args.device
        self.psnr_func = pyiqa.create_metric("psnr", device=self.device)
        self.lpips_func = pyiqa.create_metric("lpips", device=self.device)
        self.best_psnr = self.args.best_psnr if hasattr(self.args, "best_psnr") else 0
        self.grad_clip = 1

    def train(self):
        print("Start_Epoch:", self.args.start_epoch)
        print("End_Epoch:", self.args.end_epoch)
        print("Model:", self.args.model_name)
        print(f"Optimizer:{self.optimizer.__class__.__name__}")
        print(
            f"Scheduler:{self.scheduler.__class__.__name__ if self.scheduler else None}"
        )
        print("start train")

        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):
            self.epoch = epoch
            self._train_epoch()

            if (
                epoch % self.args.validation_epoch
            ) == 0 or epoch == self.args.end_epoch:
                self.valid()

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
            loss = self.model(x1=blur, condition=condition, x0=sharp)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
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
        sample_x0 = sharp if self.args.rf_init_type == "sharp" else None
        output = self.model.sample(
            condition=condition,
            x0=sample_x0,
            sample_timesteps=self.sample_timesteps,
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
            sample_x0 = sharp if self.args.rf_init_type == "sharp" else None
            output = self.model.sample(
                condition=condition,
                x0=sample_x0,
                sample_timesteps=self.sample_timesteps,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_epoch", default=5000, type=int)
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--crop_size", default=128, type=int)
    parser.add_argument("--init_lr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--beta_1", default=1e-6, type=float)
    parser.add_argument("--beta_T", default=1e-2, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--num_timesteps", default=2000, type=int)
    parser.add_argument("--dir_path", default="./experiments/ID_Blau", type=str)
    parser.add_argument("--data_path", default="./dataset/GOPRO_Large", type=str)
    parser.add_argument("--flow_data_path", default="./dataset/GOPRO_flow", type=str)
    parser.add_argument("--flow_norm", default=True, type=bool)
    parser.add_argument("--model_name", default="ID_Blau_FM", type=str)
    parser.add_argument("--model", default="UNet", choices=["UNet"], type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--opt_beta1", default=0.9, type=float)
    parser.add_argument("--scheduler", default=None, type=str)
    parser.add_argument("--sample_timesteps", default=20, type=int)
    parser.add_argument("--base_channels", default=64, type=int)
    parser.add_argument("--time_dim", default=256, type=int)
    parser.add_argument("--channel_mults", default=(1, 2, 3), type=int, nargs="+")
    parser.add_argument("--num_res_blocks", default=2, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--validation_epoch", default=50, type=int)
    parser.add_argument("--val_save_epochs", default=50, type=int)
    parser.add_argument("--check_point_epoch", default=200, type=int)
    parser.add_argument("--save_last_epoch", default=1, type=int)
    parser.add_argument("--criterion", default="l2", choices=["l1", "l2"], type=str)
    parser.add_argument(
        "--rf_init_type",
        default="sharp",
        choices=["sharp", "gaussian"],
        type=str,
        help="RF source distribution. 'sharp' preserves ID-Blau sharp-to-blur training; 'gaussian' matches standard RF.",
    )
    parser.add_argument("--rf_noise_scale", default=1.0, type=float)
    parser.add_argument("--rf_t_eps", default=1e-3, type=float)
    parser.add_argument("--rf_t_scale", default=999.0, type=float)
    parser.add_argument(
        "--rf_sampler", default="euler", choices=["euler", "heun"], type=str
    )
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
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
        crop_size=None,
        flow_norm=args.flow_norm,
    )
    dataloader_val = DataLoader(
        Val_set,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        **dataloader_kwargs,
    )

    if args.model == "UNet":
        net = UNet(
            img_channels=9,
            base_channels=args.base_channels,
            channel_mults=args.channel_mults,
            time_dim=args.time_dim,
            num_res_blocks=args.num_res_blocks,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError("model error")

    diffusionModel = ConditionalRectifiedFlow(
        net,
        img_channels=3,
        criterion=args.criterion,
        device=device,
        sample_steps=args.sample_timesteps,
        init_type=args.rf_init_type,
        noise_scale=args.rf_noise_scale,
        t_eps=args.rf_t_eps,
        t_scale=args.rf_t_scale,
    )

    if args.optimizer == "adam":
        optimizer = optim.Adam(
            net.parameters(), lr=args.init_lr, betas=(args.opt_beta1, 0.999)
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=args.init_lr, weight_decay=1e-4)
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
            os.path.join(args.dir_path, "last_{}.pth".format(args.model_name))
        )
        args.start_epoch = training_state["epoch"] + 1
        saved_args = training_state.get("args")
        if hasattr(saved_args, "best_psnr"):
            args.best_psnr = saved_args.best_psnr
        elif isinstance(saved_args, dict) and "best_psnr" in saved_args:
            args.best_psnr = saved_args["best_psnr"]
        training_state["model_state"] = judge_and_remove_module_dict(
            training_state["model_state"]
        )
        new_weight = diffusionModel.state_dict()
        new_weight.update(training_state["model_state"])
        diffusionModel.load_state_dict(new_weight)
        new_optimizer = optimizer.state_dict()
        new_optimizer.update(training_state["optimizer_state"])
        optimizer.load_state_dict(new_optimizer)
        if scheduler:
            new_scheduler = scheduler.state_dict()
            new_scheduler.update(training_state["scheduler_state"])
            scheduler.load_state_dict(new_scheduler)
    elif args.resume:
        print("load_resume_pretrained")
        model_load = torch.load(args.resume)
        if "model_state" in model_load.keys():
            diffusionModel.load_state_dict(model_load["model_state"])
        else:
            diffusionModel.load_state_dict(model_load)
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
    logging.info(f"model: {diffusionModel}")
    logging.info(f"model parameters: {count_parameters(diffusionModel)}")
    logging.info(f"Optimizer:{optimizer.__class__.__name__}")
    logging.info(f"Scheduler:{scheduler.__class__.__name__ if scheduler else None}")

    writer = SummaryWriter(os.path.join("log", args.model_name))
    writer.add_text("args", str(args))

    trainer = Trainer(
        dataloader_train,
        dataloader_val,
        diffusionModel,
        optimizer,
        scheduler,
        args,
        writer,
    )
    trainer.train()
