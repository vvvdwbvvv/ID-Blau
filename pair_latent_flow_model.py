import torch
import torch.nn as nn

from rectified_flow_train import ConditionalRectifiedFlow, build_rf_config


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.SiLU(inplace=True),
    )


class DegradationLatentEncoder(nn.Module):
    """q_phi(S, B): compact implicit degradation descriptor for one pair."""

    def __init__(self, latent_dim=64, base_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(6, base_channels, stride=2),
            conv_block(base_channels, base_channels * 2, stride=2),
            conv_block(base_channels * 2, base_channels * 4, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        feature_dim = base_channels * 4
        self.head = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, sharp, blur):
        return self.head(self.encoder(torch.cat([sharp, blur], dim=1)))


class PathBendingNet(nn.Module):
    """h_psi(S, e, z): endpoint-preserving quadratic path bending term."""

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        out_channels=3,
        max_magnitude=1.0,
    ):
        super().__init__()
        self.max_magnitude = max_magnitude
        self.net = nn.Sequential(
            conv_block(in_channels, hidden_channels),
            conv_block(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, condition):
        return self.net(condition) * self.max_magnitude


def pair_latent_condition_channels(args):
    return 3 + args.latent_map_channels + args.path_code_dim


def build_pair_latent_rf_config(args):
    config = build_rf_config(args)
    config.data.input_channels = 3 + pair_latent_condition_channels(args)
    return config


class PairLatentRectifiedFlow(ConditionalRectifiedFlow):
    """
    Pair-conditioned latent flow matching.

    The original ID-Blau rectified-flow implementation remains untouched. This
    wrapper changes only the conditioning and optional probability path:

      e = q_phi(S, B)
      x_t = (1 - t) x_0 + t B                         linear
      x_t = (1 - t) x_0 + t B + gamma t(1 - t) h(S,e)  quadratic
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
        pad_multiple=None,
        pad_mode="reflect",
        latent_dim=64,
        latent_map_channels=8,
        latent_encoder_channels=32,
        train_latent_perturb_std=0.0,
        path_type="linear",
        path_gamma=0.25,
        path_code_dim=0,
        train_path_code_std=1.0,
        path_bend_channels=32,
        path_bend_scale=1.0,
    ):
        super().__init__(
            model=model,
            img_channels=img_channels,
            criterion=criterion,
            device=device,
            sample_steps=sample_steps,
            use_dataparallel=use_dataparallel,
            init_type=init_type,
            noise_scale=noise_scale,
            t_eps=t_eps,
            t_scale=t_scale,
            pad_multiple=pad_multiple,
            pad_mode=pad_mode,
        )
        self.latent_dim = latent_dim
        self.latent_map_channels = latent_map_channels
        self.train_latent_perturb_std = train_latent_perturb_std
        self.path_type = path_type
        self.path_gamma = path_gamma
        self.path_code_dim = path_code_dim
        self.train_path_code_std = train_path_code_std

        self.degradation_encoder = DegradationLatentEncoder(
            latent_dim=latent_dim,
            base_channels=latent_encoder_channels,
        ).to(device)
        self.latent_to_map = nn.Sequential(
            nn.Linear(latent_dim, latent_map_channels),
            nn.SiLU(inplace=True),
            nn.Linear(latent_map_channels, latent_map_channels),
        ).to(device)

        if path_type == "quadratic":
            self.path_bender = PathBendingNet(
                in_channels=self.condition_channels,
                hidden_channels=path_bend_channels,
                out_channels=img_channels,
                max_magnitude=path_bend_scale,
            ).to(device)
        elif path_type == "linear":
            self.path_bender = None
        else:
            raise ValueError("path_type must be 'linear' or 'quadratic'")

    @property
    def condition_channels(self):
        return 3 + self.latent_map_channels + self.path_code_dim

    def sample_path_code(self, batch_size, device, std=1.0):
        if self.path_code_dim <= 0:
            return None
        if std <= 0:
            return torch.zeros(batch_size, self.path_code_dim, device=device)
        return torch.randn(batch_size, self.path_code_dim, device=device) * std

    def encode_latent(self, sharp, blur):
        return self.degradation_encoder(sharp, blur)

    def build_condition(
        self,
        sharp,
        blur=None,
        latent=None,
        latent_noise_std=0.0,
        path_code=None,
    ):
        if latent is None:
            if blur is None:
                raise ValueError("build_condition requires blur or latent")
            latent = self.encode_latent(sharp, blur)
        if latent_noise_std > 0:
            latent = latent + torch.randn_like(latent) * latent_noise_std

        b, _, h, w = sharp.shape
        latent_map = self.latent_to_map(latent).view(b, self.latent_map_channels, 1, 1)
        condition_parts = [sharp, latent_map.expand(-1, -1, h, w)]

        if self.path_code_dim > 0:
            if path_code is None:
                path_code = self.sample_path_code(b, sharp.device, std=0.0)
            path_map = path_code.view(b, self.path_code_dim, 1, 1)
            condition_parts.append(path_map.expand(-1, -1, h, w))

        return torch.cat(condition_parts, dim=1), latent

    def compute_loss(self, x1, sharp, blur):
        path_code = self.sample_path_code(
            x1.shape[0],
            x1.device,
            std=self.train_path_code_std if self.path_type == "quadratic" else 0.0,
        )
        condition, _ = self.build_condition(
            sharp=sharp,
            blur=blur,
            latent_noise_std=self.train_latent_perturb_std,
            path_code=path_code,
        )

        source = self.get_x0(x1)
        t = torch.rand(x1.shape[0], device=x1.device) * (1.0 - self.t_eps) + self.t_eps
        t_img = self.expand_time(t, x1)
        xt = t_img * x1 + (1.0 - t_img) * source
        target_velocity = x1 - source

        if self.path_type == "quadratic":
            bend = self.path_bender(condition)
            xt = xt + self.path_gamma * t_img * (1.0 - t_img) * bend
            target_velocity = (
                target_velocity + self.path_gamma * (1.0 - 2.0 * t_img) * bend
            )

        model_input = torch.cat([xt, condition], dim=1)
        pred_velocity = self.model(model_input, t * self.t_scale)
        return self.criterion(pred_velocity, target_velocity)

    def forward(self, x1, sharp, blur):
        return self.compute_loss(x1=x1, sharp=sharp, blur=blur)
