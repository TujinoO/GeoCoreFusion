"""Optional scene-specific self-supervised coefficient refiner."""

from __future__ import annotations

import numpy as np

from .config import FusionConfig
from .degradation import PsfModel, degrade_coefficients, upsample_coefficients
from .fusion import CoefficientFusionResult


def refine_coefficients_neural(low_coeff: np.ndarray, rgb: np.ndarray, psf: PsfModel, config: FusionConfig) -> CoefficientFusionResult:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError("Neural refiner requires the optional torch dependency") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial = upsample_coefficients(low_coeff, rgb.shape[:2])
    rgb01 = np.asarray(rgb, dtype=np.float32)
    if rgb01.max() > 2.0:
        rgb01 = rgb01 / 255.0
    rank = low_coeff.shape[2]

    class ResidualNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(config.neural_hidden_channels)
            self.net = nn.Sequential(
                nn.Conv2d(rank + 3, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, rank, 3, padding=1),
            )

        def forward(self, coeff: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
            return coeff + 0.1 * self.net(torch.cat([coeff, guide], dim=1))

    model = ResidualNet().to(device)
    coeff0 = torch.from_numpy(initial.transpose(2, 0, 1)[None]).to(device)
    guide = torch.from_numpy(rgb01[:, :, :3].transpose(2, 0, 1)[None]).to(device)
    low_target = torch.from_numpy(low_coeff.transpose(2, 0, 1)[None]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.neural_learning_rate))
    history: list[dict[str, float]] = []
    for step in range(max(1, int(config.neural_steps))):
        optimizer.zero_grad(set_to_none=True)
        pred = model(coeff0, guide)
        low_pred = F.interpolate(pred, size=low_coeff.shape[:2], mode="area")
        dx_c = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy_c = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx_g = guide[:, :, :, 1:] - guide[:, :, :, :-1]
        dy_g = guide[:, :, 1:, :] - guide[:, :, :-1, :]
        wx = torch.exp(-12.0 * torch.mean(dx_g**2, dim=1, keepdim=True))
        wy = torch.exp(-12.0 * torch.mean(dy_g**2, dim=1, keepdim=True))
        observation = F.mse_loss(low_pred, low_target)
        smoothness = torch.mean(wx * torch.abs(dx_c)) + torch.mean(wy * torch.abs(dy_c))
        anchor = F.mse_loss(F.avg_pool2d(pred, 5, stride=1, padding=2), F.avg_pool2d(coeff0, 5, stride=1, padding=2))
        residual_penalty = F.mse_loss(pred, coeff0)
        loss = observation + 0.015 * smoothness + 0.05 * anchor + 0.005 * residual_penalty
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0 or step == 0:
            history.append({"iteration": float(step + 1), "loss": float(loss.detach().cpu()), "observation": float(observation.detach().cpu())})
    with torch.no_grad():
        refined = model(coeff0, guide).cpu().numpy()[0].transpose(1, 2, 0).astype(np.float32)
    residual = low_coeff - degrade_coefficients(refined, psf)
    uncertainty = np.sqrt(np.mean(upsample_coefficients(residual**2, rgb.shape[:2]), axis=2))
    scale = float(np.percentile(uncertainty, 95)) if uncertainty.size else 1.0
    uncertainty = np.clip(uncertainty / max(scale, 1e-6), 0.0, 1.0)
    return CoefficientFusionResult(
        coefficients=refined,
        uncertainty_map=uncertainty.astype(np.float32),
        detail_gain_map=np.ones(rgb.shape[:2], dtype=np.float32),
        additive_detail_map=np.zeros(rgb.shape[:2], dtype=np.float32),
        history=history,
        details={"method": "scene_specific_self_supervised_residual_cnn", "device": str(device), "steps": int(config.neural_steps)},
    )
