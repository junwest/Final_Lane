"""Lane overlay utilities used by make_full_video.

This module bundles the segmentation network, geometry helpers and convenience
APIs (`LaneDepartureAnalyzer`, `load_roi_mask`, `generate_overlay_video`) that
are consumed by the CLI script.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
#  Segmentation model
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """A lightweight UNet for binary segmentation."""

    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        def double_conv(in_channels: int, out_channels: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.inc = double_conv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), double_conv(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(nn.MaxPool2d(2), double_conv(512, 1024 // factor))
        self.up1 = nn.ConvTranspose2d(1024, 512 // factor, kernel_size=2, stride=2)
        self.conv1 = double_conv(1024, 512 // factor)
        self.up2 = nn.ConvTranspose2d(512, 256 // factor, kernel_size=2, stride=2)
        self.conv2 = double_conv(512, 256 // factor)
        self.up3 = nn.ConvTranspose2d(256, 128 // factor, kernel_size=2, stride=2)
        self.conv3 = double_conv(256, 128 // factor)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = double_conv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    @staticmethod
    def _pad_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        diff_y = ref.size(2) - x.size(2)
        diff_x = ref.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self._pad_to_match(x, x4)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = self._pad_to_match(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = self._pad_to_match(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = self._pad_to_match(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        logits = self.outc(x)
        return logits


class LaneSegmenter:
    """Wrapper around the UNet segmentation network."""

    def __init__(self, model_path: Path, device: str = "auto", threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "mps":
            self.device = torch.device("mps")
        else:
            self.device = torch.device(device)
        self.model: nn.Module = UNet(n_channels=3, n_classes=1, bilinear=False)
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and any(k.startswith("model") for k in state):
            for key in ("model_state_dict", "state_dict", "model_state"):
                if key in state:
                    state = state[key]
                    break
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return (prob > self.threshold).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
#  Lane geometry utilities
# ---------------------------------------------------------------------------


class LaneGeometry:
    """Compute simple metrics from a binary mask using a BEV perspective."""

    def __init__(
        self,
        bev_width: int = 700,
        bev_height: int = 1000,
        src_points: Tuple[Tuple[float, float], ...] = (
            (294.0, 264.0),
            (99.0, 349.0),
            (803.0, 355.0),
            (673.0, 268.0),
        ),
        lane_width_m: float = 3.5,
        depart_ratio_threshold: float = 50.0,
    ) -> None:
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(
            [
                [0.0, 0.0],
                [0.0, bev_height - 1.0],
                [bev_width - 1.0, bev_height - 1.0],
                [bev_width - 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.matrix: np.ndarray = cv2.getPerspectiveTransform(self.src_points, dst_points)
        self.inv_matrix: np.ndarray = cv2.getPerspectiveTransform(dst_points, self.src_points)
        self.lane_width_m = lane_width_m
        self.depart_ratio_threshold = depart_ratio_threshold
        self.offset_history: Deque[float] = collections.deque(maxlen=30)

    def warp_to_bev(self, mask: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(mask, self.matrix, (self.bev_width, self.bev_height))

    def find_lane_positions(self, bev: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        row_sums = bev.sum(axis=1)
        row_idx = int(np.argmax(row_sums))
        y0 = max(0, row_idx - 10)
        y1 = min(self.bev_height - 1, row_idx + 10)
        strip = bev[y0 : y1 + 1, :]
        _ys, xs = np.where(strip > 0)
        if len(xs) == 0:
            return None, None
        centre = self.bev_width // 2
        left_candidates = xs[xs < centre]
        right_candidates = xs[xs > centre]
        left_x = int(np.max(left_candidates)) if len(left_candidates) else None
        right_x = int(np.min(right_candidates)) if len(right_candidates) else None
        if left_x is None or right_x is None or right_x <= left_x:
            return None, None
        return left_x, right_x

    def compute_metrics(
        self,
        bev: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Optional[int], Optional[int], Optional[float], float, float, float, float, bool, bool]:
        left_x, right_x = self.find_lane_positions(bev)
        if left_x is None or right_x is None:
            self.offset_history.append(0.0)
            return None, None, None, 0.0, 0.0, 0.0, 0.0, False, False
        lane_width_px = float(right_x - left_x)
        left_x_frame = int(left_x / self.bev_width * frame_width)
        right_x_frame = int(right_x / self.bev_width * frame_width)
        lane_centre_bev = (left_x + right_x) / 2.0
        car_centre_bev = self.bev_width / 2.0
        offset_px = car_centre_bev - lane_centre_bev
        px_per_m = lane_width_px / self.lane_width_m
        offset_m = offset_px / px_per_m
        lane_half_width_m = self.lane_width_m / 2.0
        left_ratio = 0.0
        right_ratio = 0.0
        if offset_m < 0:
            left_ratio = min(100.0, abs(offset_m) / lane_half_width_m * 100.0)
        elif offset_m > 0:
            right_ratio = min(100.0, abs(offset_m) / lane_half_width_m * 100.0)
        departed = max(left_ratio, right_ratio) >= self.depart_ratio_threshold
        self.offset_history.append(offset_m)
        lane_change = False
        if len(self.offset_history) >= 30:
            past_mean = float(np.mean(list(self.offset_history)[:15]))
            current_mean = float(np.mean(list(self.offset_history)[15:]))
            if np.sign(past_mean) != np.sign(current_mean) and abs(current_mean) > 0.3 * lane_half_width_m:
                lane_change = True
        return (
            left_x_frame,
            right_x_frame,
            offset_px,
            offset_m,
            left_ratio,
            right_ratio,
            lane_width_px,
            departed,
            lane_change,
        )


# ---------------------------------------------------------------------------
#  Overlay drawing utilities
# ---------------------------------------------------------------------------


def draw_overlay(
    frame: np.ndarray,
    left_x: Optional[int],
    right_x: Optional[int],
    car_centre_x: int,
    offset_px: Optional[float],
    offset_m: float,
    left_ratio: float,
    right_ratio: float,
    departed: bool,
    lane_change: bool,
    show_depth_text: bool = True,
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    cv2.line(img, (car_centre_x, 0), (car_centre_x, h - 1), (0, 0, 255), 2)
    if left_x is not None:
        cv2.line(img, (left_x, 0), (left_x, h - 1), (255, 0, 0), 2)  # blue
    if right_x is not None:
        cv2.line(img, (right_x, 0), (right_x, h - 1), (255, 0, 255), 2)  # purple
    if left_x is not None and right_x is not None:
        lane_centre = int((left_x + right_x) / 2)
        for y in range(0, h, 10):
            cv2.line(img, (lane_centre, y), (lane_centre, min(h - 1, y + 5)), (0, 255, 0), 1)
    if show_depth_text:
        lines: List[str] = [
            f"Left ratio : {left_ratio:6.2f}%",
            f"Right ratio: {right_ratio:6.2f}%",
        ]
        if offset_px is not None:
            lines.append(f"Offset(px): {offset_px:+.1f}")
        lines.append(f"Offset(m) : {offset_m:+.2f}")
        lines.append(f"Departed? : {'YES' if departed else 'NO'}")
        lines.append(f"Lane change?: {'YES' if lane_change else 'NO'}")
        pad_x = 10
        pad_y = 5
        line_height = 20
        box_width = max(len(s) for s in lines) * 8 + 2 * pad_x
        box_height = len(lines) * line_height + 2 * pad_y
        cv2.rectangle(img, (0, 0), (box_width, box_height), (0, 0, 0), -1)
        cv2.rectangle(img, (0, 0), (box_width, box_height), (255, 255, 255), 1)
        for i, line in enumerate(lines):
            cv2.putText(
                img,
                line,
                (pad_x, pad_y + (i + 1) * line_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return img


# ---------------------------------------------------------------------------
#  Public helpers
# ---------------------------------------------------------------------------


def load_roi_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"ROI mask not found: {mask_path}")
    return (mask > 0).astype(np.uint8) * 255


class LaneDepartureAnalyzer:
    """High-level helper that ties segmentation and overlay drawing together."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        threshold: float = 0.5,
        use_resnet: bool = False,  # kept for compatibility, unused
        vehicle_center_x: Optional[float] = None,
        bev_size: Tuple[int, int] = (700, 1000),
        roi_points: Tuple[Tuple[float, float], ...] = (
            (294.0, 264.0),
            (99.0, 349.0),
            (803.0, 355.0),
            (673.0, 268.0),
        ),
        lane_width_m: float = 3.5,
        depart_ratio_threshold: float = 50.0,
    ) -> None:
        self.segmenter = LaneSegmenter(model_path=Path(model_path), device=device, threshold=threshold)
        self.geometry = LaneGeometry(
            bev_width=bev_size[0],
            bev_height=bev_size[1],
            src_points=roi_points,
            lane_width_m=lane_width_m,
            depart_ratio_threshold=depart_ratio_threshold,
        )
        self.vehicle_center_x = vehicle_center_x

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
        bottom_ratio: float = 0.3,
        return_metrics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Union[int, float, bool, None]]]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask = self.segmenter.predict(frame_rgb)
        if 0.0 < bottom_ratio < 1.0:
            h = mask.shape[0]
            cut = max(0, min(h, int(h * (1.0 - bottom_ratio))))
            mask[cut:, :] = 0
        if roi_mask is not None:
            roi = roi_mask
            if roi.shape != mask.shape:
                roi = cv2.resize(roi, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_and(mask, roi)
        bev = self.geometry.warp_to_bev(mask)
        (
            left_x_frame,
            right_x_frame,
            offset_px,
            offset_m,
            left_ratio,
            right_ratio,
            _lane_width_px,
            departed,
            lane_change,
        ) = self.geometry.compute_metrics(bev, frame_width=frame_bgr.shape[1], frame_height=frame_bgr.shape[0])
        car_centre = int(self.vehicle_center_x) if self.vehicle_center_x is not None else frame_bgr.shape[1] // 2
        overlay = draw_overlay(
            frame=frame_bgr,
            left_x=left_x_frame,
            right_x=right_x_frame,
            car_centre_x=car_centre,
            offset_px=offset_px,
            offset_m=offset_m,
            left_ratio=left_ratio,
            right_ratio=right_ratio,
            departed=departed,
            lane_change=lane_change,
        )
        metrics = {
            "left_x": left_x_frame,
            "right_x": right_x_frame,
            "offset_px": float(offset_px) if offset_px is not None else None,
            "offset_m": float(offset_m),
            "left_ratio": float(left_ratio),
            "right_ratio": float(right_ratio),
            "departed": bool(departed),
            "lane_change": bool(lane_change),
        }
        if return_metrics:
            return overlay, metrics
        return overlay


def generate_overlay_video(args, analyzer: LaneDepartureAnalyzer, roi_mask: Optional[np.ndarray]) -> Path:
    video_path = Path(args.video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not input_fps or input_fps <= 1e-3:
        input_fps = float(args.video_fps)
    output_fps = float(input_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not determine video dimensions.")

    output_path = Path(args.video_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {output_path}")

    frame_idx = 0
    max_frames = getattr(args, "video_max_frames", None)
    window = max(1, int(output_fps * getattr(args, "video_window_sec", 1.0)))
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1
            overlay = analyzer.process_frame(frame_bgr, roi_mask=roi_mask, bottom_ratio=args.bottom_ratio)
            writer.write(overlay)
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % window == 0:
                print(f"[Info] processed {frame_idx} frames")
    finally:
        cap.release()
        writer.release()
    return output_path
