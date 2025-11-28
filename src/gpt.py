"""Lane overlay video generator for the 진짜차선탐지 project.

This script reads an input driving video, uses a segmentation model to predict lane
masks for each frame, performs a bird’s‑eye view (BEV) warp over a defined
region of interest and extracts the positions of the left and right lane
markings.  Distances between the vehicle centre and the detected lanes are
computed in metres assuming a nominal lane width, and simple heuristics are
used to decide if the vehicle has departed its lane or executed a lane change.

For each frame the following information is overlaid onto the original video:

    • Vehicle centre line (red)
    • Detected left and right lane lines (blue and green)
    • A text panel listing the per‑frame left/right departure ratios, the
      lateral offset in pixels/metres, lane departure status and lane change
      status.

You can run the script directly from the command line:

    python src/lane_overlay.py \
        --video data/이벤트\\ 4.mp4 \
        --model model/best.pth \
        --output outputs/이벤트4_full_overlay.mp4

Before running, ensure that the input video and model weights exist at the
specified paths.  The output directory will be created if necessary.

This module intentionally avoids external dependencies beyond OpenCV and
PyTorch so that it can run in a standard Python environment.  If GPU is
available, PyTorch will use it automatically when predicting lane masks.
"""

import argparse
import collections
import os
from pathlib import Path
from typing import Deque, List, Optional, Tuple

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
    """A simple UNet architecture for binary segmentation.

    This is a lightweight UNet implementation adapted from the colab_model.py
    provided in the original project.  It is sufficient for predicting lane
    masks when trained on lane segmentation data.  If you already have a
    trained UNet saved in ``model_path``, it will be loaded into this
    architecture.
    """

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
    """Wrapper around a UNet segmentation network.

    Parameters
    ----------
    model_path : Path
        Path to the PyTorch state dictionary (.pth or .pt) of the trained UNet.
    device : str
        Device identifier to run inference on; "cuda", "cpu", or "auto".
    threshold : float
        Threshold for converting the sigmoid output to a binary mask.
    """

    def __init__(self, model_path: Path, device: str = "auto", threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        # instantiate model
        self.model: nn.Module = UNet(n_channels=3, n_classes=1, bilinear=False)
        # load weights
        state = torch.load(model_path, map_location=self.device)
        # accept both raw state dict or checkpoint containing model_state_dict
        if isinstance(state, dict) and any(k.startswith("model") for k in state.keys()):
            # try to find the model weights under known keys
            for key in ("model_state_dict", "state_dict", "model_state"):
                if key in state:
                    state = state[key]
                    break
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        # standard ImageNet normalisation (UNet may have been trained with similar stats)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> np.ndarray:
        """Predict a binary lane mask for an RGB image.

        Parameters
        ----------
        img : np.ndarray
            RGB image array of shape (H, W, 3) and dtype uint8.

        Returns
        -------
        mask : np.ndarray
            Binary mask of shape (H, W), dtype uint8 where lane pixels are 255.
        """
        h, w = img.shape[:2]
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob > self.threshold).astype(np.uint8) * 255
        return mask


# ---------------------------------------------------------------------------
#  Lane detection and BEV geometry
# ---------------------------------------------------------------------------

class LaneGeometry:
    """Compute lane metrics from a binary mask using a BEV perspective.

    This class encapsulates the homography transform and methods to find the
    left and right lane positions, compute distances to the vehicle centre and
    decide lane departure or lane change events.
    """

    def __init__(
        self,
        bev_width: int = 700,
        bev_height: int = 1000,
        src_points: Tuple[Tuple[float, float], ...] = (
            (294.0, 264.0),  # top-left
            (99.0, 349.0),   # bottom-left
            (803.0, 355.0),  # bottom-right
            (673.0, 268.0),  # top-right
        ),
        lane_width_m: float = 3.5,
        depart_ratio_threshold: float = 50.0,
    ) -> None:
        """Parameters
        ----------
        bev_width : int
            Width in pixels of the BEV image.
        bev_height : int
            Height in pixels of the BEV image.
        src_points : Tuple[Tuple[float, float], ...]
            Four points (x, y) in the original image defining the ROI trapezoid.
            Points should be ordered clockwise starting from top‑left.
        lane_width_m : float
            Real‑world width of a lane in metres; used for converting pixels to metres.
        depart_ratio_threshold : float
            Percentage threshold of offset beyond which the frame is marked as
            lane departure.
        """
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.src_points = np.array(src_points, dtype=np.float32)
        # destination is full rectangle of BEV
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
        # history for lane change detection (store offsets in metres)
        self.offset_history: Deque[float] = collections.deque(maxlen=30)

    def warp_to_bev(self, mask: np.ndarray) -> np.ndarray:
        """Apply the homography to obtain a top‑view BEV mask."""
        return cv2.warpPerspective(mask, self.matrix, (self.bev_width, self.bev_height))

    def find_lane_positions(self, bev: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """Locate the left and right lane x‑coordinates in BEV space.

        The algorithm sums pixels across rows and selects the row with the
        maximum number of lane pixels, then finds the cluster of lane pixels
        to the left and right of the centre of the BEV image.
        """
        # sum over columns for each row
        row_sums = bev.sum(axis=1)
        # choose the row with maximum lane presence
        row_idx = int(np.argmax(row_sums))
        # inspect a band of ±10 rows around the maximum to reduce noise
        y0 = max(0, row_idx - 10)
        y1 = min(self.bev_height - 1, row_idx + 10)
        strip = bev[y0:y1 + 1, :]
        ys, xs = np.where(strip > 0)
        if len(xs) == 0:
            return None, None
        centre = self.bev_width // 2
        left_x = None
        right_x = None
        left_candidates = xs[xs < centre]
        right_candidates = xs[xs > centre]
        if len(left_candidates) > 0:
            left_x = int(np.max(left_candidates))
        if len(right_candidates) > 0:
            right_x = int(np.min(right_candidates))
        if left_x is None or right_x is None or right_x <= left_x:
            return None, None
        return left_x, right_x

    def compute_metrics(
        self,
        bev: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Optional[int], Optional[int], Optional[float], float, float, float, float, bool, bool]:
        """Compute lane distances and departure metrics for a BEV mask.

        Returns
        -------
        left_x_frame : Optional[int]
            Approximated x‑coordinate of the left lane line in original frame space.
        right_x_frame : Optional[int]
            Approximated x‑coordinate of the right lane line in original frame space.
        offset_px : Optional[float]
            Lateral offset in pixels relative to the vehicle centre.  Positive
            means the vehicle is to the left of the lane centre.
        offset_m : float
            Lateral offset converted to metres; sign same as above.
        left_ratio : float
            Percentage of lane departure on the left side (0–100%).
        right_ratio : float
            Percentage of lane departure on the right side (0–100%).
        lane_width_px : float
            Width of the lane in BEV pixels (difference between right_x and left_x).
        departed : bool
            True if the vehicle is deemed to have departed the lane.
        lane_change : bool
            True if the vehicle has changed lanes based on history.
        """
        left_x, right_x = self.find_lane_positions(bev)
        if left_x is None or right_x is None:
            # cannot compute metrics
            self.offset_history.append(0.0)
            return None, None, None, 0.0, 0.0, 0.0, 0.0, False, False
        lane_width_px = float(right_x - left_x)
        # convert BEV positions to approximate pixel positions in the original frame
        # using a simple scaling (assume horizontal scaling only)
        left_x_frame = int(left_x / self.bev_width * frame_width)
        right_x_frame = int(right_x / self.bev_width * frame_width)
        # lane centre in BEV
        lane_centre_bev = (left_x + right_x) / 2.0
        # assume vehicle centre is in the middle of the frame
        car_centre_bev = self.bev_width / 2.0
        offset_px = car_centre_bev - lane_centre_bev
        # pixel to metre conversion using lane width
        px_per_m = lane_width_px / self.lane_width_m
        offset_m = offset_px / px_per_m
        lane_half_width_m = self.lane_width_m / 2.0
        left_ratio = 0.0
        right_ratio = 0.0
        if offset_m < 0:
            # vehicle centre is left of lane centre
            left_ratio = min(100.0, abs(offset_m) / lane_half_width_m * 100.0)
        elif offset_m > 0:
            # vehicle centre is right of lane centre
            right_ratio = min(100.0, abs(offset_m) / lane_half_width_m * 100.0)
        departed = max(left_ratio, right_ratio) >= self.depart_ratio_threshold
        # update history and detect lane change
        self.offset_history.append(offset_m)
        lane_change = False
        if len(self.offset_history) >= 30:
            past_mean = float(np.mean(list(self.offset_history)[:15]))
            current_mean = float(np.mean(list(self.offset_history)[15:]))
            if np.sign(past_mean) != np.sign(current_mean) and abs(current_mean) > 0.3 * lane_half_width_m:
                lane_change = True
        return left_x_frame, right_x_frame, offset_px, offset_m, left_ratio, right_ratio, lane_width_px, departed, lane_change


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
    """Draw lane lines and info panel onto a frame.

    Parameters
    ----------
    frame : np.ndarray
        RGB frame to draw on (will be copied).
    left_x, right_x : Optional[int]
        X coordinates of the left/right lanes in frame space.  If None, lines
        will not be drawn.
    car_centre_x : int
        X coordinate of the vehicle centre in frame space.
    offset_px : Optional[float]
        Lateral offset in pixels (vehicle centre – lane centre).  If None,
        offset will not be displayed.
    offset_m : float
        Lateral offset in metres.
    left_ratio, right_ratio : float
        Left/right departure ratios (0–100).
    departed : bool
        True if lane departure is flagged.
    lane_change : bool
        True if a lane change is detected.
    show_depth_text : bool
        If True, distances/ratios are shown; otherwise only lines are drawn.
    """
    img = frame.copy()
    h, w = img.shape[:2]
    # draw vehicle centre (red)
    cv2.line(img, (car_centre_x, 0), (car_centre_x, h - 1), (0, 0, 255), 2)
    # draw left/right lanes
    if left_x is not None:
        cv2.line(img, (left_x, 0), (left_x, h - 1), (255, 0, 0), 2)  # blue
    if right_x is not None:
        cv2.line(img, (right_x, 0), (right_x, h - 1), (0, 255, 0), 2)  # green
    # draw lane centre (dotted white)
    if left_x is not None and right_x is not None:
        lane_centre = int((left_x + right_x) / 2)
        for y in range(0, h, 10):
            cv2.line(img, (lane_centre, y), (lane_centre, y + 5), (255, 255, 255), 1)
    if show_depth_text:
        # text panel
        lines: List[str] = []
        lines.append(f"Left ratio : {left_ratio:6.2f}%")
        lines.append(f"Right ratio: {right_ratio:6.2f}%")
        if offset_px is not None:
            lines.append(f"Offset(px): {offset_px:+.1f}")
        lines.append(f"Offset(m) : {offset_m:+.2f}")
        lines.append(f"Departed? : {'YES' if departed else 'NO'}")
        lines.append(f"Lane change?: {'YES' if lane_change else 'NO'}")
        # draw box
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
#  Main processing function
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    model_path: Path,
    output_path: Path,
    roi_points: Tuple[Tuple[float, float], ...] = (
        (294.0, 264.0),
        (99.0, 349.0),
        (803.0, 355.0),
        (673.0, 268.0),
    ),
    bev_size: Tuple[int, int] = (700, 1000),
    lane_width_m: float = 3.5,
    depart_ratio_threshold: float = 50.0,
    threshold: float = 0.5,
    device: str = "auto",
    fps_override: Optional[float] = None,
    car_centre_x: Optional[float] = None,
) -> None:
    """Generate an overlay video from an input driving video.

    Parameters
    ----------
    video_path : Path
        Path to the input video file (e.g., ``이벤트 4.mp4``).
    model_path : Path
        Path to the trained segmentation model weights (.pth).
    output_path : Path
        Path to save the generated overlay video (MP4).
    roi_points : tuple of tuple
        Four vertices defining the trapezoid ROI in the original frame for BEV.
    bev_size : tuple of ints
        (width, height) of the BEV image.
    lane_width_m : float
        Real‑world lane width in metres.
    depart_ratio_threshold : float
        Departure ratio threshold for flagging lane departures.
    threshold : float
        Segmentation threshold for binary mask.
    device : str
        Device identifier for the segmentation model.  Use ``auto`` to let
        PyTorch choose ``cuda`` if available.
    fps_override : float, optional
        Override the FPS of the output video.  If not provided, the FPS of
        the input video is used.
    """
    # ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # load segmentation model
    segmenter = LaneSegmenter(model_path=model_path, device=device, threshold=threshold)
    # lane geometry handler
    geom = LaneGeometry(
        bev_width=bev_size[0],
        bev_height=bev_size[1],
        src_points=roi_points,
        lane_width_m=lane_width_m,
        depart_ratio_threshold=depart_ratio_threshold,
    )
    # open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not input_fps or input_fps <= 1e-3:
        input_fps = fps_override if fps_override is not None else 30.0
    output_fps = fps_override if fps_override is not None else input_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not determine video dimensions.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    # process frames
    car_centre = int(car_centre_x) if car_centre_x is not None else width // 2
    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1
            # convert to RGB for segmentation
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mask = segmenter.predict(frame_rgb)
            # remove bottom 30% of the mask to avoid hood
            mask_clean = mask.copy()
            h_mask = mask_clean.shape[0]
            mask_clean[int(h_mask * 0.6):, :] = 0
            # warp to BEV
            bev = geom.warp_to_bev(mask_clean)
            # compute metrics
            (
                left_x_frame,
                right_x_frame,
                offset_px,
                offset_m,
                left_ratio,
                right_ratio,
                lane_width_px,
                departed,
                lane_change,
            ) = geom.compute_metrics(bev, frame_width=width, frame_height=height)
            # overlay on original frame (convert BGR->RGB->BGR for consistency)
            overlay = draw_overlay(
                frame=cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
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
            writer.write(overlay)
    finally:
        cap.release()
        writer.release()


# ---------------------------------------------------------------------------
#  Command line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lane overlay video from a driving video.")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video (e.g., 'data/이벤트 4.mp4')",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained segmentation model (.pth)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the output overlay video (.mp4)",
    )
    parser.add_argument(
        "--car-centre-x",
        type=float,
        default=None,
        help="Override the vehicle centre x position (pixels).  If omitted, the frame centre is used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold for binary mask (0–1)",
    )
    parser.add_argument(
        "--lane-width-m",
        type=float,
        default=3.5,
        help="Real‑world lane width in metres",
    )
    parser.add_argument(
        "--depart-threshold",
        type=float,
        default=50.0,
        help="Percentage threshold for lane departure detection",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS for the output video; if not set, use input video FPS",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_video(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        lane_width_m=args.lane_width_m,
        depart_ratio_threshold=args.depart_threshold,
        threshold=args.threshold,
        fps_override=args.fps,
        car_centre_x=args.car_centre_x,
    )


if __name__ == "__main__":
    main()