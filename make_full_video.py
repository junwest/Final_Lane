#!/usr/bin/env python3
"""전체 이벤트 영상을 차선 오버레이 비디오로 변환하는 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from make_masks import LaneDepartureAnalyzer, load_roi_mask, generate_overlay_video

DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "lane_detect.pth"
DEFAULT_ROI_PATH = PROJECT_ROOT / "data" / "masks" / "masked.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="전체 영상 차선 탐지 오버레이 생성기")
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="처리할 원본 동영상 파일 경로 (예: data/이벤트 4.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="결과 동영상이 저장될 디렉터리",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="결과 파일 이름(확장자 제외). 지정하지 않으면 overlay_<stem> 사용",
    )
    parser.add_argument(
        "--output-ext",
        type=str,
        default=".mov",
        choices=[".mov", ".mp4"],
        help="결과 동영상 확장자",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="학습된 모델 파일 경로",
    )
    parser.add_argument(
        "--roi-mask",
        type=Path,
        default=DEFAULT_ROI_PATH,
        help="ROI 마스크 이미지 경로",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="추론에 사용할 디바이스",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="차선 마스크 이진화 임계값",
    )
    parser.add_argument(
        "--bottom-ratio",
        type=float,
        default=0.3,
        help="차량 보닛 등 하단 불필요 영역 제거 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="출력 비디오 FPS (원본 FPS를 못 읽을 경우 사용)",
    )
    parser.add_argument(
        "--vehicle-centre-x",
        type=float,
        default=450.0,
        help="차량 중심선 X 좌표 (픽셀)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video_path.exists():
        print(f"[Error] 입력 영상을 찾을 수 없습니다: {args.video_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem_no_space = args.video_path.stem.replace(" ", "")
    base_name = args.output_name or f"overlay_{stem_no_space}"
    suffix = args.output_ext if args.output_ext.startswith(".") else f".{args.output_ext}"
    output_path = args.output_dir / f"{base_name}{suffix}"

    print(f"[Info] 모델 로드 중... ({args.model_path})")
    print(f"[Info] 디바이스: {args.device}")

    analyzer = LaneDepartureAnalyzer(
        model_path=str(args.model_path),
        device=args.device,
        threshold=args.threshold,
        use_resnet=False,
        vehicle_center_x=args.vehicle_centre_x,
    )

    roi_mask = None
    if args.roi_mask.exists():
        roi_mask = load_roi_mask(args.roi_mask)
        print(f"[Info] ROI 마스크 적용: {args.roi_mask}")
    else:
        print("[Warn] ROI 마스크를 찾을 수 없어 적용하지 않습니다.")

    print(f"[Info] 영상 처리 시작: {args.video_path}")
    print(f"       저장 경로: {output_path}")

    class VideoArgs:
        video_path = args.video_path
        video_output = output_path
        video_fps = args.video_fps
        video_window_sec = 2.0
        video_max_frames = None
        threshold = args.threshold
        bottom_ratio = args.bottom_ratio

    video_args = VideoArgs()

    try:
        saved_path = generate_overlay_video(video_args, analyzer, roi_mask)
        print(f"\n[Done] 처리가 완료되었습니다. 파일: {saved_path}")
    except Exception as exc:  # pragma: no cover
        print(f"\n[Error] 영상 처리 중 오류 발생: {exc}")
        raise


if __name__ == "__main__":
    main()
