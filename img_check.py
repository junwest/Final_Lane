#!/usr/bin/env python3
"""전체 이벤트 영상을 차선 오버레이 이미지로 변환하는 스크립트.(fps 10)

- 960 * 540 해상도
- 전체 영상을 10fps로 변환
- 영상을 차선 오버레이 이미지로 변환
- 차량 중심선 빨간색 선 추가
- 차량 중심선은 코드를 변경하기 쉽게 코드 맨 위에 변수로 정의
- 중앙(초록), left(파랑), right(보라) 차선 오버레이 이미지 저장
- 왼쪽 위에 left ratio, right ratio, Departed, Lane change 텍스트 추가
- 오버레이 이미지를 저장 (img_check/<이벤트명>_img_check_<인덱스>.png)
- 각 <이벤트명> 파일 내에 LLM이 참조 가능한 JSON 파일 생성
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from make_masks import LaneDepartureAnalyzer, load_roi_mask  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "lane_detect.pth"
DEFAULT_ROI_PATH = PROJECT_ROOT / "data" / "masks" / "masked.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "img_check"

TARGET_WIDTH = 960
TARGET_HEIGHT = 540
TARGET_FPS = 10.0
DEFAULT_VEHICLE_CENTRE_X = 430.0
IMAGE_PATTERN = "{event}_img_check_{index:04d}.png"
SUMMARY_PATTERN = "{event}_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이벤트 영상을 이미지 오버레이로 변환")
    parser.add_argument("--video-path", type=Path, required=True, help="입력 영상 경로")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="이미지/JSON이 저장될 디렉터리",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="모델 가중치 경로")
    parser.add_argument("--roi-mask", type=Path, default=DEFAULT_ROI_PATH, help="ROI 마스크 경로")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="추론 디바이스",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="분할 임계값")
    parser.add_argument("--bottom-ratio", type=float, default=0.3, help="하단 제거 비율")
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS, help="샘플링 FPS (기본 10)")
    parser.add_argument(
        "--vehicle-centre-x",
        type=float,
        default=DEFAULT_VEHICLE_CENTRE_X,
        help="차량 중심선 X 좌표",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="저장할 최대 프레임 수 (디버그용)",
    )
    return parser.parse_args()


def ensure_roi(mask_path: Path) -> Optional[np.ndarray]:
    if mask_path.exists():
        return load_roi_mask(mask_path)
    print(f"[Warn] ROI 마스크를 찾을 수 없습니다: {mask_path}")
    return None


def main() -> None:
    args = parse_args()
    if not args.video_path.exists():
        print(f"[Error] 입력 영상을 찾을 수 없습니다: {args.video_path}")
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    event_name = args.video_path.stem.replace(" ", "")

    print(f"[Info] 모델 로드 중... ({args.model_path})")
    analyzer = LaneDepartureAnalyzer(
        model_path=str(args.model_path),
        device=args.device,
        threshold=args.threshold,
        use_resnet=False,
        vehicle_center_x=args.vehicle_centre_x,
    )
    roi_mask = ensure_roi(args.roi_mask)

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        print(f"[Error] 영상을 열 수 없습니다: {args.video_path}")
        sys.exit(1)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not input_fps or input_fps <= 1e-3:
        input_fps = args.target_fps
    interval = 1.0 / args.target_fps
    next_capture_time = 0.0

    frame_idx = 0
    saved = 0
    records = []

    print(f"[Info] 샘플링 FPS: {args.target_fps} (입력 FPS: {input_fps:.2f})")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_idx / input_fps
        frame_idx += 1

        if current_time + 1e-9 < next_capture_time:
            continue
        next_capture_time += interval

        resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        overlay, metrics = analyzer.process_frame(
            resized,
            roi_mask=roi_mask,
            bottom_ratio=args.bottom_ratio,
            return_metrics=True,
        )

        image_path = output_dir / IMAGE_PATTERN.format(event=event_name, index=saved)
        cv2.imwrite(str(image_path), overlay)

        metrics_record = {
            "capture_index": saved,
            "source_frame_index": frame_idx - 1,
            "timestamp_sec": round(current_time, 3),
            "image": str(image_path),
        }
        metrics_record.update(metrics)
        records.append(metrics_record)
        saved += 1

        if args.max_samples is not None and saved >= args.max_samples:
            break
        if saved % 50 == 0:
            print(f"[Info] 저장된 이미지 수: {saved}")

    cap.release()

    summary = {
        "video": str(args.video_path),
        "event": event_name,
        "target_fps": args.target_fps,
        "input_fps": input_fps,
        "vehicle_centre_x": args.vehicle_centre_x,
        "bottom_ratio": args.bottom_ratio,
        "frames_saved": saved,
        "images": records,
    }
    json_path = output_dir / SUMMARY_PATTERN.format(event=event_name)
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(f"[Done] 이미지 {saved}장을 저장했습니다.")
    print(f"[Done] JSON 요약: {json_path}")


if __name__ == "__main__":
    main()
