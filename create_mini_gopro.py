import argparse
import os
import shutil
from pathlib import Path


def link_or_copy(src, dst, copy_files=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_files:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collect_videos(data_root, flow_root, split):
    data_split = data_root / split
    flow_split = flow_root / split
    if not data_split.is_dir():
        raise FileNotFoundError(f"Missing data split: {data_split}")
    if not flow_split.is_dir():
        raise FileNotFoundError(f"Missing flow split: {flow_split}")

    videos = []
    for video_dir in sorted(data_split.iterdir()):
        if not video_dir.is_dir():
            continue
        video = video_dir.name
        if (flow_split / video).is_dir():
            videos.append(video)
    return videos


def create_split(
    data_root,
    flow_root,
    out_data_root,
    out_flow_root,
    split,
    max_videos,
    max_frames,
    copy_files,
):
    videos = collect_videos(data_root, flow_root, split)[:max_videos]
    total = 0

    for video in videos:
        flow_files = sorted((flow_root / split / video).glob("*.npy"))[:max_frames]
        for flow_file in flow_files:
            stem = flow_file.stem
            blur_file = data_root / split / video / "blur" / f"{stem}.png"
            sharp_file = data_root / split / video / "sharp" / f"{stem}.png"
            if not blur_file.exists() or not sharp_file.exists():
                print(f"skip missing pair: {split}/{video}/{stem}")
                continue

            link_or_copy(
                flow_file,
                out_flow_root / split / video / flow_file.name,
                copy_files=copy_files,
            )
            link_or_copy(
                blur_file,
                out_data_root / split / video / "blur" / blur_file.name,
                copy_files=copy_files,
            )
            link_or_copy(
                sharp_file,
                out_data_root / split / video / "sharp" / sharp_file.name,
                copy_files=copy_files,
            )
            total += 1

    print(f"{split}: {len(videos)} videos, {total} paired frames")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a small paired GoPro + GOPRO_flow dataset for smoke tests."
    )
    parser.add_argument("--data_path", default="./dataset/GOPRO_Large")
    parser.add_argument("--flow_path", default="./dataset/GOPRO_flow")
    parser.add_argument("--out_data_path", default="./dataset/GOPRO_Large_mini")
    parser.add_argument("--out_flow_path", default="./dataset/GOPRO_flow_mini")
    parser.add_argument("--train_videos", type=int, default=1)
    parser.add_argument("--test_videos", type=int, default=1)
    parser.add_argument("--train_frames", type=int, default=32)
    parser.add_argument("--test_frames", type=int, default=8)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of hardlinking them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_path)
    flow_root = Path(args.flow_path)
    out_data_root = Path(args.out_data_path)
    out_flow_root = Path(args.out_flow_path)

    create_split(
        data_root=data_root,
        flow_root=flow_root,
        out_data_root=out_data_root,
        out_flow_root=out_flow_root,
        split="train",
        max_videos=args.train_videos,
        max_frames=args.train_frames,
        copy_files=args.copy,
    )
    create_split(
        data_root=data_root,
        flow_root=flow_root,
        out_data_root=out_data_root,
        out_flow_root=out_flow_root,
        split="test",
        max_videos=args.test_videos,
        max_frames=args.test_frames,
        copy_files=args.copy,
    )

    print(f"mini data: {out_data_root}")
    print(f"mini flow: {out_flow_root}")


if __name__ == "__main__":
    main()
