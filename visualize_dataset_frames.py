"""This code was largely written by an LLM."""

from datasets import load_dataset
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def save_trajectory_frames(
    sequences_unnormalized,
    labels_true,
    dataset_name,
    seq_name,
    seq_type="full",
    frame_indices=None,
):
    """
    Saves selected frames as PNG images overlayed on the original video.
    """

    def _unwrap_one(x):
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 1:
            return _unwrap_one(x[0])
        if torch.is_tensor(x) and (x.ndim == 0 or x.numel() == 1):
            try:
                return x.item()
            except Exception:
                pass
        return x

    seq_name_str = str(_unwrap_one(seq_name))
    seq_type_str = str(_unwrap_one(seq_type)) if seq_type is not None else "full"

    in_video_path = f"data/{dataset_name}/{seq_name_str}/{seq_name_str}.avi"
    out_image_directory = f"data/{dataset_name}/{seq_name_str}/out/"
    os.makedirs(out_image_directory, exist_ok=True)

    cap = cv2.VideoCapture(in_video_path)
    have_video = cap.isOpened()
    if have_video:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        frame_width, frame_height = 640, 480
        print(
            f"Warning: Could not open video {in_video_path}, "
            "falling back to black background."
        )

    su = sequences_unnormalized
    if torch.is_tensor(su):
        su = su.detach().cpu()
    elif isinstance(su, np.ndarray):
        su = torch.from_numpy(su)
    else:
        raise TypeError(
            "sequences_unnormalized must be a torch.Tensor or numpy.ndarray"
        )

    def _get_points_for_frame(su_t: "torch.Tensor", idx: int) -> "torch.Tensor":
        if su_t.ndim == 4:
            if su_t.shape[-1] == 2:
                return su_t[0, :, idx, :]
            if su_t.shape[2] == 2:
                return su_t[0, :, :, idx]
        elif su_t.ndim == 3:
            if su_t.shape[-1] == 2:
                return su_t[:, idx, :]
            if su_t.shape[1] == 2:
                return su_t[:, :, idx]
        raise ValueError(
            f"Unsupported unnormalized trajectories shape: {tuple(su_t.shape)}"
        )

    if su.ndim == 4:
        num_frames = su.shape[2] if su.shape[-1] == 2 else su.shape[3]
    elif su.ndim == 3:
        num_frames = su.shape[1] if su.shape[-1] == 2 else su.shape[2]
    else:
        raise ValueError(
            f"Unsupported unnormalized trajectories shape: {tuple(su.shape)}"
        )

    if not frame_indices:
        frame_indices = [num_frames // 2]
    frame_indices = [i for i in frame_indices if 0 <= i < num_frames]
    if not frame_indices:
        print("No valid frame indices to save; all out of range.")
        if have_video:
            cap.release()
        return

    # colors
    cluster_colors_bgr = [
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 128, 128),
        (0, 165, 255),
        (128, 0, 255),
        (0, 128, 0),
    ]

    labels_array = np.asarray(labels_true).reshape(-1)

    for frame_idx in frame_indices:
        pts = _get_points_for_frame(su, frame_idx)  # [P,2]

        if pts.shape[0] != labels_array.shape[0]:
            m = min(pts.shape[0], labels_array.shape[0])
            pts = pts[:m]
            labels_use = labels_array[:m]
            print(
                f"Warning: label/point mismatch for frame {frame_idx}. Trimming to {m}."
            )
        else:
            labels_use = labels_array

        if have_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                print(
                    f"Could not read frame {frame_idx} from {in_video_path}; "
                    "using black background."
                )
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        else:
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        for i in range(pts.shape[0]):
            x = int(round(float(pts[i, 0].item())))
            y = int(round(float(pts[i, 1].item())))
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                continue
            label = int(labels_use[i])
            color = cluster_colors_bgr[label % len(cluster_colors_bgr)]
            cv2.circle(frame, (x, y), radius=3, color=color, thickness=-1)

        if str(seq_type_str).lower() != "full":
            filename = (
                f"trajectory_{seq_name_str}_{seq_type_str}_frame_{frame_idx:04d}.png"
            )
        else:
            filename = f"trajectory_{seq_name_str}_frame_{frame_idx:04d}.png"

        filepath = os.path.join(out_image_directory, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved trajectory frame: {filepath}")

    if have_video:
        cap.release()


def save_all_sequence_trajectory_frames(dataset_name="Hopkins155", frame_indices=None):
    """
    Save trajectory frames for all sequences in a dataset.
    """
    dataset = load_dataset(dataset_name.lower())

    try:
        total = len(dataset)
        print(f"Processing {total} sequences from {dataset_name}")
    except Exception:
        print(f"Processing sequences from {dataset_name}")

    for i, sequence in enumerate(dataset):
        seq_name = sequence["seq_name"]
        seq_type = sequence.get("seq_type", ["full"])

        if "unnormalized_trajectories" not in sequence:
            print(f"Skipping {seq_name} - no unnormalized trajectories")
            continue

        seq_unnormalized = sequence["unnormalized_trajectories"]
        labels_true = sequence["labels"].squeeze(0)

        print(f"Processing sequence {i + 1}: {seq_name} ({seq_type})")

        try:
            save_trajectory_frames(
                sequences_unnormalized=seq_unnormalized,
                labels_true=labels_true,
                dataset_name=dataset_name,
                seq_name=seq_name,
                seq_type=seq_type,
                frame_indices=frame_indices,
            )
        except Exception as e:
            print(f"Error processing {seq_name}: {e}")
            continue

    print("Finished processing all sequences")


def save_trajectory_frames_grid(
    sequences_unnormalized,
    labels_true,
    dataset_name,
    seq_name,
    seq_type="full",
    frame_indices=None,
    num_frames=6,
    cols=3,
):
    """
    Save a grid of trajectory frames in a single image.
    """

    def _unwrap_one(x):
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 1:
            return _unwrap_one(x[0])
        if torch.is_tensor(x) and (x.ndim == 0 or x.numel() == 1):
            try:
                return x.item()
            except Exception:
                pass
        return x

    seq_name_str = str(_unwrap_one(seq_name))
    seq_type_str = str(_unwrap_one(seq_type)) if seq_type is not None else "full"

    in_video_path = f"data/{dataset_name}/{seq_name_str}/{seq_name_str}.avi"
    out_image_directory = f"data/{dataset_name}/{seq_name_str}/out/"
    os.makedirs(out_image_directory, exist_ok=True)

    cap = cv2.VideoCapture(in_video_path)
    have_video = cap.isOpened()
    if have_video:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        frame_width, frame_height = 640, 480
        print(
            f"Warning: Could not open video {in_video_path}, using default dimensions"
        )

    su = sequences_unnormalized
    if torch.is_tensor(su):
        su = su.detach().cpu()
    elif isinstance(su, np.ndarray):
        su = torch.from_numpy(su)
    else:
        raise TypeError(
            "sequences_unnormalized must be a torch.Tensor or numpy.ndarray"
        )

    def _get_points_for_frame(su_t, idx):
        if su_t.ndim == 4:
            if su_t.shape[-1] == 2:
                return su_t[0, :, idx, :]
            if su_t.shape[2] == 2:
                return su_t[0, :, :, idx]
        elif su_t.ndim == 3:
            if su_t.shape[-1] == 2:
                return su_t[:, idx, :]
            if su_t.shape[1] == 2:
                return su_t[:, :, idx]
        raise ValueError(
            f"Unsupported unnormalized trajectories shape: {tuple(su_t.shape)}"
        )

    if su.ndim == 4:
        total_frames = su.shape[2] if su.shape[-1] == 2 else su.shape[3]
    elif su.ndim == 3:
        total_frames = su.shape[1] if su.shape[-1] == 2 else su.shape[2]
    else:
        raise ValueError(
            f"Unsupported unnormalized trajectories shape: {tuple(su.shape)}"
        )

    if frame_indices is None:
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]

    frame_indices = [i for i in frame_indices if 0 <= i < total_frames]
    if not frame_indices:
        print("No valid frame indices to save; all out of range.")
        if have_video:
            cap.release()
        return

    rows = (len(frame_indices) + cols - 1) // cols

    cluster_colors_bgr = [
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 128, 128),
        (0, 165, 255),
        (128, 0, 255),
        (0, 128, 0),
    ]

    labels_array = np.asarray(labels_true).reshape(-1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[idx]

        pts = _get_points_for_frame(su, frame_idx)

        if pts.shape[0] != labels_array.shape[0]:
            m = min(pts.shape[0], labels_array.shape[0])
            pts = pts[:m]
            labels_use = labels_array[:m]
        else:
            labels_use = labels_array

        if have_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        else:
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        for i in range(pts.shape[0]):
            x = int(round(float(pts[i, 0].item())))
            y = int(round(float(pts[i, 1].item())))
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                label = int(labels_use[i])
                color = cluster_colors_bgr[label % len(cluster_colors_bgr)]
                cv2.circle(frame, (x, y), radius=3, color=color, thickness=-1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ax.imshow(frame_rgb)
        ax.set_title(f"Frame {frame_idx}", fontsize=12)
        ax.axis("off")

    for idx in range(len(frame_indices), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if str(seq_type_str).lower() != "full":
        filename = f"trajectory_grid_{seq_name_str}_{seq_type_str}.png"
    else:
        filename = f"trajectory_grid_{seq_name_str}.png"

    filepath = os.path.join(out_image_directory, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    if have_video:
        cap.release()

    print(f"Saved trajectory grid: {filepath}")


def save_all_sequence_trajectory_grids(
    dataset_name="Hopkins155", frame_indices=None, num_frames=6, cols=3
):
    """
    Save trajectory frame grids for all sequences in a dataset.
    """
    dataset = load_dataset(dataset_name.lower())

    try:
        total = len(dataset)
        print(f"Processing {total} sequences from {dataset_name}")
    except Exception:
        print(f"Processing sequences from {dataset_name}")

    for i, sequence in enumerate(dataset):
        seq_name = sequence["seq_name"]
        seq_type = sequence.get("seq_type", ["full"])

        if "unnormalized_trajectories" not in sequence:
            print(f"Skipping {seq_name} - no unnormalized trajectories")
            continue

        seq_unnormalized = sequence["unnormalized_trajectories"]
        labels_true = sequence["labels"].squeeze(0)

        print(f"Processing sequence {i + 1}: {seq_name} ({seq_type})")

        try:
            save_trajectory_frames_grid(
                sequences_unnormalized=seq_unnormalized,
                labels_true=labels_true,
                dataset_name=dataset_name,
                seq_name=seq_name,
                seq_type=seq_type,
                frame_indices=frame_indices,
                num_frames=num_frames,
                cols=cols,
            )
        except Exception as e:
            print(f"Error processing {seq_name}: {e}")
            continue

    print("Finished processing all sequences")


if __name__ == "__main__":
    # example usage:
    save_all_sequence_trajectory_grids(dataset_name="Hopkins155", num_frames=6, cols=3)
