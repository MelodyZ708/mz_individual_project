"""
Headless (no GUI) version of como_dataset.py for batch experiments.
Bypasses Open3D GUI entirely to avoid segfault on fast-motion datasets.
"""
import torch
import numpy as np
import time
import yaml
import argparse

from como.odom.sequential.TrackingSeq import TrackingSeq
from como.odom.sequential.MappingSeq import MappingSeq
from como.utils.multiprocessing import transfer_data
from como.data.dataset_factory import get_dataset
from como.gui.GuiWindow import save_traj
from torch.utils.data import DataLoader


def main(dataset, slam_cfg):
    torch.manual_seed(0)

    intrinsics = dataset.intrinsics
    img_size = dataset.img_size

    tracking = TrackingSeq(slam_cfg["tracking"], intrinsics, img_size)
    mapping = MappingSeq(slam_cfg["mapping"], intrinsics)
    tracking.setup()
    mapping.setup()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    it = iter(dataloader)

    # Record data
    timestamps = []
    est_poses = np.array([]).reshape(0, 4, 4)

    # KF storage: use dict keyed by timestamp for reliable accumulation
    kf_ts_dict = {}  # {timestamp_float: pose_tensor_4x4_cpu}
    device = torch.device("cuda:0")
    dtype = torch.float

    n = len(dataset)
    print(f"[Headless] Running on {n} frames...")

    for idx in range(n):
        timestamp, rgb = next(it)

        if mapping.is_init:
            track_data_in = (timestamp, rgb.clone())
            track_data_in = transfer_data(track_data_in, tracking.device, tracking.dtype)
            track_data_viz, track_data_map = tracking.track(track_data_in)
            track_data_viz = transfer_data(track_data_viz, device, dtype)
            tracked_timestamp, tracked_pose = track_data_viz
            timestamps.append(tracked_timestamp)
            est_poses = np.concatenate((est_poses, tracked_pose.cpu().numpy()))
        else:
            track_data_map = ("init", timestamp, rgb.clone())

        kf_viz_data, kf_ref_data = mapping.map(track_data_map)

        if kf_ref_data is not None:
            kf_ref_data = transfer_data(kf_ref_data, tracking.device, tracking.dtype)
            tracking.update_kf_reference(kf_ref_data)

        if kf_viz_data is not None:
            # NOTE: do NOT call transfer_data here - kf_ps must stay on cuda for SLAM,
            # we only copy timestamps/poses to CPU buffer for saving
            (
                kf_ts, kf_rgbs, kf_ps, kf_depths,
                kf_sparse_coords, P_sparse, obs_ref_mask,
                one_way_poses, kf_pairs, one_way_pairs,
            ) = kf_viz_data

            # Store all keyframes seen so far using a simple dict keyed by timestamp
            for ii, ts in enumerate(kf_ts):
                ts_key = float(ts)
                kf_ts_dict[ts_key] = kf_ps[ii].cpu()

        if (idx + 1) % 200 == 0:
            print(f"[Headless] {idx+1}/{n} frames processed")

    print("[Headless] Done. Saving trajectory...")

    # Save trajectory from kf_ts_dict
    if dataset.seq_path is not None and len(kf_ts_dict) > 0:
        filename = "./results/" + dataset.save_traj_name + ".txt"
        sorted_ts = sorted(kf_ts_dict.keys())
        valid_ts_list = []
        valid_poses_list = []
        for ts in sorted_ts:
            pose = kf_ts_dict[ts]
            if not torch.isnan(pose).any():
                valid_ts_list.append(ts)
                valid_poses_list.append(pose)
        if len(valid_ts_list) > 0:
            valid_poses_tensor = torch.stack(valid_poses_list, dim=0)
            save_traj(filename, valid_ts_list, valid_poses_tensor)
            print(f"Saved trajectory to {filename} ({len(valid_ts_list)} keyframes)")
        else:
            print("[Headless] WARNING: All poses contain NaN")
    else:
        print("[Headless] WARNING: No trajectory saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--dataset_dir", type=str, default=None)
    args = parser.parse_args()

    img_size = [192, 256]
    dataset = get_dataset(args.dataset_type, img_size, args.dataset_dir)

    with open("./config/como.yml", "r") as f:
        slam_cfg = yaml.safe_load(f)

    main(dataset, slam_cfg)
