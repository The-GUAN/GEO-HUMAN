import os
import torch
import trimesh
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np
import warnings
import os.path as osp
import lib
from lib.models import load_tokenhmr
from lib.utils import recursive_to
from lib.datasets.vitdet_dataset import ViTDetDataset
from detectron2.engine.defaults import DefaultPredictor
from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
from lib.utils import select_xy_supervising_view


def calculate_weighted_body_pose_v2(all_body_poses, weights, control_points):
    weighted_rotation_matrices = torch.zeros_like(all_body_poses[0])

    for idx, body_pose in enumerate(all_body_poses):
        view_angle = list(weights.keys())[idx]
        weight = weights[view_angle]
        points = control_points[view_angle]
        for p in points:
            x_weight, y_weight, z_weight = weight['x'], weight['y'], weight['z']
            weighted_rotation_matrices[:, p, :, :] += (
                body_pose[:, p, :, :] * torch.tensor([x_weight, y_weight, z_weight],
                                                     device=body_pose.device).view(1, 3, 1)
            )

    return weighted_rotation_matrices



def main():
    import argparse
    parser = argparse.ArgumentParser(description='TokenHMR Weighted Body Pose Processing')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to pretrained model checkpoint')
    parser.add_argument('--model_config', type=str, default='model_config.yaml', help='Path to model config file')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='weighted_body_pose_out',
                        help='Output folder to save results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    args = parser.parse_args()

    model, model_cfg = load_tokenhmr(checkpoint_path=args.checkpoint,
                                     model_cfg=args.model_config,
                                     is_train_state=False, is_demo=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    cfg_path = Path(lib.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    img_paths = sorted(list(Path(args.img_folder).glob('*')))[:4]
    assert len(img_paths) == 4
    all_body_poses = []


    for img_path in img_paths:
        print(f"Processing image: {img_path}")
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            print(f"Error: Could not read image at {img_path}. Skipping.")
            continue

        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        if boxes is None or len(boxes) == 0:
            print(f"No valid boxes found in {img_path}. Skipping.")
            continue

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            body_pose = out['pred_smpl_params']['body_pose']
            print(f"Extracted body_pose shape: {body_pose.shape}")
            all_body_poses.append(body_pose)

    if len(all_body_poses) == len(img_paths):
        weights = {
            '0': {'x': 0.8, 'y': 0.5, 'z': 0.2},
            '90': {'x': 0.2, 'y': 0.5, 'z': 0.8},
            '270': {'x': 0.2, 'y': 0.5, 'z': 0.8},
            '180': {'x': 0.8, 'y': 0.5, 'z': 0.2},
        }

        torso_joints = [0, 1, 2, 3, 6]
        left_limb_joints = [4, 7, 10, 13, 16, 18, 20, 22]
        right_limb_joints = [5, 8, 11, 14, 17, 19, 21, 23]
        front_or_back_for_left = select_xy_supervising_view(left_limb_joints)
        front_or_back_for_right = select_xy_supervising_view(right_limb_joints)

        control_points = {
            '0': torso_joints if front_or_back_for_left == '0' else [],
            '180': torso_joints if front_or_back_for_left == '180' else [],
            '90': torso_joints,
            '270': [],
        }

        control_points['90'].extend(left_limb_joints)
        control_points['270'].extend(right_limb_joints)
        if front_or_back_for_left == '0':
            control_points['0'].extend(left_limb_joints)
        else:
            control_points['180'].extend(left_limb_joints)

        if front_or_back_for_right == '0':
            control_points['0'].extend(right_limb_joints)
        else:
            control_points['180'].extend(right_limb_joints)

        avg_weighted_body_pose = calculate_weighted_body_pose_v2(all_body_poses, weights, control_points)
        print(f"Weighted Body Pose Shape: {avg_weighted_body_pose.shape}")
    else:
        print("Error: Not all poses were collected. Skipping weighted calculation.")
        return

    print(f"Generating final weighted SMPL model...")
    pred_smpl_params = {'body_pose': avg_weighted_body_pose}

    smpl_output = model.smpl(**{k: v for k, v in pred_smpl_params.items()}, pose2rot=True)
    path = args.out_folder
    joints = smpl_output.joints
    pelvis_joint_coord = joints[:, 0, :].squeeze().cpu().numpy()
    pelvis_file = osp.join(path, "pelvis_joint_coordinates.txt")
    np.savetxt(pelvis_file, pelvis_joint_coord, fmt="%.6f", delimiter=" ", header="Pelvis Joint Coordinates (x, y, z)")
    print(f"Pelvis joint coordinates saved to: {pelvis_file}")
    vertices = smpl_output.vertices

    os.makedirs(path, exist_ok=True)
    mesh_file = osp.join(path, "weighted_smpl.obj")
    mesh = trimesh.Trimesh(vertices.squeeze().cpu().numpy(), model.smpl.faces)
    mesh.export(mesh_file)
    print(f"Final SMPL model saved to: {mesh_file}")


if __name__ == '__main__':
    main()
