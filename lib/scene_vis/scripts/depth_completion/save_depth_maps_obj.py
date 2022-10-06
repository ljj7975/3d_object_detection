import os
import sys
import time

import numpy as np

from scene_vis.core.builders.dataset_builder import DatasetBuilder
from scene_vis.datasets.kitti.obj import obj_utils, calib_utils
from ip_basic import ip_basic
from nova.core.depth_map import DepthMap
from nova.core.camera_info import CameraInfo

def main():
    """Interpolates the lidar point cloud using IP-Basic
    and saves a dense depth map of the scene.
    https://github.com/kujason/ip_basic
    """

    ##############################
    # Options
    ##############################

    dataset = DatasetBuilder.build_kitti_obj_dataset(DatasetBuilder.KITTI_TRAINVAL)

    # Fill algorithm ('ip_basic_{...}')
    fill_type = 'multiscale'

    save_depth_maps = True

    out_depth_map_dir = 'outputs/obj/depth_2_{}'.format(fill_type)

    samples_to_use = None
    # samples_to_use = ['000764']

    ##############################
    # End of Options
    ##############################
    os.makedirs(out_depth_map_dir, exist_ok=True)

    # Rolling average array of times for time estimation
    avg_time_arr_length = 5
    last_fill_times = np.repeat([1.0], avg_time_arr_length)
    last_total_times = np.repeat([1.0], avg_time_arr_length)

    if samples_to_use is None:
        samples_to_use = [sample.name for sample in dataset.sample_list]

    for sample_idx, sample_name in enumerate(samples_to_use):

        # Calculate average time with last n fill times
        avg_fill_time = np.mean(last_fill_times)
        avg_total_time = np.mean(last_total_times)

        # Print progress
        sys.stdout.write('\rProcessing {} / {}, Idx {}, Avg Fill Time: {:.5f}s, '
                         'Avg Time: {:.5f}s, Est Time: {:.3f}s'.format(
                             sample_idx, dataset.num_samples - 1, sample_name,
                             avg_fill_time, avg_total_time,
                             avg_total_time * (dataset.num_samples - sample_idx)))
        sys.stdout.flush()

        # Start timing
        start_total_time = time.time()

        # Load sample info
        image = obj_utils.get_image(sample_name, dataset.image_2_dir)
        image_shape = image.shape[0:2]
        frame_calib = calib_utils.get_frame_calib(dataset.calib_dir, sample_name)
        cam_p = frame_calib.p2

        # Load point cloud
        point_cloud = obj_utils.get_lidar_point_cloud(sample_name, frame_calib, dataset.velo_dir)

        # Fill depth map
        if fill_type == 'multiscale':
            cam_info = CameraInfo.from_mats(cam_p[0:3, 0:3], cam_p, [0.0] * 5)
            # Project point cloud to depth map
            projected_depths = DepthMap.from_point_cloud(point_cloud, cam_info, image_shape)

            start_fill_time = time.time()
            final_depth_map, _ = ip_basic.fill_in_multiscale(projected_depths)
            end_fill_time = time.time()
        else:
            raise ValueError('Invalid fill algorithm')

        # Save depth maps
        if save_depth_maps:
            out_depth_map_path = out_depth_map_dir + '/{}.png'.format(sample_name)
            DepthMap(final_depth_map).save(out_depth_map_path)

        # Stop timing
        end_total_time = time.time()

        # Update fill times
        last_fill_times = np.roll(last_fill_times, -1)
        last_fill_times[-1] = end_fill_time - start_fill_time

        # Update total times
        last_total_times = np.roll(last_total_times, -1)
        last_total_times[-1] = end_total_time - start_total_time


if __name__ == "__main__":
    main()
