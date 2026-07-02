import time

from como.utils.multiprocessing import transfer_data
from como.odom.Mapping import Mapping


class MappingSeq(Mapping):
    def __init__(self, cfg, intrinsics):
        super().__init__(cfg, intrinsics)

    def map(self, data):
        kf_viz_data = None
        kf_ref_data = None

        kf_updated = False

        # Handle incoming data
        if data is not None:
            data = transfer_data(data, self.device, self.dtype)

            if not self.is_init:
                if data[0] == "init":
                    timestamp, rgb = data[1:]
                    kf_updated = self.attempt_two_frame_init(timestamp, rgb)
                    
            else:
                if data[0] == "keyframe_world":
                    kf_viz_data, kf_updated = self.handle_world_pose_data(data)
                else:
                    kf_viz_data, kf_updated = self.handle_tracking_data(data)

        # Mapping iteration
        if self.is_init and not self.converged:
            self.converged = self.iterate()
            kf_updated = True

        # Send updated mapping data if not sent for awhile
        curr_time = time.time()
        if self.is_init and (curr_time - self.last_kf_send_time > 1.0):
            kf_viz_data = self.get_kf_viz_data()

        # Send updated keyframe data after iteration
        if data is not None:
            if data[0] == "keyframe":
                kf_viz_data = self.get_kf_viz_data()

        # Send updated keyframe data
        if kf_updated:
            kf_ref_data = self.get_kf_ref_data()

        return kf_viz_data, kf_ref_data
    
    def map_sensor_depth(self, data):
        kf_viz_data = None
        kf_ref_data = None

        kf_updated = False

        # Handle incoming data
        if data is not None:
            data = transfer_data(data, self.device, self.dtype)

            if not self.is_init:
                if data[0] == "init_sensor_depth":
                    timestamp, rgb, depth = data[1:]
                    kf_updated = self.init_keyframe_from_sensor_depth(
                        timestamp, rgb, depth
                    )
            else:
                kf_viz_data, kf_updated = self.handle_tracking_data_sensor_depth(data)

        # No backend optimization here:
        # mapping is only used as keyframe manager + visualization + ref provider

        # Send updated mapping data if not sent for awhile
        curr_time = time.time()
        if self.is_init and (curr_time - self.last_kf_send_time > 1.0):
            kf_viz_data = self.get_kf_viz_data_sensor_depth()

        # Send updated keyframe data after new keyframe arrives
        if data is not None:
            if data[0] == "keyframe_sensor_depth":
                kf_viz_data = self.get_kf_viz_data_sensor_depth()

        # Send updated keyframe ref data
        if kf_updated:
            kf_ref_data = self.get_kf_ref_data_sensor_depth()

        return kf_viz_data, kf_ref_data
