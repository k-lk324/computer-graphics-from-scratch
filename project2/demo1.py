import os
import numpy as np
import cv2
from project2_functions import render_object


def demo1_cam_static():
    """
    Renders a sequence of images simulating a car driving along a circular road,
    with a camera mounted on the car looking forward. The function loads geometry
    and scene parameters from 'hw2.npy', computes the car and camera positions
    for each frame, and renders the scene using the provided render_object function.
    The resulting images are saved to the 'demo1_frames' directory.
    """
    # Load all data from file
    all_data = np.load("../assets/project2/hw2.npy", allow_pickle=True).item()

    # Unpack mesh data: vertices, UVs, faces, and colors
    v_pos = all_data["v_pos"].T  # (16x3) vertex positions
    v_uvs = all_data["v_uvs"]  # (16x2) vertex UV coordinates
    t_pos_idx = all_data["t_pos_idx"].astype(
        np.int32
    )  # (6x3) triangle indices
    # (16x3) vertex colors, default to white if missing
    # In this case, we don't need it because we are using a texture
    # but we keep it for compatibility with the render_object function
    v_clr = all_data.get("v_clr", np.ones_like(v_pos))

    # Car and camera parameters
    C = all_data["k_road_center"].reshape(3)  # (3,) road center
    r = float(all_data["k_road_radius"])  # road radius
    v = float(all_data["car_velocity"])  # car velocity
    omega = v / r  # angular speed
    up = all_data["k_cam_up"].reshape(3)  # camera up vector
    k_cam_rel_pos = all_data["k_cam_car_rel_pos"].reshape(
        3
    )  # camera position relative to car

    # Sensor, focal length, and timing parameters
    plane_w = float(all_data["k_sensor_width"])  # sensor width
    plane_h = float(all_data["k_sensor_height"])  # sensor height
    focal = float(all_data["k_f"])  # focal length
    duration = float(all_data["k_duration"])  # animation duration (seconds)
    fps = float(all_data["k_fps"])  # frames per second
    n_frames = int(duration * fps)  # total number of frames

    # Create output directory for frames
    os.makedirs("../output/project2/demo1_frames", exist_ok=True)

    for i in range(n_frames):
        t = i / fps
        theta = omega * t

        # Compute car position on XZ-plane circle
        car_pos = np.array(
            [C[0] + r * np.cos(theta), C[1], C[2] + r * np.sin(theta)],
            dtype=np.float64,
        )

        # Compute car's forward direction (tangent to circle)
        forward = np.array(
            [-np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64
        )
        forward /= np.linalg.norm(forward)

        # Compute car's right direction (cross product of forward and up)
        right_car = np.cross(forward, up)
        right_car /= np.linalg.norm(right_car)

        # Car-body rotation matrix (columns: right, up, forward)
        car_body_R = np.vstack([right_car, up, forward]).T  # (3x3)

        # Compute camera position in world coordinates
        eye_cam = car_pos + (car_body_R @ k_cam_rel_pos)

        # Compute camera look-at target (look forward)
        target_cam = eye_cam + forward

        # Render the scene (image resolution is 512x512)
        img = render_object(
            v_pos=v_pos,
            v_clr=v_clr,
            t_pos_idx=t_pos_idx,
            v_uvs=v_uvs,
            plane_h=plane_h,
            plane_w=plane_w,
            res_h=512,
            res_w=512,
            focal=focal,
            eye=eye_cam,
            up=up,
            target=target_cam,
            diffuse_path="../assets/project2/stone-72_diffuse.jpg",
        )

        # Save the rendered frame as PNG
        cv2.imwrite(
            f"../output/project2/demo1_frames/frame_{i:03d}.png",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )
        print(f"Demo1: saved ../output/project2/demo1_frames/frame_{i:03d}.png")


if __name__ == "__main__":
    demo1_cam_static()
    print("Demo1: Finished rendering frames.")
    
