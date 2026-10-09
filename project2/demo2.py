import os
import numpy as np
import cv2
from project2_functions import render_object

def demo2_cam_rotating():
    """
    Renders a sequence of images simulating a car driving along a circular road,
    with a camera mounted on the car looking at a fixed target. The function loads geometry
    and scene parameters from 'hw2.npy', computes the car and camera positions
    for each frame, and renders the scene using the provided render_object function.
    The resulting images are saved to the 'demo2_frames' directory.
    """
    # Load geometry and scene parameters
    all_data = np.load("../assets/project2/hw2.npy", allow_pickle=True).item()

    v_pos = all_data["v_pos"].T
    v_uvs = all_data["v_uvs"]
    t_pos_idx = all_data["t_pos_idx"].astype(np.int32)
    v_clr = all_data.get("v_clr", np.ones_like(v_pos))

    C = all_data["k_road_center"].reshape(3)
    r = float(all_data["k_road_radius"])
    v = 5 * float(all_data["car_velocity"])
    omega = v / r

    up = all_data["k_cam_up"].reshape(3)
    k_cam_rel_pos = all_data["k_cam_car_rel_pos"].reshape(3)
    k_cam_target = all_data["k_cam_target"].reshape(3)

    plane_w = float(all_data["k_sensor_width"])
    plane_h = float(all_data["k_sensor_height"])
    focal = float(all_data["k_f"])
    duration = float(all_data["k_duration"])
    fps = float(all_data["k_fps"])
    n_frames = int(duration * fps)

    print (f"Demo2: Car velocity = {v} m/s, angular speed = {omega} rad/s")
    circle_fraction = duration * v / (2 * np.pi * r)
    print(f"Demo2: The car will complete {circle_fraction:.2f} of a full circle during the animation.")
    # Create output directory for frames
    os.makedirs("../output/project2/demo2_frames", exist_ok=True)

    for i in range(n_frames):
        t = i / fps
        theta = omega * t

        # Compute car position on the circular path
        car_pos = np.array(
            [C[0] + r * np.cos(theta), C[1], C[2] + r * np.sin(theta)],
            dtype=np.float64,
        )

        # Compute car's forward direction
        forward = np.array(
            [-np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64
        )
        forward /= np.linalg.norm(forward)

        # Compute car's right direction
        right_car = np.cross(forward, up)
        right_car /= np.linalg.norm(right_car)

        # Construct car's rotation matrix
        car_body_R = np.vstack([right_car, up, forward]).T

        # Compute camera position in world coordinates
        eye_cam = car_pos + (car_body_R @ k_cam_rel_pos)

        # Camera looks at the fixed target point
        target_cam = k_cam_target

        # Render the scene
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

        # Save the rendered frame
        cv2.imwrite(
            f"../output/project2/demo2_frames/frame_{i:03d}.png",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )
        print(f"Demo2: saved ../output/project2/demo2_frames/frame_{i:03d}.png")


if __name__ == "__main__":
    demo2_cam_rotating()
    print("Demo2: Finished rendering frames in ../output/project2/demo2_frames directory.")
