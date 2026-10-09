# demo.py

import numpy as np
from PIL import Image
import cv2
from project3_utils import MatPhong, render_object

def save_img(arr, name):
    cv2.imwrite(name, (arr).astype(np.uint8))

def main():
    tex_im  = np.array(Image.open("../assets/project3/Mona-Lisa-Exist-in-Real-Life-2635825581.jpg"))  # HxWx3
    data = np.load("../assets/project3/hw3.npy", allow_pickle=True).item()
    up       = data["up"]
    plane_h  = data["plane_h"]
    plane_w  = data["plane_w"]
    res_h    = data["res_h"]
    res_w    = data["res_w"]
    focal    = data["focal"]
    verts    = data["v_pos"]
    uvs      = data["v_uvs"]
    faces    = data["t_pos_idx"]
    target   = data["target"]
    eye      = data["cam_pos"]
    l_pos    = data["l_pos"]
    l_int    = data["l_int"]
    l_amb    = data["l_amb"]
    ka       = data["ka"]
    kd       = data["kd"]
    ks       = data["ks"]
    n        = data["n"]

    l_pos = np.array(data["l_pos"])
    l_int = np.array(data["l_int"])
    l_amb = np.array(data["l_amb"])


    comp_masks = {
        "ambient":   (l_pos, l_int, l_amb),
        "diffuse":   (l_pos,    l_int,   l_amb),
        "specular":  (l_pos,    l_int,   l_amb),
        "combined":  (l_pos,    l_int,   l_amb),
    }

    components = {
        "ambient": dict(ka=ka, kd=0.0, ks=0.0),
        "diffuse": dict(ka=0.0, kd=kd, ks=0.0),
        "specular": dict(ka=0.0, kd=0.0, ks=ks),
        "combined": dict(ka=ka, kd=kd, ks=ks),
    }

    import os
    os.makedirs("../output/project3", exist_ok=True)
    for shader in ["gouraud", "phong"]:
        for comp_name, (lpos, lint, lamb) in comp_masks.items():
            mat_vals = components[comp_name]
            mat = MatPhong(ka=mat_vals["ka"], kd=mat_vals["kd"], ks=mat_vals["ks"], n=n)

            img = render_object(
                verts, uvs, faces, tex_im,
                plane_h, plane_w, res_h, res_w,
                focal, eye, up, target,
                mat, lpos, lint, lamb, shader
            )
            outfile = f"../output/project3/{shader}_{comp_name}.png"
            save_img(img*255, outfile)
            print(f"Saved {outfile}")

if __name__ == "__main__":
    main()
