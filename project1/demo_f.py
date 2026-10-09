import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from project2.triangle_fill import render_img

def main():
    data = np.load("../assets/project1/hw1.npy", allow_pickle=True).item()
    img = render_img(
        faces=data['t_pos_idx'],
        vertices=data['v_pos2d'],
        vcolors=data['v_clr'],
        uvs=data['v_uvs'],
        depth=data['depth'],
        shading='f'
    )
    # Convert BGR to RGB for display
    # Not sure what should be displayed in this case
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    os.makedirs("../output/project1", exist_ok=True)
    cv2.imwrite("../output/project1/output_flat.jpg", (img).astype(np.uint8))


if __name__ == "__main__":
    main()
