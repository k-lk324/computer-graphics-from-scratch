import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from project2.triangle_fill import render_img


def main():
    data = np.load("../assets/project1/hw1.npy", allow_pickle=True).item()
    texImg = cv2.imread("../assets/project1/texImg.jpg")

    if texImg is None:
        raise FileNotFoundError(
            "The texture image '../assets/project1/texImg.jpg' could not be loaded."
        )

    img = render_img(
        faces=data["t_pos_idx"],
        vertices=data["v_pos2d"],
        vcolors=data["v_clr"],
        uvs=data["v_uvs"],
        depth=data["depth"],
        shading="t",
        textImg=texImg,
    )
    # cv2.imshow("Textured Image", (img).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    os.makedirs("../output/project1", exist_ok=True)
    cv2.imwrite("../output/project1/output_texture.jpg", (img).astype(np.uint8))


if __name__ == "__main__":
    main()
