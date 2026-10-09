# Computer Graphics Projects

This repository contains three Computer Graphics projects implemented in Python. These projects cover fundamental concepts such as 2D/3D transformations, perspective projections, rasterization, and advanced shading techniques (Flat, Gouraud, and Phong shading).

## Project Structure

The codebase is organized into `project1/`, `project2/`, and `project3/` directories containing the core source code. Inputs such as geometries (`.npy` files) and textures are stored in `assets/`, while output images and frames are saved to `output/`.

*   **`project1/`**: Implements 2D rasterization of 3D objects projected onto a camera plane. It supports both flat shading and texture mapping using barycentric coordinates.
*   **`project2/`**: Simulates 3D transformations, world-to-camera coordinate conversions, and perspective projection. The demos involve rendering a moving car on a circular road with either a static or rotating camera.
*   **`project3/`**: Focuses on advanced lighting models and shading techniques. It implements Gouraud shading (interpolating vertex colors based on normals) and Phong shading (interpolating normals per pixel).

## Requirements

The projects rely on the following Python packages. You can install them via `pip`:

```bash
pip install numpy opencv-python Pillow trimesh
```

## Running the Demos

Each project includes demo scripts that showcase its functionality. The generated images/frames will be saved inside the respective `output/projectX/` directories.

### Project 1: Rasterization & Shading
Renders a 3D object using flat shading and texture mapping.
```bash
cd project1
python3 demo_f.py  # Renders the object using flat shading (saves output_flat.jpg)
python3 demo_t.py  # Renders the object using texture shading (saves output_texture.jpg)
```

### Project 2: 3D Transformations & Camera Simulation
Simulates a camera mounted on a car moving along a circular path.
```bash
cd project2
python3 demo1.py  # Car moves, camera is static facing forward (generates frames in output/project2/demo1_frames)
python3 demo2.py  # Car moves, camera rotates to keep looking at a specific target (generates frames in output/project2/demo2_frames)
```

### Project 3: Illumination & Advanced Shading
Renders an object using various lighting components (Ambient, Diffuse, Specular, Combined) for both Gouraud and Phong shading models.
```bash
cd project3
python3 demo.py   # Generates 8 images combining Gouraud/Phong shading with different lighting models.
```
