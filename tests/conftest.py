import os
import sys
import types
import numpy as np

# Add project subdirectories to sys.path
root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub_dir in ["project1", "project2", "project3"]:
    dir_path: str = os.path.join(root_dir, sub_dir)
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

# Lightweight shim for assets pickled with trimesh.caching.TrackedArray
if "trimesh" not in sys.modules:
    trimesh_module = types.ModuleType("trimesh")
    caching_module = types.ModuleType("trimesh.caching")

    class TrackedArray(np.ndarray):
        pass

    caching_module.TrackedArray = TrackedArray
    trimesh_module.caching = caching_module
    sys.modules["trimesh"] = trimesh_module
    sys.modules["trimesh.caching"] = caching_module
