import os
import sys

# Add project subdirectories to sys.path
root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub_dir in ["project1", "project2", "project3"]:
    dir_path: str = os.path.join(root_dir, sub_dir)
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)
