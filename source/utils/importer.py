"""
Library open3d is a bit finicky with imports.
One can either import from cuda (for GPU-enables machines) or cpu (for others).
By checking the config, based on the "device" attribute, we can decide which one to import.
For use simply import from utils.importer, i.e. "from utils.importer import Pointcloud".
"""

import open3d

from utils.recursive_config import Config

_conf = Config()
if _conf["device"] == "cuda" and hasattr(open3d, "cuda"):
    from open3d.cuda.pybind.geometry import (
        AxisAlignedBoundingBox,
        PointCloud,
        TriangleMesh,
    )
    from open3d.cuda.pybind.utility import Vector3dVector
else:
    # The standard PyPI Open3D wheel is CPU-only even when PyTorch uses CUDA.
    from open3d.geometry import (
        AxisAlignedBoundingBox,
        PointCloud,
        TriangleMesh,
    )
    from open3d.utility import Vector3dVector
