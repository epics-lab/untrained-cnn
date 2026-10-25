"""Description of the machine and software a timing ran on."""

import os
import platform

import numpy as np
import sklearn
import torch


def system_info():
    info = {
        "cluster": os.environ.get("SLURM_CLUSTER_NAME", ""),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "node": platform.node(),
        "cpu": "",
        "os": "",
        "kernel": platform.release(),
        "arch": platform.machine(),
        "job_cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "job_mem_mb": os.environ.get("SLURM_MEM_PER_NODE") or os.environ.get("SLURM_MEM_PER_CPU", ""),
        "node_ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    try:
        import monai
        info["monai"] = monai.__version__
    except ImportError:
        pass
    try:
        info["os"] = platform.freedesktop_os_release().get("PRETTY_NAME", "")
    except (OSError, AttributeError):
        pass
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return info
