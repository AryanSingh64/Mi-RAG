import os
import platform
import subprocess
from typing import Any, Dict


class HardwareDetector:
    """
    Real-time hardware specification and GPU VRAM detector for Windows and Linux.
    Determines model compatibility based on detected GPU memory and RAM.
    """

    @staticmethod
    def get_specs() -> Dict[str, Any]:
        specs = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor() or "CPU",
            "cores": os.cpu_count() or 4,
            "ram_gb": 8.0,
            "ram_avail_gb": 4.0,
            "gpu": "Integrated / CPU",
            "vram_gb": 0.0,
            "has_nvidia": False,
            "performance_tier": "entry"
        }

        # 1. Detect RAM via Windows memory status
        if platform.system() == "Windows":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                specs["ram_gb"] = round(stat.ullTotalPhys / (1024**3), 1)
                specs["ram_avail_gb"] = round(stat.ullAvailPhys / (1024**3), 1)
            except Exception:
                pass

        # 2. Detect NVIDIA GPU & VRAM via nvidia-smi
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            lines = [l.strip() for l in smi.strip().split("\n") if l.strip()]
            if lines:
                parts = lines[0].split(",")
                specs["gpu"] = parts[0].strip()
                specs["vram_gb"] = round(float(parts[1].strip()) / 1024, 1)
                specs["has_nvidia"] = True
        except Exception:
            pass

        # 3. Calculate Performance Tier
        vram = specs["vram_gb"]
        if vram >= 12.0:
            specs["performance_tier"] = "ultra"
        elif vram >= 6.0:
            specs["performance_tier"] = "high"
        elif vram >= 3.5:
            specs["performance_tier"] = "recommended_4gb"
        else:
            specs["performance_tier"] = "cpu_entry"

        return specs

    @classmethod
    def evaluate_model_compatibility(cls, required_vram_gb: float) -> Dict[str, Any]:
        """
        Evaluates whether a model fits in local GPU memory.
        """
        specs = cls.get_specs()
        vram = specs["vram_gb"]
        ram = specs["ram_gb"]

        if vram >= required_vram_gb:
            return {
                "status": "recommended",
                "badge": "⚡ PERFECT MATCH",
                "color": "#2be26c",
                "note": f"Fits fully in {vram} GB VRAM for RTX acceleration"
            }
        elif (vram + 2.0) >= required_vram_gb:
            return {
                "status": "compatible",
                "badge": "✓ COMPATIBLE",
                "color": "#ffe814",
                "note": f"Runs with partial GPU offload ({vram} GB VRAM)"
            }
        elif ram >= (required_vram_gb * 1.5):
            return {
                "status": "heavy",
                "badge": "🟡 HEAVY (RAM Offload)",
                "color": "#f97316",
                "note": f"Exceeds {vram} GB VRAM. Will run on CPU RAM"
            }
        else:
            return {
                "status": "unsupported",
                "badge": "⚠️ REQUIRES 16GB+ VRAM",
                "color": "#ef4444",
                "note": f"Exceeds system memory ({vram}GB VRAM / {ram}GB RAM)"
            }
