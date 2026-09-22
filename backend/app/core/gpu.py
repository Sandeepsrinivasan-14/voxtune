from dataclasses import asdict, dataclass


@dataclass
class GpuInfo:
    available: bool
    name: str = "CPU"
    total_gb: float = 0.0
    free_gb: float = 0.0
    utilization_pct: int = 0
    driver: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def gpus_live() -> list[dict]:
    """Per-device live stats straight from NVML; any counter the driver doesn't expose is None."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return []

    def opt(fn, scale=1.0):
        try:
            return round(fn() / scale, 1)
        except Exception:
            return None

    out = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append({
                "index": i,
                "name": name.decode() if isinstance(name, bytes) else name,
                "util_pct": opt(lambda: pynvml.nvmlDeviceGetUtilizationRates(h).gpu),
                "mem_used_gb": round(mem.used / 1024**3, 2),
                "mem_total_gb": round(mem.total / 1024**3, 2),
                "temp_c": opt(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
                "power_w": opt(lambda: pynvml.nvmlDeviceGetPowerUsage(h), 1000),
                "power_limit_w": opt(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h), 1000),
                "clock_mhz": opt(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)),
            })
    finally:
        pynvml.nvmlShutdown()
    return out


def detect_gpu() -> GpuInfo:
    """Real hardware detection via NVML, falling back to torch."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            driver = pynvml.nvmlSystemGetDriverVersion()
            return GpuInfo(
                available=True,
                name=name.decode() if isinstance(name, bytes) else name,
                total_gb=round(mem.total / 1024**3, 2),
                free_gb=round(mem.free / 1024**3, 2),
                utilization_pct=int(util.gpu),
                driver=driver.decode() if isinstance(driver, bytes) else driver,
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return GpuInfo(True, torch.cuda.get_device_name(0), round(total / 1024**3, 2), round(free / 1024**3, 2))
    except Exception:
        pass
    return GpuInfo(available=False)
