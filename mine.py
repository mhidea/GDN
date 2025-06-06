import torch


import torch
import warnings

gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True

if not gpu_ok:
    warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
        "than expected."
    )


print(torch.__version__)
# torch.set_default_device("cuda")
# torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.device_count())
print(device)


print(torch.version.cuda)
