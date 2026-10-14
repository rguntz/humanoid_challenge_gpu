#!/usr/bin/env python
"""The EGL device index (MUJOCO_EGL_DEVICE_ID) of a CUDA GPU.

EGL numbers the GPUs in its own order, unrelated to CUDA's or nvidia-smi's (on the 4 x H200
machine, EGL 0-3 are CUDA 3, 2, 0, 1), and it lists every GPU whatever CUDA_VISIBLE_DEVICES
says. So MUJOCO_EGL_DEVICE_ID=$GPU renders on another GPU than the one training uses. This
asks each EGL device which CUDA device it is (EGL_CUDA_DEVICE_NV, numbered as the caller's
CUDA_VISIBLE_DEVICES sees them) and prints the index of the matching one:
    CUDA_VISIBLE_DEVICES=1 python resfit/pick_place/egl_device.py      # EGL index of GPU 1
    python resfit/pick_place/egl_device.py 2                           # EGL index of CUDA device 2
"""
import ctypes
import os
import sys

EGL_CUDA_DEVICE_NV = 0x323A


def egl_device_for_cuda(cuda_index=0):
    """EGL index of CUDA device cuda_index (as CUDA_VISIBLE_DEVICES numbers them)."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mujoco.egl import egl_ext as EGL

    lib = ctypes.CDLL("libEGL.so.1")
    lib.eglGetProcAddress.restype = ctypes.c_void_p
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_long))(
        lib.eglGetProcAddress(b"eglQueryDeviceAttribEXT"))
    for i, device in enumerate(EGL.eglQueryDevicesEXT()):
        value = ctypes.c_long(-1)
        # Fails (returns 0) on devices that are not NVIDIA GPUs (Mesa, software rendering)
        if query(ctypes.cast(device, ctypes.c_void_p).value, EGL_CUDA_DEVICE_NV, ctypes.byref(value)) \
                and value.value == cuda_index:
            return i
    raise RuntimeError(f"no EGL device is CUDA device {cuda_index} "
                       f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")


if __name__ == "__main__":
    print(egl_device_for_cuda(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
