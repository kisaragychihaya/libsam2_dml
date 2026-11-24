"""Downloads and extracts ONNX Runtime native packages for one or more execution providers.

Only needs the standard library (urllib + zipfile). Downloaded .nupkg files are cached in
native/third_party/nupkg/ (shared across EPs); each EP extracts into its own directory:

    native/third_party/onnxruntime-directml/   Microsoft.ML.OnnxRuntime.DirectML + Microsoft.AI.DirectML
    native/third_party/onnxruntime-gpu/        Microsoft.ML.OnnxRuntime.Gpu.Windows (CUDA + TensorRT
                                               provider plugins; headers from the base package)
    native/third_party/onnxruntime-cpu/        Microsoft.ML.OnnxRuntime (CPU only)

Per EP directory layout:

    include/*.h        ONNX Runtime headers
    lib/*.lib          import libraries
    bin/*.dll          runtime DLLs shipped by the package
    (nupkg cache one level up, in third_party/nupkg/)

The CUDA and TensorRT provider plugins still require NVIDIA's own libraries on the target
machine (CUDA 12.x runtime + cuDNN 9.x for CUDA, plus TensorRT for TensorRT) - those DLLs
are not redistributable through the ORT NuGet packages.

Usage (any Python 3.8+):

    python native/third_party/fetch_ort.py                      # directml (default)
    python native/third_party/fetch_ort.py cuda                 # one EP
    python native/third_party/fetch_ort.py directml cuda cpu    # several at once
"""

import argparse
import os
import urllib.request
import zipfile

ORT_VERSION = "1.24.4"
DML_VERSION = "1.15.4"  # dependency declared by the ORT DirectML nuspec

HERE = os.path.dirname(os.path.abspath(__file__))
NUPKG_DIR = os.path.join(HERE, "nupkg")

# ep key -> (output dir suffix, [(nuget id lowercased, version, extraction mode), ...])
# mode "ort": headers (build/native/include/*.h), runtimes/win-x64/native/*.dll -> bin/, *.lib -> lib/
# mode "dml": bin/x64-win/DirectML.dll + DirectML.lib only
EP_PKGS = {
    "directml": ("directml", [
        (f"microsoft.ml.onnxruntime.directml/{ORT_VERSION}/"
         f"microsoft.ml.onnxruntime.directml.{ORT_VERSION}.nupkg", "ort"),
        (f"microsoft.ai.directml/{DML_VERSION}/"
         f"microsoft.ai.directml.{DML_VERSION}.nupkg", "dml"),
    ]),
    "cuda": ("gpu", [
        (f"microsoft.ml.onnxruntime/{ORT_VERSION}/"
         f"microsoft.ml.onnxruntime.{ORT_VERSION}.nupkg", "ort_headers"),
        (f"microsoft.ml.onnxruntime.gpu.windows/{ORT_VERSION}/"
         f"microsoft.ml.onnxruntime.gpu.windows.{ORT_VERSION}.nupkg", "ort"),
    ]),
    "tensorrt": ("gpu", [
        (f"microsoft.ml.onnxruntime/{ORT_VERSION}/"
         f"microsoft.ml.onnxruntime.{ORT_VERSION}.nupkg", "ort_headers"),
        (f"microsoft.ml.onnxruntime.gpu.windows/{ORT_VERSION}/"
         f"microsoft.ml.onnxruntime.gpu.windows.{ORT_VERSION}.nupkg", "ort"),
    ]),
    "cpu": ("cpu", [
        (f"microsoft.ml.onnxruntime/{ORT_VERSION}/"
         f"microsoft.ml.onnxruntime.{ORT_VERSION}.nupkg", "ort"),
    ]),
}


def fetch(path, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"cached  {os.path.basename(dest)}")
        return
    url = "https://api.nuget.org/v3-flatcontainer/" + path
    print(f"fetch   {url}")
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        f.write(r.read())


def extract_ort(path, base, headers=True, dlls=True):
    z = zipfile.ZipFile(path)
    with z:
        for name in z.namelist():
            if headers and name.startswith("build/native/include/") and name.endswith(".h"):
                os.makedirs(os.path.join(base, "include"), exist_ok=True)
                with open(os.path.join(base, "include", os.path.basename(name)), "wb") as f:
                    f.write(z.read(name))
            elif dlls and name.startswith("runtimes/win-x64/native/") and name.endswith(".dll"):
                os.makedirs(os.path.join(base, "bin"), exist_ok=True)
                with open(os.path.join(base, "bin", os.path.basename(name)), "wb") as f:
                    f.write(z.read(name))
            elif name == "runtimes/win-x64/native/onnxruntime.lib":
                os.makedirs(os.path.join(base, "lib"), exist_ok=True)
                with open(os.path.join(base, "lib", "onnxruntime.lib"), "wb") as f:
                    f.write(z.read(name))


def extract_dml(path, base):
    z = zipfile.ZipFile(path)
    with z:
        os.makedirs(os.path.join(base, "bin"), exist_ok=True)
        os.makedirs(os.path.join(base, "lib"), exist_ok=True)
        with open(os.path.join(base, "bin", "DirectML.dll"), "wb") as f:
            f.write(z.read("bin/x64-win/DirectML.dll"))
        with open(os.path.join(base, "lib", "DirectML.lib"), "wb") as f:
            f.write(z.read("bin/x64-win/DirectML.lib"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eps", nargs="*", default=["directml"],
                    help="execution provider(s) to fetch: " + ", ".join(EP_PKGS))
    args = ap.parse_args()

    for ep in args.eps:
        if ep not in EP_PKGS:
            raise SystemExit(f"unknown EP '{ep}' - choose from: {', '.join(EP_PKGS)}")
        suffix, pkgs = EP_PKGS[ep]
        base = os.path.join(HERE, f"onnxruntime-{suffix}")
        os.makedirs(NUPKG_DIR, exist_ok=True)
        print(f"== {ep} -> {base}")
        for path, mode in pkgs:
            dest = os.path.join(NUPKG_DIR, os.path.basename(path))
            fetch(path, dest)
            if mode == "dml":
                extract_dml(dest, base)
            elif mode == "ort_headers":
                # Base package contributes headers only; its CPU onnxruntime.dll
                # must not shadow the Gpu.Windows one in the shared gpu dir.
                extract_ort(dest, base, headers=True, dlls=False)
            else:
                extract_ort(dest, base)
        print(f"done -> {base}")
        if suffix == "gpu":
            print("note: CUDA/TensorRT additionally need NVIDIA CUDA 12.x + cuDNN 9.x "
                  "(+ TensorRT) DLLs on the target machine, on PATH or next to the exe.")


if __name__ == "__main__":
    main()
