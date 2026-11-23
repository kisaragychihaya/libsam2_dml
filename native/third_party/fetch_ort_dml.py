"""Downloads and extracts the ONNX Runtime DirectML native package.

Only needs the standard library (urllib + zipfile). Writes into
native/third_party/onnxruntime/:

    include/*.h        ONNX Runtime + DirectML provider headers
    lib/onnxruntime.lib, lib/DirectML.lib
    bin/onnxruntime.dll, bin/DirectML.dll, bin/onnxruntime_providers_shared.dll
    nupkg/*.nupkg      the downloaded packages (for provenance)

Usage (any Python 3.8+):
    python native/third_party/fetch_ort_dml.py
"""

import io
import os
import urllib.request
import zipfile

ORT_VERSION = "1.24.4"
DML_VERSION = "1.15.4"  # dependency declared by the ORT DirectML nuspec

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "onnxruntime")

PKGS = [
    (f"https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/"
     f"{ORT_VERSION}/microsoft.ml.onnxruntime.directml.{ORT_VERSION}.nupkg",
     f"ort_directml_{ORT_VERSION}.nupkg"),
    (f"https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/"
     f"{DML_VERSION}/microsoft.ai.directml.{DML_VERSION}.nupkg",
     f"directml_{DML_VERSION}.nupkg"),
]


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"cached  {os.path.basename(dest)}")
        return
    print(f"fetch   {url}")
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        f.write(r.read())


def extract_ort(path):
    z = zipfile.ZipFile(path)
    with z:
        os.makedirs(os.path.join(BASE, "include"), exist_ok=True)
        os.makedirs(os.path.join(BASE, "lib"), exist_ok=True)
        os.makedirs(os.path.join(BASE, "bin"), exist_ok=True)
        for name in z.namelist():
            if name.startswith("build/native/include/") and name.endswith(".h"):
                target = os.path.join(BASE, "include", os.path.basename(name))
                with open(target, "wb") as f:
                    f.write(z.read(name))
            elif name == "runtimes/win-x64/native/onnxruntime.lib":
                with open(os.path.join(BASE, "lib", "onnxruntime.lib"), "wb") as f:
                    f.write(z.read(name))
            elif name.startswith("runtimes/win-x64/native/") and name.endswith(".dll"):
                with open(os.path.join(BASE, "bin", os.path.basename(name)), "wb") as f:
                    f.write(z.read(name))


def extract_dml(path):
    z = zipfile.ZipFile(path)
    with z:
        with open(os.path.join(BASE, "bin", "DirectML.dll"), "wb") as f:
            f.write(z.read("bin/x64-win/DirectML.dll"))
        with open(os.path.join(BASE, "lib", "DirectML.lib"), "wb") as f:
            f.write(z.read("bin/x64-win/DirectML.lib"))


def main():
    os.makedirs(os.path.join(BASE, "nupkg"), exist_ok=True)
    paths = []
    for url, name in PKGS:
        dest = os.path.join(BASE, "nupkg", name)
        fetch(url, dest)
        paths.append(dest)
    extract_ort(paths[0])
    extract_dml(paths[1])
    print(f"done -> {BASE}")


if __name__ == "__main__":
    main()
