#!/usr/bin/env bash
# Builds native/build/sam2_ncnn.dll (Release, x64, /MD) with the ncnn engine
# statically linked - the only runtime artifacts are the DLL itself plus the
# ncnn model files. Run it from Git Bash; it shells out to cmd so that
# vcvars64.bat can set up the MSVC environment.
#
#   ./native/build.sh                       # CPU backend, log -> native/build.log
#   VULKAN=1 ./native/build.sh              # Vulkan backend (needs Vulkan SDK)
#   NCNN_ROOT=/path/to/ncnn-install ./native/build.sh
#
# Overrides: VCVARS, CMAKE, NINJA, VULKAN (0/1), NCNN_ROOT (prebuilt ncnn
# install prefix; empty = fetch+build from source, see native/CMakeLists.txt).
#
# Tool discovery order: $VCVARS / $CMAKE / $NINJA, then the Visual Studio
# installation reported by vswhere, then whatever is on PATH. When vcvars64.bat
# is not found the script assumes cl.exe is already usable (GitHub Actions
# after ilammy/msvc-dev-cmd).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"
LOG="$ROOT/build.log"
VULKAN="${VULKAN:-0}"
case "$VULKAN" in
  1|ON|on) VULKAN="ON" ;;
  *)       VULKAN="OFF" ;;
esac

find_vs() {
  local vswhere='/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe'
  local p=""
  if [ -f "$vswhere" ]; then
    p="$("$vswhere" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
         -property installationPath 2>/dev/null | tr -d '\r' | head -1 || true)"
  fi
  if [ -z "$p" ]; then
    local edition
    for edition in Community Professional Enterprise BuildTools; do
      p="/c/Program Files/Microsoft Visual Studio/2022/$edition"
      [ -d "$p" ] && break
      p=""
    done
  fi
  [ -n "$p" ] || return 1
  printf '%s' "$p"
}

VS=""
if [ -z "${VCVARS:-}" ] || [ -z "${CMAKE:-}" ] || [ -z "${NINJA:-}" ]; then
  VS="$(find_vs || true)"
fi

if [ -z "${VCVARS:-}" ]; then
  if [ -n "$VS" ] && [ -f "$VS/VC/Auxiliary/Build/vcvars64.bat" ]; then
    VCVARS="$VS/VC/Auxiliary/Build/vcvars64.bat"
  else
    VCVARS=""
  fi
fi
if [ -z "${CMAKE:-}" ]; then
  if [ -n "$VS" ] && [ -f "$VS/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" ]; then
    CMAKE="$VS/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
  else
    CMAKE="$(command -v cmake || true)"
  fi
fi
if [ -z "${NINJA:-}" ]; then
  if [ -n "$VS" ] && [ -f "$VS/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe" ]; then
    NINJA="$VS/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
  else
    NINJA="$(command -v ninja || command -v ninja-build || true)"
  fi
fi

[ -n "$CMAKE" ] || { echo "cmake not found; set CMAKE=/path/to/cmake" >&2; exit 1; }
[ -n "$NINJA" ] || { echo "ninja not found; set NINJA=/path/to/ninja" >&2; exit 1; }
if [ -z "$VCVARS" ] && ! command -v cl >/dev/null 2>&1; then
  echo "no vcvars64.bat found and cl.exe is not on PATH; set VCVARS=/path/to/vcvars64.bat" >&2
  exit 1
fi

rm -rf "$BUILD"
mkdir -p "$BUILD"

# cmd.exe cannot be handed a quoted one-liner from Git Bash without mangling the
# quotes, so the vcvars + cmake + ninja sequence goes into a temporary .cmd file.
WIN_ROOT="$(cygpath -w "$ROOT")"
WIN_BUILD="$(cygpath -w "$BUILD")"
WIN_NCNN_ROOT=""
if [ -n "${NCNN_ROOT:-}" ]; then
  NCNN_ROOT="$(cd "$NCNN_ROOT" && pwd)"
  WIN_NCNN_ROOT="$(cygpath -w "$NCNN_ROOT")"
fi
BAT="$BUILD/_build.cmd"
{
  printf '@echo off\r\n'
  printf 'chcp 65001 >nul\r\n'
  if [ -n "$VCVARS" ]; then
    printf 'call "%s" >nul || exit /b 1\r\n' "$VCVARS"
  fi
  printf '"%s" -S "%s" -B "%s" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSAM2_VULKAN="%s" -DNCNN_ROOT="%s" -DCMAKE_MAKE_PROGRAM="%s" || exit /b 1\r\n' \
    "$CMAKE" "$WIN_ROOT" "$WIN_BUILD" "$VULKAN" "$WIN_NCNN_ROOT" "$NINJA"
  printf '"%s" -C "%s" || exit /b 1\r\n' "$NINJA" "$WIN_BUILD"
} > "$BAT"

echo "== backend: vulkan=${VULKAN}  ncnn_root: ${NCNN_ROOT:-<fetch from source>}"
echo "== vcvars: ${VCVARS:-<using the current environment>}"
echo "== configure + build (log: $LOG)"
set +e
(cd "$BUILD" && cmd //c _build.cmd) 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
set -e
if [ "$rc" -ne 0 ]; then
  echo "BUILD FAILED (exit $rc); see $LOG" >&2
  exit "$rc"
fi

echo "== staged artifact"
ls -la "$BUILD"/sam2_ncnn.dll
