#!/bin/bash

# Fix git safe directory issues for Docker environments
git config --global --add safe.directory '*'

cd sgl-kernel

# # # # Install ccache (skip the non-existent NVTX package)
# apt-get update && apt-get install -y ccache

# # # Install Python packages
# pip install numpy clang-format lm_eval

# # Setup ccache environment variables as per README.md
# export CCACHE_DIR="/root/.cache/huggingface/sglang_cache"
# export CCACHE_BACKEND=""
# export CCACHE_KEEP_LOCAL_STORAGE="TRUE"
# unset CCACHE_READONLY

# # Create ccache directory if it doesn't exist
# mkdir -p "$CCACHE_DIR"

# # Clean previous builds to ensure we rebuild against current CUDA
# echo "Cleaning previous builds..."
# rm -rf build dist *.egg-info

# # Create stub NVTX library if not present 
# if [ ! -f /usr/local/cuda/lib64/libnvToolsExt.so ]; then
#   echo "Creating stub libnvToolsExt.so for compatibility..."
#   apt-get update && apt-get install -y gcc make
#   echo 'void nvtxRangePushA(const char*){} void nvtxRangePop(){}' | \
#     gcc -shared -fPIC -x c - -o /usr/local/cuda/lib64/libnvToolsExt.so
# fi

# # Set cmake variables to handle NVTX issues
# export CMAKE_ARGS="-DUSE_NVTX=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=.a;.so"

# # Build sgl-kernel with ccache and CMake overrides (pass definitions directly)
# python -m uv build --wheel -Cbuild-dir=build --color=always .


# # Install the built wheel
# pip install dist/*.whl --force-reinstall

cd ..

# Install SGLang in development mode                                                                                                                                                                                                                                                     
pip install -e python -vvv 

# Update flashinfer with TRTLLM support
bash zscripts/update_flashinfer_trtllm.sh


export SGL_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_TORCH_COMPILE=0


python - <<'PY'
import sglang, sgl_kernel, importlib, subprocess, sys, os
print("sglang path ->", sglang.__file__)
print("sgl_kernel path ->", sgl_kernel.__file__)
print("Done verifying install. Have a nice day!")
PY

pip install -U nvidia-cudnn-cu12
