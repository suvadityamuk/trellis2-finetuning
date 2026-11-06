# =============================================================================
# TRELLIS.2 fine-tuning image (published as cr.hf.co/suvadityamuk/trellis2-finetuning,
# mirrored at docker.io/hfsuvadityamuk/trellis2-starter).
#
# Everything TRELLIS.2 needs to train is compiled in: flash-attn, FlexGEMM, CuMesh,
# nvdiffrast, nvdiffrec, o-voxel, both Blenders, PEFT / Accelerate / Trackio.
# Weights and data are NOT baked in: they download at run time (DINOv3 is gated).
#
# The TRELLIS.2 tree is upstream microsoft/TRELLIS.2 at a pinned commit, plus
#   patches/trellis2-75fbf01.patch   fixes to six upstream files (DINOv3 on
#                                    transformers 5.x, GPU Cycles, no sudo, worker caps)
#   trellis2/                        our files, copied over the tree at the same paths
# and the two one-script entry points (preprocess.py, finetune.py) in /workspace.
#
# Build from the release/ folder (the build context must contain container/ and the
# two scripts):
#   docker build -f container/Dockerfile -t trellis2-finetuning .
#
# The base image already ships the CUDA 12.4 toolkit (nvcc) + cuDNN, and the
# torch cu124 wheels bring the matching NVIDIA runtime. We deliberately do NOT
# install any CUDA toolkit via conda; CUDA_HOME points at the system toolkit.
# =============================================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# Compile our CUDA extensions for every GPU family a reader is likely to have:
# Ampere (A100 = 8.0; A10 / A40 / RTX 30xx = 8.6), Ada (L4 / L40S / RTX 40xx = 8.9)
# and Hopper (H100 = 9.0). Override at build time to trim, e.g. --build-arg
# TORCH_CUDA_ARCH_LIST="8.0".
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
# Only used if the prebuilt flash-attn wheel below is unavailable and it falls back
# to a source build (sm80 kernels also run on 8.6 / 8.9).
ENV FLASH_ATTN_CUDA_ARCHS="80;90"
# Cap parallel nvcc so the FlexGEMM / CuMesh / o-voxel compiles fit RAM. cpu-xl has
# 124 GB, so 6 concurrent units (~10 GB each) is safe; raise on a bigger host.
ENV MAX_JOBS=6

WORKDIR /workspace

# --- System dependencies (root, build time) ---------------------------------
# GL/X libs + xvfb for headless Blender, libjpeg-dev for pillow-simd.
COPY container/packages.txt packages.txt
RUN apt-get update && xargs -r -a packages.txt apt-get install -y && rm -rf /var/lib/apt/lists/*

# --- TRELLIS.2 source: pinned upstream + our patch + our files -----------------
ARG TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
RUN git clone https://github.com/microsoft/TRELLIS.2.git /workspace/TRELLIS.2 && \
    git -C /workspace/TRELLIS.2 checkout ${TRELLIS2_COMMIT}
WORKDIR /workspace/TRELLIS.2
COPY container/patches/trellis2-75fbf01.patch /tmp/trellis2.patch
RUN git apply /tmp/trellis2.patch && rm /tmp/trellis2.patch
COPY container/trellis2/ /workspace/TRELLIS.2/

# --- PyTorch first (cu124), so --no-build-isolation builds can resolve torch -
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# --- Basic dependencies (mirrors setup.sh --basic) --------------------------
# transformers must be 5.x: the DINOv3 feature extractor calls the encoder nested at
# DINOv3ViTModel.model, which 4.x does not have.
RUN pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers==5.18.0 gradio==6.0.1 tensorboard pandas lpips zstandard
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
RUN pip install pillow-simd
RUN pip install kornia timm

# --- Fine-tuning dependencies -----------------------------------------------
RUN pip install peft accelerate trackio safetensors transformers==5.18.0

# --- flash-attn ---------------------------------------------------------------
# Upstream publishes a prebuilt wheel for this exact torch / CUDA / python / C++-ABI
# combination (covering sm80 + sm90), which saves a ~1h nvcc build. If it can't be
# fetched, fall back to a source build (setup.sh forgets --no-build-isolation here).
RUN PY=$(python -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")') && \
    ABI=$(python -c 'import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())') && \
    pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abi${ABI}-${PY}-${PY}-linux_x86_64.whl" \
    || pip install flash-attn==2.7.3 --no-build-isolation

# --- Native / CUDA extensions (all built with --no-build-isolation) ---------
RUN mkdir -p /tmp/extensions && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast --no-build-isolation
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    pip install /tmp/extensions/nvdiffrec --no-build-isolation
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive && \
    pip install /tmp/extensions/CuMesh --no-build-isolation
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive && \
    pip install /tmp/extensions/FlexGEMM --no-build-isolation
# o-voxel's `third_party/eigen` submodule (pinned to gitlab.com/libeigen/eigen)
# is not checked out by the plain clone above — fetch Eigen 3.4.0
# headers into it, or the CPU sources fail with "Eigen/Dense: No such file".
RUN cp -r o-voxel /tmp/extensions/o-voxel && \
    rm -rf /tmp/extensions/o-voxel/third_party/eigen && \
    git clone --depth 1 -b 3.4.0 https://gitlab.com/libeigen/eigen.git /tmp/extensions/o-voxel/third_party/eigen && \
    pip install /tmp/extensions/o-voxel --no-build-isolation

# --- Blender (pre-installed so no runtime download/apt on Jobs) --------------
# Two versions on purpose: render_cond.py pins 3.0.1, dump_mesh.py pins 4.5.1.
# download.blender.org 403s datacenter IPs, so both come from the RWTH Aachen
# mirror (with retries + a browser UA). GL/X libs come from packages.txt.
RUN wget --tries=3 --waitretry=10 -U "Mozilla/5.0" \
      https://ftp.halifax.rwth-aachen.de/blender/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz -P /tmp && \
    tar -xf /tmp/blender-3.0.1-linux-x64.tar.xz -C /tmp && \
    rm /tmp/blender-3.0.1-linux-x64.tar.xz
RUN wget --tries=3 --waitretry=10 -U "Mozilla/5.0" \
      https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz -P /tmp && \
    tar -xf /tmp/blender-4.5.1-linux-x64.tar.xz -C /tmp && \
    rm /tmp/blender-4.5.1-linux-x64.tar.xz

# --- Starter-image extras -------------------------------------------------------
# Model weights are not baked in: TRELLIS.2-4B downloads into HF_HOME on first use
# (mount a volume there to keep it), and DINOv3 is gated under Meta's license, so it
# can't be redistributed — users pass an HF_TOKEN that has been granted access.
ENV HF_HOME=/root/.cache/huggingface
COPY container/STARTER.md /workspace/STARTER.md
# Command-line entry points: the one-script pipeline, and the trellis2-prep /
# trellis2-train commands of the published -prep and -train images.
COPY preprocess.py finetune.py /workspace/
RUN install -m 755 byod/bin/trellis2-prep byod/bin/trellis2-train /usr/local/bin/
LABEL org.opencontainers.image.title="trellis2-starter" \
      org.opencontainers.image.description="TRELLIS.2 with its CUDA extensions, Blender and PEFT/Accelerate/Trackio preinstalled: write only your fine-tuning code." \
      org.opencontainers.image.licenses="MIT"
CMD ["bash"]
