# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage with sensible defaults for standalone builds
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    build-essential \
    g++ \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed (for newer CUDA versions)
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# ============ ACE-Step 1.5 ============
# Custom node para geração de música
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/billwuhao/ComfyUI_ACE-Step.git && \
    cd ComfyUI_ACE-Step && \
    uv pip install --no-cache-dir -r requirements.txt

# Instala transformers 4.44.0 DEPOIS do requirements (para não ser sobrescrito)
# Esta versão é compatível com o modelo UMT5 do ACE-Step
RUN uv pip install --no-cache-dir transformers==4.44.0 huggingface_hub accelerate
# ======================================

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
# Set default model type if none is provided
ARG MODEL_TYPE=ace-step

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/text_encoders models/diffusion_models models/model_patches

# Download checkpoints/vae/unet/clip models to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -q -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -q -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -q -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "sd3" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
      wget -q -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "z-image-turbo" ]; then \
      wget -q -O models/text_encoders/qwen_3_4b.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors && \
      wget -q -O models/diffusion_models/z_image_turbo_bf16.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors && \
      wget -q -O models/vae/ae.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors && \
      wget -q -O models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors; \
    fi

# ACE-Step 1.5 - Geração de Música
# O custom node ComfyUI_ACE-Step espera modelos em /comfyui/models/TTS/
RUN if [ "$MODEL_TYPE" = "ace-step" ]; then \
      mkdir -p models/TTS/ACE-Step-v1-3.5B/ace_step_transformer && \
      mkdir -p models/TTS/ACE-Step-v1-3.5B/music_dcae_f8c8 && \
      mkdir -p models/TTS/ACE-Step-v1-3.5B/music_vocoder && \
      mkdir -p models/TTS/ACE-Step-v1-3.5B/umt5-base && \
      echo "Downloading ACE-Step config..." && \
      wget --progress=dot:giga -O models/TTS/ACE-Step-v1-3.5B/config.json \
        https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/config.json && \
      echo "Downloading ace_step_transformer..." && \
      wget --progress=dot:giga -O models/TTS/ACE-Step-v1-3.5B/ace_step_transformer/config.json \
        https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/ace_step_transformer/config.json && \
      wget --progress=dot:giga -O models/TTS/ACE-Step-v1-3.5B/ace_step_transformer/diffusion_pytorch_model.safetensors \
        https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/ace_step_transformer/diffusion_pytorch_model.safetensors && \
      echo "Downloading music_dcae_f8c8..." && \
      wget --progress=dot:giga -O models/TTS/ACE-Step-v1-3.5B/music_dcae_f8c8/config.json \
        https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_dcae_f8c8/config.json && \
      wget --progress=dot:giga -O models/TTS/ACE-Step-v1-3.5B/music_dcae_f8c8/diffusion_pytorch_model.safetensors \
        https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_dcae_f8c8/diffusion_pytorch_model.safetensors && \
      echo "Downloading music_vocoder..." && \
      wget --progress=dot:giga -O models/TTS/ACE-Step-v1-3.5B/music_vocoder/config.json \
        https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/config.json && \
      wget --progress=dot:giga -O models/TTS/ACE-Step-v1-3.5B/music_vocoder/diffusion_pytorch_model.safetensors \
        https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/diffusion_pytorch_model.safetensors && \
      echo "Downloading UMT5 from Google (complete model)..." && \
      HF_HUB_DISABLE_PROGRESS_BARS=1 huggingface-cli download google/umt5-base \
        --local-dir models/TTS/ACE-Step-v1-3.5B/umt5-base && \
      echo "Converting UMT5 pytorch_model.bin to model.safetensors..." && \
      echo 'import torch; from safetensors.torch import save_file; p="models/TTS/ACE-Step-v1-3.5B/umt5-base"; sd=torch.load(p+"/pytorch_model.bin",map_location="cpu",weights_only=True); save_file(sd,p+"/model.safetensors"); print(f"Converted {len(sd)} tensors")' > /tmp/convert_umt5.py && \
      python /tmp/convert_umt5.py && \
      rm -f models/TTS/ACE-Step-v1-3.5B/umt5-base/pytorch_model.bin /tmp/convert_umt5.py && \
      echo "ACE-Step models downloaded successfully!"; \
    fi

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models
