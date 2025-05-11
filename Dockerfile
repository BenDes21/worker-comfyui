# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base

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
    python3.11 \
    python3.11-dev \
    build-essential \ 
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Install comfy-cli
RUN uv pip install comfy-cli --system

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.3.30 --cuda-version 12.6 --nvidia

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install Python dependencies for problematic custom nodes BEFORE snapshot restore
# These need to be present when the custom nodes are loaded/initialized
RUN uv pip install piexif ultralytics simpleeval --system

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# install dependencies
RUN uv pip install runpod requests --system

# Add files
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy the snapshot file
ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
RUN /restore_snapshot.sh

# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

# TEST ONLY
ARG HUGGINGFACE_ACCESS_TOKEN="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"
# Set default model type if none is provided
ARG MODEL_TYPE=flux1-dev-fp8

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/loras models/clip models/unet

# Flux Setup

RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/BEN_Merge_V8UV4.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/BEN_Merge_V8UV4.safetensors            
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/clip/ViT-L-14-GmP-SAE-FULL-model.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/ViT-L-14-GmP-SAE-FULL-model.safetensors 
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/clip/t5xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors 
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/ae.safetensors 

# Lora's

RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/alice-blanche.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/AliceBlanche_flux2_V1.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/anna-smirnov.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/AnnaSmirnov_flux2_V1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/aria-kelly.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/AriaKelly_Lokr_v1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/charlotte-davis.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/CharlotteDavis_flux2_V1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/chloe-williams.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/ChloeWilliams_flux2_V1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/claire-mila.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/ClaireMila_flux2_V1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/elizabeth-jones.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/ElizabethJones_flux2_V1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/emilie-martin.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/EmilyMartin_flux2_V11_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/evelyn-addison.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/EvelynAddison_flux2_V1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/jane-red.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/JaneRed_flux2_V1.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/mary-rose.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/MaryRose_flux2_V1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/nalu-kamala.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/NaluKalama_flux2_V1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/natcha-saetang.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/NatchaSaetang_flux2_V1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/olivia-daniel.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/OliviaDanielB_flux2_V1_000004500.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/julie-stephani.safetensors https://huggingface.co/Jehex/Jibv8/resolve/main/JulieStephani_V2_000005500.safetensors


# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start container
CMD ["/start.sh"]
