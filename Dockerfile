
# # ───────────────────────────────────────────────
# # Stage 1: Build with dependencies and CUDA 12.2
# # ───────────────────────────────────────────────
# FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3.10 python3.10-venv python3.10-dev python3-pip \
#     build-essential cmake git ninja-build \
#     tesseract-ocr libgl1-mesa-glx libglib2.0-0 \
#     autoconf automake libtool \
#     && rm -rf /var/lib/apt/lists/*

# # CUDA linker fix
# RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && \
#     ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1

# # Create working dir and virtual env
# WORKDIR /app
# RUN python3.10 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # Copy requirements first for caching
# COPY requirements.txt .

# # Install Python packages
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Build llama-cpp-python with CUDA 12.2
# ENV CMAKE_ARGS="-DGGML_CUDA=on"
# ENV LLAMA_CUBLAS=1
# RUN pip install --no-cache-dir --force-reinstall llama-cpp-python --no-binary :all:

# # ───────────────────────────────────────────────
# # Stage 2: Runtime image (also CUDA 12.2)
# # ───────────────────────────────────────────────
# FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# # Runtime dependencies
# RUN apt-get update && apt-get install -y \
#     python3.10 python3.10-venv python3.10-dev \
#     tesseract-ocr libgl1-mesa-glx libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# # Copy venv from build stage
# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # Set working directory and copy app
# WORKDIR /app
# COPY . .

# # ✅ Set NLTK data dir and download inside final container
# ENV NLTK_DATA="/app/nltk_data"
# RUN mkdir -p /app/nltk_data && \
#     python3.10 -m nltk.downloader -d /app/nltk_data stopwords punkt wordnet omw-1.4

# # Expose FastAPI port
# EXPOSE 8080

# # ✅ Runtime CMD
# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]




# ───────────────────────────────────────────────
# Stage 1: Build with dependencies and CUDA 12.2
# ───────────────────────────────────────────────
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential cmake git ninja-build \
    tesseract-ocr libgl1-mesa-glx libglib2.0-0 \
    autoconf automake libtool \
    && rm -rf /var/lib/apt/lists/*

# CUDA linker fix
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1

# Create working dir and virtual env
WORKDIR /app
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for caching
COPY requirements.txt .

# Install Python packages (including pymongo) and dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pymongo

# Download spaCy English model
RUN python3.10 -m spacy download en_core_web_sm

# Build llama-cpp-python with CUDA 12.2
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV LLAMA_CUBLAS=1
RUN pip install --no-cache-dir --force-reinstall llama-cpp-python --no-binary :all:

# ───────────────────────────────────────────────
# Stage 2: Runtime image (also CUDA 12.2)
# ───────────────────────────────────────────────
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    tesseract-ocr libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from build stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory and copy app
WORKDIR /app
COPY . .

# ✅ Set NLTK data dir and download inside final container
ENV NLTK_DATA="/app/nltk_data"
RUN mkdir -p /app/nltk_data && \
    python3.10 -m nltk.downloader -d /app/nltk_data stopwords punkt wordnet omw-1.4

# Expose FastAPI port
EXPOSE 8080

# ✅ Runtime CMD
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

