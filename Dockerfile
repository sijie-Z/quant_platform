FROM python:3.12-slim

WORKDIR /app

# System dependencies for numpy/scipy/numba
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Make quant_platform importable: the repo root IS the package,
# but since the directory is named 'app' (not 'quant_platform'),
# we create a symlink so `import quant_platform` resolves.
RUN ln -sf /app /quant_platform
ENV PYTHONPATH=/:/app

# Build frontend
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && cd frontend && npm ci && npm run build \
    && apt-get purge -y nodejs npm && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.npm

EXPOSE 8000

CMD ["python", "main.py", "web"]
