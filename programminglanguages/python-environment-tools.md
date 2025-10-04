# Python Development Environment Management

A guide to understanding and choosing tools for managing Python versions, virtual environments, and packages.

## The Core Tools

### pyenv - Python Version Manager
Manages multiple Python interpreter versions on your system.

```bash
pyenv install 3.11.5
pyenv install 3.9.10
pyenv local 3.11.5  # Use 3.11.5 for this project
```

**Use when**: Different projects need different Python versions (3.8, 3.11, etc.)

### venv - Virtual Environment Creator
Creates isolated environments for package dependencies. Built into Python 3.3+.

```bash
python -m venv myproject_env
source myproject_env/bin/activate
pip install numpy pandas
```

**Use when**: Projects need different versions of the same package

### conda - All-in-One Solution
Manages Python versions, environments, and packages in one tool. Can also handle non-Python dependencies.

```bash
conda create -n myenv python=3.11
conda activate myenv
conda install numpy pytorch cudatoolkit
```

**Use when**: Scientific computing, complex binary dependencies, or learning environments

### uv - Modern Fast Alternative
Rust-based tool that replaces pip and venv with 10-100x speed improvements.

```bash
uv venv
source .venv/bin/activate
uv pip install torch transformers
```

**Use when**: You want speed and modern tooling for production environments

## Understanding the Relationships

### Two-Layer Architecture: pyenv + venv
Most production setups separate concerns:
- **pyenv**: Controls which Python version you're using
- **venv**: Creates isolated package environments within that Python version

```bash
pyenv local 3.11.5       # Layer 1: Choose Python version
python -m venv .venv     # Layer 2: Create isolated environment
source .venv/bin/activate
pip install requirements.txt
```

This gives you:
- Project A: Python 3.11 + Django 4.x
- Project B: Python 3.11 + Django 3.x
- Project C: Python 3.9 + Flask 2.x

### Single-Layer Architecture: conda
Conda collapses both layers into one tool:

```bash
conda create -n projectA python=3.11 django=4.0
conda create -n projectB python=3.11 django=3.2
```

Treats Python itself as just another package.

## Choosing Your Stack

### For Modern Web/ML Production: pyenv + uv
**Best for**: Web development, ML infrastructure, microservices, CI/CD

```bash
# Setup
pyenv install 3.11.5
pyenv local 3.11.5
uv venv
uv pip install fastapi torch vllm sqlalchemy
```

**Advantages**:
- Fast iteration (uv is 10-100x faster than pip)
- Small Docker images
- Matches production deployment practices
- Works with all PyPI packages (500k+)

**Works for ML**: PyTorch, TensorFlow, vLLM all ship pip wheels with CUDA support now

### For Scientific Computing: conda/miniconda
**Best for**: Data science research, bioinformatics, academic work, complex C/Fortran dependencies

```bash
conda create -n research python=3.11
conda activate research
conda install numpy scipy matplotlib jupyter r-base
```

**Advantages**:
- Handles non-Python dependencies (CUDA, system libraries)
- Pre-compiled binaries for hard-to-build packages
- Works across languages (Python, R, Julia)
- One-stop shop for beginners

**Downsides**:
- Slower than pip/uv
- Large footprint (~500MB+ base)
- Less suitable for production containers

### For Traditional Simplicity: pyenv + venv + pip
**Best for**: Learning, simple projects, following standard tutorials

```bash
pyenv local 3.11.5
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Advantages**:
- Well-documented everywhere
- Minimal abstraction
- Standard Python tooling

**Downsides**:
- Slower than uv
- More verbose than alternatives

## Common Questions

### Can I use pip/uv for ML workloads?
**Yes!** Modern ML packages work perfectly with pip:

```bash
uv pip install torch torchvision torchaudio
uv pip install transformers accelerate vllm
uv pip install tensorflow scikit-learn xgboost

# Specific CUDA versions also work
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

This works because PyPI now hosts pre-compiled CUDA wheels. You only need conda for exotic scientific packages.

### What about conda vs Anaconda vs Miniconda?
- **conda**: The package manager tool itself (like git)
- **Anaconda**: 3GB distribution with conda + Python + 250+ packages pre-installed
- **Miniconda**: 200MB minimal installer with just conda + Python
- **Miniforge**: Community alternative to Miniconda, better for Apple Silicon

For experienced users: Use Miniconda or Miniforge, not full Anaconda.

### Is conda still relevant?
**Yes, but for specific domains**:
- **Still dominant**: Bioinformatics, computational chemistry, academic research, HPC
- **Declining**: Production ML, web development, modern software engineering

The ML production world has largely moved to pip/uv for speed and lighter containers.

## Historical Context

### Why conda was created (2012)
Installing scientific Python packages used to require compiling C/Fortran code:

```bash
# 2012 experience:
pip install numpy
# ERROR: Microsoft Visual C++ required
# ERROR: BLAS libraries not found
# ERROR: Fortran compiler missing
```

Conda solved this by shipping pre-compiled binaries for everything.

### What changed (2020-2024)
- pip adopted binary wheels (borrowed conda's best idea)
- PyTorch/TensorFlow started shipping CUDA wheels on PyPI
- Docker made system dependencies less problematic
- Tools like uv made pip incredibly fast

Result: pip caught up to conda's advantages while staying lightweight.

### The current landscape (2025)
- **Scientific computing**: conda still essential
- **ML production**: pip/uv is standard
- **Academia**: conda remains default
- **Modern startups**: pip/uv preferred

## Recommendations by Use Case

| Your Work | Recommended Stack | Why |
|-----------|------------------|-----|
| Web development | pyenv + uv | Fast, standard, production-ready |
| ML infrastructure | pyenv + uv | Handles PyTorch/CUDA, matches production |
| Full-stack development | pyenv + uv | Covers web + ML + databases |
| Data science research | conda/miniconda | Easy setup, comprehensive packages |
| Bioinformatics | conda/miniforge | Bioconda channel is essential |
| Learning Python | pyenv + venv + pip | Well-documented, standard approach |
| Academic coursework | conda | If course materials use it |

## Quick Start Guide

### Setting up pyenv + uv (Recommended for most)

```bash
# Install tools (macOS)
brew install pyenv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Per project
cd myproject
pyenv install 3.11.5
pyenv local 3.11.5
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Development
uv pip install package-name  # Fast installs
```

### Setting up conda

```bash
# Install Miniconda
brew install --cask miniconda
conda init zsh  # or bash

# Per project
conda create -n myproject python=3.11
conda activate myproject
conda install numpy pandas jupyter
```

## Key Takeaways

1. **Two philosophies**: Specialized tools (pyenv+uv) vs all-in-one (conda)
2. **Modern ML works with pip**: No need for conda in production inference/training
3. **Speed matters**: uv is 10-100x faster than pip or conda
4. **Production prefers pip**: Lighter containers, faster builds
5. **Conda still essential**: For bioinformatics and complex scientific computing
6. **Choose based on domain**: Web/ML infra → uv, Scientific research → conda

Most developers doing modern web, ML infrastructure, or full-stack work should use **pyenv + uv**. Use conda only if you have specific needs for scientific packages or are following educational materials that require it.
