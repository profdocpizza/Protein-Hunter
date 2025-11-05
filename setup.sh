#!/bin/bash
set -e

echo "🚀 Setting up Boltz Design Environment..."

# ------------------------------------------------------------------------------
# 1️⃣ Check Conda
# ------------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# ------------------------------------------------------------------------------
# 2️⃣ Create and activate environment
# ------------------------------------------------------------------------------
echo "📦 Creating conda environment..."
conda create -n proteinhunter python=3.10 -y
conda activate proteinhunter

# Verify we're in the right environment
echo "📍 Active environment: $CONDA_DEFAULT_ENV"
if [ "$CONDA_DEFAULT_ENV" != "proteinhunter" ]; then
    echo "❌ Failed to activate proteinhunter environment"
    exit 1
fi
# ------------------------------------------------------------------------------
# 3️⃣ Install Boltz
# ------------------------------------------------------------------------------
if [ -d "boltz_ph" ]; then
    echo "📂 Installing Boltz..."
    cd boltz_ph
    pip install -e .
    cd ..
else
    echo "❌ boltz_ph directory not found. Please run this script from the project root."
    exit 1
fi

# ------------------------------------------------------------------------------
# 4️⃣ Install conda dependencies
# ------------------------------------------------------------------------------
echo "🔧 Installing conda dependencies..."
conda install -c anaconda ipykernel -y

# ------------------------------------------------------------------------------
# 5️⃣ Install Python dependencies
# ------------------------------------------------------------------------------
echo "🔧 Installing Python dependencies..."
pip install matplotlib seaborn prody tqdm PyYAML requests pypdb py3Dmol py2Dmol logmd==0.1.45
pip install ml_collections

# ------------------------------------------------------------------------------
# 6️⃣ Install PyRosetta
# ------------------------------------------------------------------------------
echo "⏳ Installing PyRosetta (this may take a while)..."
pip install pyrosettacolabsetup pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'

# ------------------------------------------------------------------------------
# 7️⃣ Fix NumPy + Numba compatibility (PyRosetta downgrades NumPy)
# ------------------------------------------------------------------------------
echo "🩹 Fixing NumPy/Numba version for Boltz and diffusion..."
pip install --upgrade "numpy>=1.24,<1.27" numba

# ------------------------------------------------------------------------------
# 8️⃣ Download Boltz weights and dependencies
# ------------------------------------------------------------------------------
echo "⬇️  Downloading Boltz weights and dependencies..."
python << 'PYCODE'
import sys
import os
# Ensure boltz is importable
sys.path.insert(0, os.path.join(os.getcwd(), 'boltz', 'src'))

try:
    from boltz.main import download_boltz2
    from pathlib import Path
    cache = Path('~/.boltz').expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)
    print("✅ Boltz weights downloaded successfully!")
except Exception as e:
    print(f"❌ Error downloading Boltz weights: {e}")
    sys.exit(1)
PYCODE

# ------------------------------------------------------------------------------
# 9️⃣ Setup LigandMPNN if directory exists
# ------------------------------------------------------------------------------
if [ -d "LigandMPNN" ]; then
    echo "🧬 Setting up LigandMPNN..."
    cd LigandMPNN
    bash get_model_params.sh "./model_params"
    cd ..
fi

# ------------------------------------------------------------------------------
# 🔟 Make DAlphaBall.gcc executable
# ------------------------------------------------------------------------------
chmod +x "utils/DAlphaBall.gcc" || { echo "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# ------------------------------------------------------------------------------
# 1️⃣1️⃣ Install Chai Lab (NEW SECTION)
# ------------------------------------------------------------------------------
echo "🧠 Installing chai-lab and dependencies..."
pip install --no-deps \
  git+https://github.com/sokrypton/chai-lab.git \
  'gemmi~=0.6.3' \
  'jaxtyping>=0.2.25' \
  'pandera>=0.24' \
  'antipickle==0.2.0' \
  'rdkit~=2024.9.5' \
  'modelcif>=1.0' \
  'biopython>=1.83' \
  typing_inspect \
  beartype \
  typeguard \
  ihm \
  mypy_extensions \
  equinox \
  wadler_lindig \
  py3Dmol

# check that chai-lab installed and fail if not
python -c "import chai_lab" || { echo '❌ chai-lab install failed!' ; exit 1; }

# ------------------------------------------------------------------------------
# 1️⃣2️⃣ Setup Jupyter kernel
# ------------------------------------------------------------------------------
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=proteinhunter --display-name="Protein Hunter"

echo "🎉 Installation complete!"
echo "➡️  Activate environment with: conda activate proteinhunter"