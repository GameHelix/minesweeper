# Quick Start Guide

## Setup (One-time)

### Automated Setup

**On Mac/Linux:**
```bash
./setup.sh
```

**On Windows:**
```cmd
setup.bat
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Analysis

### Option 1: Run the Python Script (Recommended)

```bash
# Activate venv first
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Run analysis
python gtzan_analysis.py
```

**Output:**
- 26 charts in `charts/` folder
- Console output with statistics
- Runtime: ~5-10 minutes

### Option 2: Use Jupyter Notebook

```bash
# Activate venv first
source venv/bin/activate

# Start Jupyter
jupyter notebook gtzan_analysis.ipynb
```

**Interactive:**
- Run cells one by one
- Explore and modify code
- See outputs inline

## What You Get

After running the analysis:

✓ **26 high-resolution charts** showing:
- Genre distributions
- Feature analysis (MFCC, tempo, spectral, etc.)
- Correlations
- PCA visualizations
- Audio waveforms and spectrograms
- Genre similarity analysis

✓ **Console summary** with:
- Dataset statistics
- Key findings
- Genre characteristics

## Project Files

```
├── gtzan_analysis.py       # Main analysis script
├── gtzan_analysis.ipynb    # Jupyter notebook version
├── requirements.txt        # Dependencies
├── setup.sh               # Auto-setup (Mac/Linux)
├── setup.bat              # Auto-setup (Windows)
├── README.md              # Full documentation
└── charts/                # Generated visualizations
```

## Common Commands

```bash
# Activate environment
source venv/bin/activate

# Run analysis
python gtzan_analysis.py

# Open notebook
jupyter notebook

# Deactivate environment
deactivate

# Re-install dependencies
pip install -r requirements.txt

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Troubleshooting

**Issue: "command not found: python3"**
- Install Python 3.8+ from python.org

**Issue: "No module named 'librosa'"**
- Activate venv: `source venv/bin/activate`
- Install deps: `pip install -r requirements.txt`

**Issue: "Permission denied: ./setup.sh"**
- Make executable: `chmod +x setup.sh`

**Issue: librosa installation fails**
- Mac: `brew install libsndfile`
- Ubuntu: `sudo apt-get install libsndfile1`

## Quick Tips

1. **First time?** Run the automated setup script
2. **Want quick results?** Use `gtzan_analysis.py`
3. **Want to explore?** Use the Jupyter notebook
4. **Charts not appearing?** Check the `charts/` folder
5. **Need help?** See README.md for detailed docs

---

**Ready to analyze music! 🎵**
