# 🧪 Commercial DOE Formulation Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://formulation-doe-optimizer.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

An interactive machine learning dashboard designed to bridge the gap between wet-lab chemistry and computational data science. This tool reverse-engineers optimal cosmetic and pharmaceutical ingredient ratios to hit target physical profiles (like viscosity) while simultaneously minimizing raw material costs.

## 💡 Key Features
- **Custom Data Ingestion:** Upload proprietary batch records (CSV/Excel) or use the synthetic demonstration data.
- **Algorithm Selection:** Train surrogate models dynamically, choosing between Polynomial Regression for smooth response surfaces or Random Forest Regressors for complex, non-linear chemical interactions.
- **Cost Optimization:** Utilizes `scipy.optimize.minimize` to balance technical specifications against real-time financial constraints.
- **Chemometric Visualization:** Interactive 3D response surfaces and 2D contour mapping via Plotly.

## 🚀 Local Deployment

This project is optimized for execution within isolated Linux/WSL environments using Mamba/Conda, but runs perfectly on standard Python setups.

**1. Clone & Navigate**
```bash
git clone https://github.com/Arashka-Ch/formulation-doe-optimizer.git
cd formulation-doe-optimizer
```

**2. Provision the Environment**
```bash
mamba create -n formulation_env python=3.10 -y
mamba activate formulation_env
pip install -r requirements.txt
```

**3. Launch the Dashboard**
```bash
streamlit run app.py
```
