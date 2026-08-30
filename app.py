import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="DOE Formulation Optimizer", layout="wide")

st.title("🧪 Commercial DOE Formulation Optimizer")
st.markdown("Optimize ingredient ratios to hit target physical profiles while minimizing raw material costs.")

# --- TABS ---
tab_data, tab_opt = st.tabs(["📊 Data & Model Setup", "🎛️ Optimization Dashboard"])

with tab_data:
    st.subheader("1. Upload Batch Records")
    uploaded_file = st.file_uploader("Upload CSV/Excel (Requires columns: Carbomer_pct, Surfactant_pct, Viscosity_cP)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
    else:
        st.info("Using synthetic historical batch data for demonstration.")
        np.random.seed(42)
        c = np.random.uniform(0.1, 1.0, 100)
        s = np.random.uniform(5.0, 15.0, 100)
        v = 1000 + (c * 5000) - (s * 50) + (c * s * 200) + np.random.normal(0, 50, 100)
        df = pd.DataFrame({'Carbomer_pct': c, 'Surfactant_pct': s, 'Viscosity_cP': v})
    
    st.dataframe(df.head())

    st.subheader("2. Train Surrogate Model")
    model_choice = st.radio("Select Algorithm", ["Polynomial Regression (Degree 2)", "Random Forest Regressor"])
    
    X = df[['Carbomer_pct', 'Surfactant_pct']]
    y = df['Viscosity_cP']
    
    if "Polynomial" in model_choice:
        poly = PolynomialFeatures(degree=2)
        X_tf = poly.fit_transform(X)
        model = LinearRegression().fit(X_tf, y)
        y_pred = model.predict(X_tf)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        y_pred = model.predict(X)
        
    st.success(f"Model trained successfully! R² Score: {r2_score(y, y_pred):.3f}")

with tab_opt:
    st.sidebar.title("Target Parameters")
    target_viscosity = st.sidebar.slider("Target Viscosity (cP)", 1000, 7000, 4000, step=100)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Raw Material Costs ($/kg)")
    cost_carbomer = st.sidebar.number_input("Carbomer Cost", value=45.0)
    cost_surfactant = st.sidebar.number_input("Surfactant Cost", value=12.0)

    # Optimization Logic
    def objective(x):
        if "Polynomial" in model_choice:
            pred = model.predict(poly.transform([[x[0], x[1]]]))[0]
        else:
            pred = model.predict([[x[0], x[1]]])[0]
        cost_penalty = (x[0] * cost_carbomer) + (x[1] * cost_surfactant)
        return (pred - target_viscosity)**2 + (cost_penalty * 0.1)

    res = minimize(objective, x0=[0.5, 10.0], bounds=[(0.1, 1.0), (5.0, 15.0)])
    opt_c, opt_s = res.x
    
    opt_pred = model.predict(poly.transform([[opt_c, opt_s]]) if "Polynomial" in model_choice else [[opt_c, opt_s]])[0]
    optimal_cost = (opt_c * cost_carbomer) + (opt_s * cost_surfactant)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal Carbomer", f"{opt_c:.3f} %")
    col2.metric("Optimal Surfactant", f"{opt_s:.2f} %")
    col3.metric("Achieved Viscosity", f"{opt_pred:.0f} cP")
    col4.metric("Est. Cost per 100kg", f"${optimal_cost:.2f}")

    # Generate Grid for Visuals
    c_grid, s_grid = np.meshgrid(np.linspace(0.1, 1.0, 30), np.linspace(5.0, 15.0, 30))
    grid_pts = np.c_[c_grid.ravel(), s_grid.ravel()]
    
    if "Polynomial" in model_choice:
        v_pred = model.predict(poly.transform(grid_pts)).reshape(c_grid.shape)
    else:
        v_pred = model.predict(grid_pts).reshape(c_grid.shape)

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        fig3d = go.Figure(data=[go.Surface(z=v_pred, x=c_grid, y=s_grid, colorscale='Viridis')])
        fig3d.add_trace(go.Scatter3d(x=[opt_c], y=[opt_s], z=[opt_pred], mode='markers', marker=dict(color='red', size=8)))
        fig3d.update_layout(title="3D Response Surface", margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig3d, use_container_width=True)

    with plot_col2:
        fig2d = go.Figure(data=[go.Contour(z=v_pred, x=np.linspace(0.1, 1.0, 30), y=np.linspace(5.0, 15.0, 30), colorscale='Viridis')])
        fig2d.add_trace(go.Scatter(x=[opt_c], y=[opt_s], mode='markers', marker=dict(color='red', size=12, symbol='star')))
        fig2d.update_layout(title="2D Contour Map", margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig2d, use_container_width=True)