import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

import utils.yield_curve as yield_curve


class YieldCurvePCA:
    def __init__(self, data, run_on_change, time_scaler):
        input_df = data.diff().dropna() if run_on_change else data

        self.run_on_change = run_on_change
        self.time_scaler = time_scaler
        self.maturities = list(data.columns)
        
        self.pca = PCA(n_components=3)
        self.pca.fit_transform(input_df)

        self.loadings_scaled = self.pca.components_ * np.sqrt(self.pca.explained_variance_[:, np.newaxis])
        self.loadings_time_scaled = self.loadings_scaled * self.time_scaler


    def explained_variance(self):
        stds = np.sqrt(self.pca.explained_variance_)
        stds_time_scaled = stds * self.time_scaler
        prop_var = self.pca.explained_variance_ratio_ * 100
        cum_prop = np.cumsum(prop_var)

        table = pd.DataFrame({
            'Standard Deviation': [f"{s:.2f} bp" for s in stds_time_scaled],
            'Proportion of Variance': [f"{p:.1f} %" for p in prop_var],
            'Cumulative Proportion': [f"{c:.1f}" for c in cum_prop]
        }, index=[f'Component #{i+1}' for i in range(len(stds_time_scaled))]).T
        
        return table

    
    def plot_components(self):
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=self.maturities,
            y=self.loadings_time_scaled[0] * 100,
            mode='lines',
            name='PC #1: Level'
        ))
    
        fig.add_trace(go.Scatter(
            x=self.maturities,
            y=self.loadings_time_scaled[1] * 100,
            mode='lines',
            name='PC #2: Slope'
        ))
    
        fig.add_trace(go.Scatter(
            x=self.maturities,
            y=self.loadings_time_scaled[2] * 100,
            mode='lines',
            name='PC #3: Curvature'
        ))
    
        fig.update_layout(
            title=f"First Three Principal Components of Yield Curve {'Changes' if self.run_on_change else 'Levels'}",
            xaxis_title="Years to Maturity",
            yaxis_title="Yield Change (bp)",
            legend=dict(
                x=1.05,  # position just outside the plot area
                y=0.5,
                xanchor='left',
                yanchor='middle',
                orientation='v'
            ),
            margin=dict(l=50, r=120, t=50, b=50),
        )
    
        return fig


    def get_fly_dv01_weights(self, wing1, center, wing2):
        wing1_ind = self.maturities.index(wing1)
        center_ind = self.maturities.index(center)
        wing2_ind = self.maturities.index(wing2)
        
        M = self.loadings_time_scaled[:, [wing1_ind, center_ind, wing2_ind]]
        pc1_w = np.cross(M[1], M[2]); pc1_w = pc1_w / sum(pc1_w)
        pc2_w = np.cross(M[0], M[2]); pc2_w = pc2_w / pc2_w[0]
        pc3_w = np.cross(M[0], M[1]); pc3_w = pc3_w / pc3_w[1]
        
        return pc1_w, pc2_w, pc3_w

    
    def get_fly_market_values(self, wing1, center, wing2, exposure, yc_curr):
        fly_par_tenors = [wing1, center, wing2]
        fly_par_rates = yc_curr.loc[fly_par_tenors]

        # Calculate each security's price change for shifts in each PC
        pc_pdeltas = [] # 3x3, pcs x components

        for loading in self.loadings_time_scaled:
            component_pdeltas = []
            # +-1sd yc realization
            yc_pos_sd = yc_curr + loading
            yc_neg_sd = yc_curr - loading
            # Iterate through each component of the fly
            for i in range(3):
                pdelta = (
                    yield_curve.price_bond(yc_pos_sd, fly_par_tenors[i], fly_par_rates.iloc[i]) - 
                    yield_curve.price_bond(yc_neg_sd, fly_par_tenors[i], fly_par_rates.iloc[i])
                ) / 2
                # Normalize price delta to per $1 notional
                pdelta /= 100
                component_pdeltas.append(pdelta)
            pc_pdeltas.append(component_pdeltas)
        
        # Solve for market values for securities and pcs using footnote 35
        M = np.array(pc_pdeltas)
        rhs = np.eye(3) * exposure
        market_values = np.linalg.solve(M, rhs).T
        
        return market_values

        