import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Added import for Pandas
import seaborn as sns
import scipy
import scipy.stats as st
import scipy.io
from scipy.stats import pearsonr
from scipy.stats import linregress


# do some important plotting and show the important statistics
def do_hddm_model_fitted_plot(df_with_v):


    # --- Plotting and Analysis ---
    #
    # use deming regression
    #
    # --- Prepare data ---
    x = df_with_v['predicted_v'].values
    y = df_with_v['self_gaze_auc'].values
    # lambda_ = 0.5  # Error variance ratio; adjust if known
    lambda_ = np.var(x, ddof=1) / np.var(y, ddof=1)

    # 1. Compute means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # 2. Compute variances and covariance
    s_xx = np.var(x, ddof=1)
    s_yy = np.var(y, ddof=1)
    s_xy = np.cov(x, y, ddof=1)[0, 1]

    # 3. Compute slope and intercept using Deming formula
    delta = (s_yy - lambda_ * s_xx)**2 + 4 * lambda_ * s_xy**2
    slope = (s_yy - lambda_ * s_xx + np.sqrt(delta)) / (2 * s_xy)
    intercept = y_mean - slope * x_mean

    # alternative, normal regression
    slope, intercept, _, _, _ = linregress(x, y)
    
    # 4. Pearson correlation (still informative)
    r_value, p_value = pearsonr(x, y)

    # 5. Plot
    fig = plt.figure(figsize=(6, 6))
    sns.scatterplot(x=x, y=y, alpha=0.6, s=50)

    # Regression line
    x_vals = np.linspace(min(x), max(x), 100)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red', linewidth=2, label='Deming Regression')

    # Labels
    plt.title('Deming Regression: Self-Gaze vs Predicted Drift Rate (v)', fontsize=16)
    plt.xlabel('Predicted Drift Rate (v) from HDDM Model', fontsize=12)
    plt.ylabel('Self Gaze AUC', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # Annotate
    stats_text = (
        f"Pearson's r = {r_value:.4f}\n"
        f"p-value = {p_value:.4f}\n"
        f"Deming slope = {slope:.4f}\n"
        f"Intercept = {intercept:.4f}"
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # plt.show()

    # Print stats
    print("Regression Statistics:")
    print(f"Pearson's r: {r_value:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Deming slope: {slope:.4f}")
    print(f"Deming intercept: {intercept:.4f}")
    
    
    return fig, r_value, p_value
