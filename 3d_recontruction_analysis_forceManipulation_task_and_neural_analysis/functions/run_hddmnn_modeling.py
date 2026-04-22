import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Added import for Pandas

import hddm
import pymc as pm # Explicitly import pymc for summary function
import arviz as az # Explicitly import arviz for summary function

def run_hddmnn_modeling(df_animal_data, animal_id, samples, burn, thin): # Modified signature
    """
    Runs the Hierarchical Drift-Diffusion Model for a single animal using the provided dataframe.

    Args:
        df_animal_data (pd.DataFrame): DataFrame for a single animal's trials.
        animal_id (str): Identifier for the animal (e.g., 'animal1').
        samples (int): Number of MCMC samples to draw.
        burn (int): Number of burn-in samples to discard.
        thin (int): Thinning interval for MCMC samples.

    Returns:
        hddm.HDDM: The fitted HDDM model object.
    """
    print(f"\n--- Running HDDM Modeling for {animal_id} ---")

    df_combined = df_animal_data # Directly use the single animal's data
    
    # Debugging prints
    print(f"\n--- DataFrame Info for {animal_id} before HDDM ---")
    print("Columns:", df_combined.columns.tolist())
    print("Head:\n", df_combined.head())
    print("Tail:\n", df_combined.tail())
    print("Shape:", df_combined.shape)
    print("Is empty:", df_combined.empty)
    print("NaN counts per column:\n", df_combined.isnull().sum())

    # List of all covariates used in depends_on and regressors
    covariates = [
        'self_gaze_auc',
        'partner_mean_speed',
        'failed_pulls_before_reward',
        'time_since_last_reward',
        'prev_trial_outcome' 
    ]

    # Ensure all covariates are numeric and drop rows with NaNs in these specific columns
    for col in covariates:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        else:
            print(f"Error: Covariate column '{col}' not found in DataFrame for {animal_id}. Please check data preparation.")
            return None # Exit if critical column is missing

    # Drop rows where any of the specified covariates or RT are NaN
    df_combined = df_combined.dropna(subset=['rt'] + covariates)

    print(f"\n--- DataFrame Info for {animal_id} AFTER NaN-dropping for HDDM ---")
    print("Columns:", df_combined.columns.tolist())
    print("Head:\n", df_combined.head())
    print("Tail:\n", df_combined.tail())
    print("Shape:", df_combined.shape)
    print("Is empty:", df_combined.empty)
    print("NaN counts per column:\n", df_combined.isnull().sum())

    if df_combined.empty:
        print(f"Error: DataFrame for {animal_id} is empty after dropping NaNs for covariates. Cannot run HDDM.")
        return None

    # Crucial check: Ensure covariates have variance. HDDM (PyMC) needs variability for regression.
    for col in covariates:
        if df_combined[col].nunique() < 2:
            print(f"Warning: Covariate '{col}' has no variance (only one unique value) in {animal_id}'s data after filtering. HDDM may fail or estimate it poorly for this covariate.")
            # If a covariate has no variance, it essentially can't be used as a regressor.
            # You might consider removing it from depends_on/regressors if this is a persistent issue.

    # Define the HDDM model
    print(f"Defining HDDMnnRegressor model for {animal_id} with dependencies:")
    print(f"  v (drift rate) depends on: self_gaze_auc, partner_mean_speed")
    print(f"  a (boundary separation) depends on: failed_pulls_before_reward, time_since_last_reward")
    print(f"  z (starting bias) depends on: prev_trial_outcome (categorical)") 

    
    # Run MCMC sampling
    print(f"Sampling HDDM with {samples} samples, {burn} burn-in...")
    # # simple HDDM model
    # model = hddm.HDDM(df_combined,
    #                   include=['v','a','z','t'], # Explicitly include all core DDM parameters
    #                   # depends_on={'v': ['self_gaze_auc', 'partner_mean_speed'],
    #                   #             'a': ['failed_pulls_before_reward', 'time_since_last_reward'],
    #                   #             'z': 'prev_trial_outcome'}
    #                  ) 
    
    # # Using HDDMRegressor for linear regression with continuous covariates
    model = hddm.HDDMnnRegressor(
                                    df_combined,
                                    [
                                        'v ~ self_gaze_auc + partner_mean_speed',
                                        'a ~ failed_pulls_before_reward + time_since_last_reward'
                                    ],
                                    include=['v', 'a', 'z', 't'],
                                    depends_on={'z': 'prev_trial_outcome'}
                                )
    
    # for hypothesis test 
    model_nogaze = hddm.HDDMnnRegressor(
                                        df_combined,
                                        [
                                            # 'v ~ partner_mean_speed',
                                            'v ~ partner_mean_speed + failed_pulls_before_reward + time_since_last_reward',
                                            'a ~ failed_pulls_before_reward + time_since_last_reward'
                                        ],
                                        include=['v', 'a', 'z', 't'],
                                        depends_on={'z': 'prev_trial_outcome'}
                                    )
    
    
    # Run MCMC sampling
    print(f"Sampling HDDM with {samples} samples, {burn} burn-in, {thin} thinning...")
    # Modified this line: Removed 'thin' and used proper args for hddm 0.8.0 / PyMC3 API
    m = model.sample(samples, burn=burn, 
                     dbname=f'traces_{animal_id}.db', db='pickle') # Saves traces to a file per animal
    m_nogaze = model_nogaze.sample(samples, burn=burn, 
                             dbname=f'traces_{animal_id}.db', db='pickle') # Saves traces to a file per animal
    
    print(f"\n--- HDDM Sampling Complete for {animal_id} ---")
    
    # Print summary of parameters
    print(f"\n--- HDDM Parameter Summary for {animal_id} ---")
    
    # model.print_stats()

    # Optional: Plot posteriors (can be slow for many parameters)
    # model.plot_posteriors()
    # plt.show()

    return model,model_nogaze
