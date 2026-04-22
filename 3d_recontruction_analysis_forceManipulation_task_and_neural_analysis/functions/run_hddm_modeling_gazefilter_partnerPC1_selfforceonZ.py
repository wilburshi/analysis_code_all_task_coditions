import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Added import for Pandas

import hddm
import pymc as pm # Explicitly import pymc for summary function
import arviz as az # Explicitly import arviz for summary function

def run_hddm_modeling_gazefilter_partnerPC1_selfforceonZ(df_animal_data, animal_id, samples, burn, thin): # Modified signature
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
    
    #
    df_combined = df_combined.rename(columns={"pull_rt": "rt"})
    df_combined['response'] = 1
    
    
    # Debugging prints
    print(f"\n--- DataFrame Info for {animal_id} before HDDM ---")
 

    # List of all covariates used in depends_on and regressors
    covariates = [
       'self_force', 'partner_force', 'self_LS_mean', 'self_LS_std',
       'num_preceding_failpull', 'time_from_last_reward', 'pull_interval',
       'gaze_accum', 'partner_PC1_mean', 'partner_PC1_std', 'subblockID',
       'delta_self_force_first', 'delta_self_force_prev',
       'delta_partner_force_first', 'delta_partner_force_prev',
       'delta_partner_force_withself', 'prev_trial_outcome',
    ]
    
    # zscore the value in these column
    for col in covariates:
        mean_val = df_combined[col].mean()
        std_val = df_combined[col].std()
        #
        if std_val == 0:
            print(f"Warning: {col} has zero variance. Leaving unchanged.")
        else:
            df_combined[col] = (df_combined[col] - mean_val) / std_val
    

    # Drop rows where any of the specified covariates or RT are NaN
    df_combined = df_combined.dropna(subset=['rt'] + covariates)
    
    

    print(f"\n--- DataFrame Info for {animal_id} AFTER NaN-dropping for HDDM ---")


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
    print(f"Defining HDDMRegressor model for {animal_id} with dependencies")

    
    # Run MCMC sampling
    print(f"Sampling HDDM with {samples} samples, {burn} burn-in...")
    # # simple HDDM model
    # model = hddm.HDDM(df_combined,
    #                   include=['v','a','z','t'], # Explicitly include all core DDM parameters
    #                   # depends_on={'v': ['self_gaze_auc', 'partner_mean_speed'],
    #                   #             'a': ['failed_pulls_before_reward', 'time_since_last_reward'],
    #                   #             'z': 'prev_trial_outcome'}
    #                  ) 
    
    
    # for hypothesis test 
    model_nogaze = hddm.HDDMRegressor(
                                        df_combined,
                                        [
                                            # 'v ~ partner_mean_speed + time_since_last_reward + C(condition)',
                                            # 'v ~ partner_mean_speed + failed_pulls_before_reward + time_since_last_reward',
                                            # 'v ~ partner_mean_speed + time_since_last_reward',
                                            # 'v ~ partner_speed_std + time_since_last_reward',
                                            # 'v ~ partner_mean_speed',
                                            # 'v ~ self_LS_mean + self_LS_std + partner_PC1_std + delta_self_force_prev + delta_partner_force_prev',
                                            # 'v ~ self_LS_mean + self_LS_std +  partner_PC1_std + delta_partner_force_withself',
                                            'v ~ self_LS_mean + self_LS_std + partner_PC1_mean + partner_PC1_std + delta_partner_force_withself',
                                            # 'v ~ partner_mean_speed + partner_speed_std + self_mean_speed',
                                            # 'v ~ partner_speed_std + self_speed_std',
                                            # 'v ~ partner_speed_std',
                                            # 'a ~ time_since_last_reward + C(condition)'
                                            # 'a ~ failed_pulls_before_reward + time_since_last_reward'
                                            # 'a ~ time_since_last_reward'
                                            # 'a ~ failed_pulls_before_reward'
                                            'z ~ prev_trial_outcome + delta_self_force_prev'
                                        ],
                                        include=['v', 'a', 'z', 't'],
                                        # depends_on={'z': ['prev_trial_outcome']}
                                    )
    
    
    # Run MCMC sampling
    print(f"Sampling HDDM with {samples} samples, {burn} burn-in, {thin} thinning...")
    # Modified this line: Removed 'thin' and used proper args for hddm 0.8.0 / PyMC3 API

    #
    m_nogaze = model_nogaze.sample(samples, burn=burn, 
                             dbname=f'traces_{animal_id}.db', db='pickle') # Saves traces to a file per animal
    
    print(f"\n--- HDDM Sampling Complete for {animal_id} ---")
    
    # Print summary of parameters
    print(f"\n--- HDDM Parameter Summary for {animal_id} ---")
    
    # model.print_stats()

    # Optional: Plot posteriors (can be slow for many parameters)
    # model.plot_posteriors()
    # plt.show()

    return model_nogaze
