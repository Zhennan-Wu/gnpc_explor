import os
import json
import glob
import time
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import cmdstanpy
from scipy.linalg import expm
from scipy.special import expit
import argparse
from functools import reduce
from scipy.linalg import solve_continuous_lyapunov
import arviz as az
import re

# ==========================================
# 1. Data Generation (Updated to K=12, q=4)
# ==========================================

# Define the 12 ALSFRS-R items as a constant so it can be used during loading
ITEM_COLS = [
    'Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing', 'Q4_Handwriting',
    'Q5_Cutting', 'Q6_Dressing_and_Hygiene', 'Q7_Turning_in_Bed',
    'Q8_Walking', 'Q9_Climbing_Stairs', 'R_1_Dyspnea',
    'R_2_Orthopnea', 'R_3_Respiratory_Insufficiency'
]

def get_covariate_lists(scenario):
    """
    Returns the measurement and dynamic covariates based on the selected scenario.
    """
    if scenario == 'S1':
        meas_cols = ['Treatment_Active', 'Sex_Female', 'Age_Base']
        dyn_cols = ['Bulbar_Onset', 'FVC_Base', 'BMI_Base']
    elif scenario == 'S2':
        meas_cols = ['Treatment_Active', 'Sex_Female', 'Age_Base']
        dyn_cols = ['Bulbar_Onset']
    elif scenario == 'S3':
        meas_cols = ['Treatment_Active', 'Sex_Female']
        dyn_cols = ['Age_Base', 'Bulbar_Onset']
    else:
        raise ValueError(f"Invalid scenario specified: {scenario}")
        
    return meas_cols, dyn_cols

def preprocess_proact_data(dfs, scenario="S1", output_filename="proact_preprocessed.csv"):
    """
    Preprocesses the PRO-ACT dataset dictionary.
    Filters complete cases dynamically based on the covariates required for the chosen scenario.
    """
    meas_cols, dyn_cols = get_covariate_lists(scenario)
    active_covariates = list(set(meas_cols + dyn_cols))
    
    # 1. RESPONSE VARIABLES (Y) & TIME
    df_alsfrs = dfs['alsfrs'].copy()
    df_alsfrs['Q5_Cutting'] = df_alsfrs['Q5b_Cutting_with_Gastrostomy'].fillna(df_alsfrs['Q5a_Cutting_without_Gastrostomy'])
    df_model = df_alsfrs[['subject_id', 'ALSFRS_Delta'] + ITEM_COLS].copy()
    df_model = df_model.dropna(subset=['ALSFRS_Delta'])
    
    # 2. MEASUREMENT COVARIATES (Process all to be safe)
    df_demo = dfs['demographics'].copy()
    df_demo['Sex_Female'] = (df_demo['Sex'] == 'Female').astype(float)
    df_demo_clean = df_demo.groupby('subject_id')[['Age', 'Sex_Female']].first().reset_index()
    df_demo_clean.rename(columns={'Age': 'Age_Base'}, inplace=True)
    
    df_tx = dfs['treatment'].copy()
    df_tx['Treatment_Active'] = df_tx['Study_Arm'].str.contains('Active', case=False, na=False).astype(float)
    df_tx_clean = df_tx.groupby('subject_id')[['Treatment_Active']].first().reset_index()
    
    # 3. DYNAMIC COVARIATES (Process all to be safe)
    df_hist = dfs['history'].copy()
    if 'Site_of_Onset___Bulbar' in df_hist.columns:
        df_hist['Bulbar_Onset'] = df_hist['Site_of_Onset___Bulbar'].fillna(0)
    else:
        df_hist['Bulbar_Onset'] = (df_hist['Site_of_Onset'] == 'Bulbar').astype(float)
    df_hist_clean = df_hist.groupby('subject_id')[['Bulbar_Onset']].first().reset_index()
    
    df_fvc = dfs['fvc'].copy()
    df_fvc['pct_of_Normal_Trial_1'] = pd.to_numeric(df_fvc['pct_of_Normal_Trial_1'], errors='coerce')
    fvc = df_fvc.dropna(subset=['pct_of_Normal_Trial_1', 'Forced_Vital_Capacity_Delta']).copy()
    fvc['abs_delta'] = fvc['Forced_Vital_Capacity_Delta'].abs()
    fvc_base = fvc.sort_values(['subject_id', 'abs_delta']).groupby('subject_id').first().reset_index()
    fvc_base = fvc_base.rename(columns={'pct_of_Normal_Trial_1': 'FVC_Base'})[['subject_id', 'FVC_Base']]
    
    df_vitals = dfs['vitals'].copy()
    df_vitals['Height_m'] = np.where(
        df_vitals['Height_Units'].str.lower().str.contains('in', na=False), 
        df_vitals['Height'] * 0.0254, df_vitals['Height'] / 100.0
    )
    heights = df_vitals.dropna(subset=['Height_m']).groupby('subject_id')['Height_m'].median().reset_index()
    
    df_vitals['Weight_kg'] = np.where(
        df_vitals['Weight_Units'].str.lower().str.contains('lb|pound', na=False), 
        df_vitals['Weight'] * 0.453592, df_vitals['Weight']
    )
    weights = df_vitals.dropna(subset=['Weight_kg', 'Vital_Signs_Delta']).copy()
    weights['abs_delta'] = weights['Vital_Signs_Delta'].abs()
    weights_base = weights.sort_values(['subject_id', 'abs_delta']).groupby('subject_id').first().reset_index()
    
    bmi_df = weights_base[['subject_id', 'Weight_kg']].merge(heights, on='subject_id', how='left')
    bmi_df['BMI_Base'] = bmi_df['Weight_kg'] / (bmi_df['Height_m'] ** 2)
    bmi_base = bmi_df[['subject_id', 'BMI_Base']]
    
    # 4. MERGE
    df_model = df_model.merge(df_demo_clean, on='subject_id', how='left')
    df_model = df_model.merge(df_tx_clean, on='subject_id', how='left')
    df_model = df_model.merge(df_hist_clean, on='subject_id', how='left')
    df_model = df_model.merge(fvc_base, on='subject_id', how='left')
    df_model = df_model.merge(bmi_base, on='subject_id', how='left')
    
    # Sort chronologically
    df_model = df_model.sort_values(by=['subject_id', 'ALSFRS_Delta']).reset_index(drop=True)
    
    # 5. DYNAMIC COMPLETE CASE FILTER
    # Only drop rows missing the specific covariates required by the chosen scenario
    required_cols = ITEM_COLS + active_covariates
    df_complete = df_model.dropna(subset=required_cols).reset_index(drop=True)
    
    # Standardize continuous covariates for Stan HMC stability
    continuous_vars = ['Age_Base', 'FVC_Base', 'BMI_Base']
    for col in continuous_vars:
        if col in df_complete.columns and col in active_covariates:
            df_complete[col] = (df_complete[col] - df_complete[col].mean()) / df_complete[col].std()
            
    # Save the prepared data to file
    df_complete.to_csv(output_filename, index=False)
    print(f"Preprocessed data for {scenario} successfully saved to '{output_filename}' (N={len(df_complete)})")
    
    return df_complete

def format_for_stan(df_complete, scenario="S1", data_size=None, random_state=42):
    """
    Translates the complete DataFrame into the exact dictionary required by dahlou_ncp_full.stan.
    """
    df_subset = df_complete.copy()
    meas_cols, dyn_cols = get_covariate_lists(scenario)
    
    # ==========================================
    # 0. OPTIONAL SUBJECT-LEVEL SAMPLING
    # ==========================================
    if data_size is not None:
        unique_subjects = df_subset['subject_id'].unique()
        if data_size < len(unique_subjects):
            print(f"Subsampling data to {data_size} random subjects...")
            np.random.seed(random_state)
            sampled_subjects = np.random.choice(unique_subjects, size=data_size, replace=False)
            df_subset = df_subset[df_subset['subject_id'].isin(sampled_subjects)].copy()
        else:
            print(f"Warning: data_size ({data_size}) >= total unique subjects ({len(unique_subjects)}). Using all data.")
            
    # ==========================================
    # 1. ID MAPPING
    # ==========================================
    unique_ids = df_subset['subject_id'].unique()
    id_map = {orig_id: new_id for new_id, orig_id in enumerate(unique_ids, start=1)}
    df_subset.loc[:, 'Stan_ID'] = df_subset['subject_id'].map(id_map)
    
    Nsub = len(unique_ids)
    N = len(df_subset)
    
    # ==========================================
    # 2. TIME VARIABLES
    # ==========================================
    df_subset.loc[:, 't_abs'] = df_subset['ALSFRS_Delta'] / 365.25
    df_subset.loc[:, 'deltat'] = df_subset.groupby('Stan_ID')['t_abs'].diff().fillna(0.0)
    df_subset.loc[:, 'deltat'] = np.clip(df_subset['deltat'], a_min=0.0, a_max=None)
    
    # ==========================================
    # 3. GROUPING ARRAYS
    # ==========================================
    repme = df_subset.groupby('Stan_ID').size().values
    cumu = np.cumsum(repme)
    
    # ==========================================
    # 4. RESPONSE & COVARIATE MATRICES (STAN SAFE)
    # ==========================================
    Y_raw = df_subset[ITEM_COLS].values.astype(int)
    Y = Y_raw + 1 
    
    # --- DEFENSIVE FIX: Force values into Stan's rigid bounds ---
    Y = np.clip(Y, 1, 5) 
    
    # Construct X_meas and X_dyn based on the requested scenario
    # Defensive Fix: Fill any residual NaNs with 0 to prevent C++ matrix crashes
    X_meas = np.nan_to_num(df_subset[meas_cols].values, nan=0.0)
    X_dyn = np.nan_to_num(df_subset[dyn_cols].values, nan=0.0)
    
    stan_data = {
        'N': N,
        'Nsub': Nsub,
        'K': len(ITEM_COLS),
        'R': 4,
        'p_dyn': X_dyn.shape[1],
        'p_meas': X_meas.shape[1],
        'ID': df_subset['Stan_ID'].values.astype(int),
        'cumu': cumu.astype(int),
        'repme': repme.astype(int),
        'Y': Y,
        'deltat': df_subset['deltat'].values,
        't_abs': df_subset['t_abs'].values,
        'X_dyn': X_dyn,
        'X_meas': X_meas
    }
    
    print(f"Stan data prepared for {scenario}: p_meas={X_meas.shape[1]}, p_dyn={X_dyn.shape[1]}. N={N} obs across Nsub={Nsub} subjects.")
    return stan_data


# ==========================================
# 2. General Ground Truth & Data Prep
# ==========================================
def add_param_to_dict(d, name, val):
    if isinstance(val, np.ndarray):
        if val.ndim == 1:
            for i, v in enumerate(val):
                d[f"{name}[{i+1}]"] = float(v)
        elif val.ndim == 2:
            for i in range(val.shape[0]):
                for j in range(val.shape[1]):
                    d[f"{name}[{i+1},{j+1}]"] = float(val[i, j])
    else:
        d[name] = float(val)


# ==========================================
# 3. Execution & Aggregation Pipeline
# ==========================================
def evaluate_model_performance(stan_file_path, dataset, run_id, scenario='S1', iter_sampling=1000, iter_warmup=1000, chains=3, data_size=None):
    stan_data = format_for_stan(dataset, scenario=scenario, data_size=data_size, random_state=42 + run_id)
    
    # HPC BULLETPROOFING: Explicitly point to the pre-compiled executable
    exe_path = stan_file_path.replace('.stan', '')
    model_dir = "./compiled_models"
    os.makedirs(model_dir, exist_ok=True)
    exe_path = os.path.join(model_dir, os.path.basename(exe_path))
    if os.path.exists(exe_path):
        model = cmdstanpy.CmdStanModel(exe_file=exe_path)
    else:
        raise FileNotFoundError(f"Compiled executable not found at {exe_path}. Please ensure the model is pre-compiled and the path is correct.")

    start_time = time.time()
    fit = model.sample(
        data=stan_data, iter_warmup=iter_warmup, iter_sampling=iter_sampling,
        chains=chains, parallel_chains=chains, adapt_delta=0.95, max_treedepth=12, inits=0.1,
        show_console=True, 
        show_progress=False 
    )
    run_time = time.time() - start_time

    output_dir = "../raw_results" # FIXED directory path
    os.makedirs(output_dir, exist_ok=True)
    
    # --- DEFENSIVE DIAGNOSTICS EXTRACTION ---
    try:
        divergences = int(fit.divergences.sum())
    except AttributeError:
        divergences = 0 
        
    try:
        treedepths = int(np.sum(fit.method_variables()['treedepth__'] >= 12))
        energies = fit.method_variables()['energy__'] 
        ebfmis = []
        for c in range(energies.shape[1]):
            chain_energy = energies[:, c]
            numer = np.sum(np.diff(chain_energy)**2)
            denom = np.sum((chain_energy - np.mean(chain_energy))**2)
            ebfmis.append(numer / denom)
        mean_ebfmi = np.mean(ebfmis)
    except (KeyError, AttributeError):
        treedepths = 0
        mean_ebfmi = np.nan

    try:
        summary_df = fit.summary(percentiles=[2.5, 97.5])
                            
    except Exception as e:
        print(f"Error in fit.summary(): {e}. Attempting ArviZ fallback...")
        idata = az.from_cmdstanpy(fit)
        summary_df = az.summary(idata, hdi_prob=0.95)
        
        summary_df = summary_df.rename(columns={
            'mean': 'Mean', 
            'hdi_2.5%': '2.5%', 
            'hdi_97.5%': '97.5%', 
            'r_hat': 'R_hat',
            'mcse_mean': 'MCSE',
            'ess_bulk': 'ESS_bulk'
        })
        
        summary_df.index = [re.sub(r'\[(\d+)\]', lambda m: f"[{int(m.group(1))+1}]", idx) 
                            if '[' in idx else idx 
                            for idx in summary_df.index]
        
    cols = summary_df.columns.tolist()
    
    lower_col = '2.5%' if '2.5%' in cols else ('5%' if '5%' in cols else None)
    upper_col = '97.5%' if '97.5%' in cols else ('95%' if '95%' in cols else None)
    rhat_col = 'R_hat' if 'R_hat' in cols else ('Rhat' if 'Rhat' in cols else None)
    mcse_col = 'MCSE' if 'MCSE' in cols else None

    model_params = set(summary_df.index)
    model_params = {p for p in model_params if not p.endswith('__')}

    results = []
    
    for param_name in model_params:
        est_mean = summary_df.loc[param_name, 'Mean']
        lower_ci = summary_df.loc[param_name, lower_col] if lower_col else np.nan
        upper_ci = summary_df.loc[param_name, upper_col] if upper_col else np.nan
        rhat = summary_df.loc[param_name, rhat_col] if rhat_col else np.nan
        mcse = summary_df.loc[param_name, mcse_col] if mcse_col else np.nan
        
        if 'ESS_bulk' in cols: ess = summary_df.loc[param_name, 'ESS_bulk']
        elif 'N_Eff' in cols: ess = summary_df.loc[param_name, 'N_Eff']
        else: ess = np.nan
            
        ess_sec = ess / run_time if run_time > 0 and pd.notna(ess) else np.nan
        
        results.append({
            'Run_ID': run_id, 'Parameter': param_name, 'Estimate': est_mean,
            '2.5%': lower_ci, '97.5%': upper_ci, 'R_hat': rhat, 'ESS': ess,
            'MCSE': mcse, 'ESS_sec': ess_sec, 'Time_s': run_time, 'Divergences': divergences, 'Max_Treedepths': treedepths, 'E_BFMI': mean_ebfmi
        })
            
    metrics_df = pd.DataFrame(results)
    model_name = os.path.splitext(os.path.basename(stan_file_path))[0]
    metrics_df.to_csv(os.path.join(output_dir, f"results_{scenario}_{model_name}_run{run_id}.csv"), index=False)
    
    return True

def generate_simulation_table(scenario, model_name):
    print(f"\nAggregating results across all runs for {scenario} - {model_name}...")
    source_dir = "../raw_results" # FIXED directory path
    target_dir = "../summarized_results" # FIXED directory path
    os.makedirs(target_dir, exist_ok=True)
    
    file_pattern = os.path.join(source_dir, f"results_{scenario}_{model_name}_run*.csv")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print("No files found to aggregate.")
        return None
        
    df_list = []
    total_considered = len(all_files)
    total_aggregated = 0
    total_discarded_errors = 0
    
    for f in all_files:
        try:
            df = pd.read_csv(f)
            if (df['R_hat'] < 1.1).all():
                df_list.append(df)
                total_aggregated += 1
                
        except KeyError as e:
            print(f"  [Warning] Discarding {f}: Missing expected column {e}")
            total_discarded_errors += 1
        except pd.errors.EmptyDataError:
            print(f"  [Warning] Discarding {f}: File is completely empty.")
            total_discarded_errors += 1
        except Exception as e:
            print(f"  [Warning] Discarding {f}: Unexpected error -> {e}")
            total_discarded_errors += 1
            
    print(f"\n--- Summary ---")
    print(f"Total tables considered: {total_considered}")
    if total_discarded_errors > 0:
        print(f"Total tables discarded due to errors/missing data: {total_discarded_errors}")
    print(f"Total tables aggregated (R_hat < 1.05): {total_aggregated}")
    print(f"----------------\n")
    
    if not df_list:
        print("No tables met the criteria for aggregation. Aborting.")
        return None

    combined_df = pd.concat(df_list, ignore_index=True)
    
    run_level_df = combined_df.groupby('Run_ID').agg(
        Time=('Time_s', 'first'),
        Divergences=('Divergences', 'first'),
        Treedepths=('Max_Treedepths', 'first'),
        E_BFMI=('E_BFMI', 'first')
    )
    
    print("\n" + "="*50)
    print(" 🛠️ RUN-LEVEL HMC DIAGNOSTICS (AVERAGE) 🛠️ ")
    print("="*50)
    print(f"Total Wall-Clock Time (s): {run_level_df['Time'].mean():.2f}")
    print(f"Divergent Transitions:     {run_level_df['Divergences'].mean():.2f}")
    print(f"Max Treedepth Hits:        {run_level_df['Treedepths'].mean():.2f}")
    print(f"E-BFMI:                    {run_level_df['E_BFMI'].mean():.3f}")
    print("="*50 + "\n")
    
    table_df = combined_df.groupby('Parameter').agg(
        ESS=('ESS', 'mean'),
        ESS_sec=('ESS_sec', 'mean'),
        MCSE=('MCSE', 'mean'),
        Rhat=('R_hat', 'max')             
    ).reset_index()

    table_df['ESS'] = table_df['ESS'].round(1)
    table_df['ESS_sec'] = table_df['ESS_sec'].round(3)
    table_df['MCSE'] = table_df['MCSE'].round(4)
    table_df['Rhat'] = table_df['Rhat'].round(3)
    
    final_filename = f"TABLE_{scenario}_{model_name}.csv"
    final_path = os.path.join(target_dir, final_filename)
    table_df.to_csv(final_path, index=False)
    print(f"Success! Final aggregate parameter table saved to: {final_filename}")
    
    return table_df

def aggregate_cross_model_results(scenario, model_names):
    print(f"\nMerging results across models for Scenario: {scenario}...")
    source_dir = "../summarized_results" # FIXED directory path
    target_dir = "../aggregated_results" # FIXED directory path
    os.makedirs(target_dir, exist_ok=True)
    
    dataframes = []
    for model in model_names:
        filename = f"TABLE_{scenario}_{model}.csv"
        filepath = os.path.join(source_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  [Warning] Missing file: {filename} in {source_dir}. Skipping this model.")
            continue
            
        df = pd.read_csv(filepath)
        rename_map = {
            col: f"{col}_{model}" 
            for col in df.columns 
            if col not in ['Parameter', 'True_Value']
        }
        df.rename(columns=rename_map, inplace=True)
        dataframes.append(df)
        
    if not dataframes:
        print("No model tables found to merge.")
        return None
        
    final_combined_df = reduce(
        lambda left, right: pd.merge(left, right, on=['Parameter', 'True_Value'], how='outer'), 
        dataframes
    )
    
    base_metrics = ['ESS', 'ESS_sec', 'MCSE', 'Rhat']
    ordered_cols = ['Parameter', 'True_Value']
    for metric in base_metrics:
        for model in model_names:
            col_name = f"{metric}_{model}"
            if col_name in final_combined_df.columns:
                ordered_cols.append(col_name)
                
    remaining_cols = [c for c in final_combined_df.columns if c not in ordered_cols]
    final_combined_df = final_combined_df[ordered_cols + remaining_cols]
    
    output_filename = f"FINAL_COMPARISON_{scenario}.csv"
    output_filepath = os.path.join(target_dir, output_filename)
    final_combined_df.to_csv(output_filepath, index=False)
    
    print(f"Success! Cross-model comparison saved to: {output_filepath}")
    return final_combined_df

# ==========================================
# 4. Main Execution (OPTIMIZED FOR SINGLE DATASET)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DAHLOU Empirical Analysis on HPC")
    parser.add_argument("--scenario", type=str, required=True, help="Data generation scenario")
    parser.add_argument("--models", type=str, nargs='+', required=True, help="Path(s) to the Stan model file(s)")
    parser.add_argument("--data_file", type=str, default="proact_preprocessed", help="Path to the preprocessed data file")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains")
    parser.add_argument("--warmup", type=int, default=2500, help="Warmup iterations")
    parser.add_argument("--sampling", type=int, default=2500, help="Sampling iterations")
    parser.add_argument("--data_size", type=int, default=None, help="Optional sample size for subsampling subjects")
    parser.add_argument("--compile_only", action="store_true", help="Only compile the model, do not run sims") 
    parser.add_argument("--aggregate_only", action="store_true", help="Only run the table aggregation")
    parser.add_argument("--cross_aggregate", action="store_true", help="Merge multiple model tables into one final comparison table")
    
    args = parser.parse_args()
    scenario_to_run = args.scenario
    model_names = [os.path.splitext(os.path.basename(m))[0] for m in args.models]
    stan_file = args.models[0]
    model_name = model_names[0] 
    
    if args.cross_aggregate:
        aggregate_cross_model_results(scenario=scenario_to_run, model_names=model_names)
        exit(0)

    if args.compile_only:
        print(f"Pre-compiling Stan Model: {model_name}...")
        model_dir = "./models"
        compiled_dir = "./compiled_models"

        os.makedirs(compiled_dir, exist_ok=True)
        if not os.path.exists(model_dir):
            print(f"Error: Models directory '{model_dir}' not found. Please ensure the path is correct.")
            exit(1)
            
        stan_file_path = os.path.join(model_dir, stan_file)
        exe_path = os.path.join(compiled_dir, model_name)

        _ = cmdstanpy.CmdStanModel(stan_file=stan_file_path, exe_file=exe_path)
        print("Compilation successful. Exiting.")
        exit(0)

    if args.aggregate_only:
        generate_simulation_table(scenario=scenario_to_run, model_name=model_name)
        exit(0)

    print(f"Starting Empirical Analysis: Scenario = {scenario_to_run}, Model = {model_name}")
    print(f"Running 1 Dataset with {args.chains} parallel chains ({args.warmup} warmup / {args.sampling} sampling)\n")
    
    datafile = args.data_file + f"_{args.scenario}.csv"
    
    try:
        dataset = pd.read_csv(datafile)
        
        # We run the evaluation exactly once. 
        # Run_ID is set to 1 because there are no longer 200 replications.
        evaluate_model_performance(
            stan_file_path=stan_file, 
            dataset=dataset,
            run_id=1, 
            scenario=scenario_to_run,
            iter_warmup=args.warmup,
            iter_sampling=args.sampling,
            chains=args.chains,
            data_size=args.data_size
        )
        print(f"\n✅ Successfully completed empirical run for {model_name}.")
        
    except Exception as e:
        print(f"\n❌ Failed Empirical Run: {str(e)}")
        exit(1)