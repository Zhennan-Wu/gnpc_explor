import os
import glob
import json
import pandas as pd

def aggregate_simulation_results(results_dir="."):
    """
    Scans the directory for JSON summaries and CSV detail files generated 
    by the Slurm array and combines them into master DataFrames.
    """
    print(f"Scanning '{results_dir}' for simulation results...")

    # 1. Aggregate JSON Summaries
    json_files = glob.glob(os.path.join(results_dir, "summary_*.json"))
    summary_data = []

    for file in json_files:
        # Extract scenario and model from filename: summary_S1_model_name.json
        basename = os.path.basename(file)
        # Strip extension and 'summary_' prefix
        name_parts = basename.replace('.json', '').replace('summary_', '').split('_', 1)
        
        if len(name_parts) == 2:
            scenario, model_name = name_parts
            
            with open(file, 'r') as f:
                try:
                    data = json.load(f)
                    data['Scenario'] = scenario
                    data['Model'] = model_name
                    summary_data.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not read {file}")

    if summary_data:
        master_summary_df = pd.DataFrame(summary_data)
        # Reorder columns to put Scenario and Model first
        cols = ['Scenario', 'Model'] + [c for c in master_summary_df.columns if c not in ['Scenario', 'Model']]
        master_summary_df = master_summary_df[cols]
        master_summary_df.sort_values(by=['Scenario', 'Model'], inplace=True)
        
        summary_out = "master_summary_metrics.csv"
        master_summary_df.to_csv(summary_out, index=False)
        print(f"Successfully created '{summary_out}' with {len(master_summary_df)} records.")
    else:
        print("No JSON summary files found.")


    # 2. Aggregate CSV Parameter Details
    csv_files = glob.glob(os.path.join(results_dir, "results_*.csv"))
    details_data = []

    for file in csv_files:
        # Extract scenario and model from filename: results_S1_model_name.csv
        basename = os.path.basename(file)
        name_parts = basename.replace('.csv', '').replace('results_', '').split('_', 1)
        
        if len(name_parts) == 2:
            scenario, model_name = name_parts
            
            try:
                df = pd.read_csv(file)
                df.insert(0, 'Model', model_name)
                df.insert(0, 'Scenario', scenario)
                details_data.append(df)
            except Exception as e:
                print(f"Warning: Could not read {file}. Error: {e}")

    if details_data:
        master_details_df = pd.concat(details_data, ignore_index=True)
        master_details_df.sort_values(by=['Scenario', 'Model', 'Parameter'], inplace=True)
        
        details_out = "master_parameter_details.csv"
        master_details_df.to_csv(details_out, index=False)
        print(f"Successfully created '{details_out}' with {len(master_details_df)} parameter rows.")
    else:
        print("No CSV result files found.")

if __name__ == "__main__":
    aggregate_results()