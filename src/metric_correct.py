import os
import glob
import pandas as pd
import numpy as np

def batch_correct_simulations(directory="../raw_results"):
    """
    Scans a directory for simulation CSV files, corrects the parameter signs 
    for the specific variables affected by software parameterization, 
    and saves new corrected CSV files into a 'corrected_results' folder 
    in the parent directory.
    """
    
    # 1. Define and create the output directory in the parent folder
    current_abs_dir = os.path.abspath(directory)
    parent_dir = os.path.dirname(current_abs_dir)
    output_dir = os.path.join(parent_dir, "corrected_results")
    
    # Create the folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ready: {output_dir}\n")

    # 2. Find all CSV files in the current directory
    search_path = os.path.join(directory, "*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print("No eligible CSV files found to process.")
        return

    processed_count = 0
    skipped_count = 0
    errors = []

    for file in csv_files:
        try:
            # Read the data safely
            df = pd.read_csv(file)
            
            # Validation: Make sure this is actually a simulation result file
            required_cols = {'Parameter', 'Estimate', '2.5%', '97.5%', 'True_Value', 'Rbias', 'MSE'}
            if not required_cols.issubset(set(df.columns)):
                skipped_count += 1
                continue
                
            # Create the mask 
            mask = df['Parameter'].astype(str).str.match(r'^theta[123]$|^beta')
            
            if mask.sum() > 0:
                # Extract variables
                E = df.loc[mask, 'Estimate'].astype(float)
                L = df.loc[mask, '2.5%'].astype(float)
                U = df.loc[mask, '97.5%'].astype(float)
                T = df.loc[mask, 'True_Value'].astype(float)
                
                # Flip Estimate and correctly swap credible intervals
                df.loc[mask, 'Estimate'] = -E
                df.loc[mask, '2.5%'] = -U
                df.loc[mask, '97.5%'] = -L
                
                # Recalculate Evaluation Metrics
                new_E = df.loc[mask, 'Estimate'].astype(float)
                
                # Robustly calculate Rbias (prevents division by zero)
                df.loc[mask, 'Rbias'] = np.where(T != 0, (new_E - T) / T, np.nan)
                df.loc[mask, 'MSE'] = (new_E - T) ** 2
                
                # Recalculate Coverage (Only if the column exists in the file)
                if 'Coverage' in df.columns:
                    new_L = df.loc[mask, '2.5%'].astype(float)
                    new_U = df.loc[mask, '97.5%'].astype(float)
                    df.loc[mask, 'Coverage'] = ((T >= new_L) & (T <= new_U)).astype(int)
                
            # 3. Save to the new parent-level directory with the original filename
            base_name = os.path.basename(file)
            new_filepath = os.path.join(output_dir, base_name)
            
            # Save out the corrected dataframe
            df.to_csv(new_filepath, index=False)
            processed_count += 1
            print(f"[{processed_count}] Saved to: ../corrected_results/{base_name}")
            
        except pd.errors.EmptyDataError:
            print(f"Skipped {os.path.basename(file)} (File is empty).")
            skipped_count += 1
        except Exception as e:
            errors.append((file, str(e)))

    # 4. Final Summary
    print("\n--- Batch Correction Complete ---")
    print(f"Successfully Processed: {processed_count} files")
    print(f"Skipped (Missing columns/Empty): {skipped_count} files")
    
    if errors:
        print(f"\nErrors encountered in {len(errors)} files:")
        for file, error in errors:
            print(f" - {os.path.basename(file)}: {error}")

# Run the function on the current folder
if __name__ == "__main__":
    batch_correct_simulations()