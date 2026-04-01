import os
import glob
import cmdstanpy

MODELS_DIR = "./models"
COMPILED_DIR = "./compiled_models"

def main():
    # Ensure output directory exists
    os.makedirs(COMPILED_DIR, exist_ok=True)

    # Find all Stan files
    stan_files = glob.glob(os.path.join(MODELS_DIR, "*.stan"))

    if not stan_files:
        print(f"No Stan files found in {MODELS_DIR}")
        return

    print(f"Found {len(stan_files)} Stan model(s). Starting compilation...\n")

    for stan_path in stan_files:
        model_name = os.path.splitext(os.path.basename(stan_path))[0]
        exe_path = os.path.join(COMPILED_DIR, model_name)

        print(f"Compiling: {model_name}")

        try:
            model = cmdstanpy.CmdStanModel(
                stan_file=stan_path,
                exe_file=exe_path
            )
            print(f"  ✔ Success -> {exe_path}\n")

        except Exception as e:
            print(f"  ✘ Failed to compile {model_name}")
            print(f"    Error: {e}\n")

    print("Done.")

if __name__ == "__main__":
    main()