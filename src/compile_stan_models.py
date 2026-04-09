import os
import glob
import shutil
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
            # 1. Compile the model by providing ONLY the stan_file.
            # cmdstanpy will compile it and drop the executable next to the .stan file.
            model = cmdstanpy.CmdStanModel(stan_file=stan_path)
            
            # 2. Move the compiled executable to the target directory
            # model.exe_file holds the path to the newly compiled executable
            shutil.move(model.exe_file, exe_path)
            
            # 3. (Optional) Move or delete the generated C++ .hpp file 
            # to keep your source directory clean.
            hpp_file = os.path.splitext(stan_path)[0] + ".hpp"
            if os.path.exists(hpp_file):
                shutil.move(hpp_file, os.path.join(COMPILED_DIR, model_name + ".hpp"))

            print(f"  ✔ Success -> {exe_path}\n")

        except Exception as e:
            print(f"  ✘ Failed to compile {model_name}")
            print(f"    Error: {e}\n")

    print("Done.")

if __name__ == "__main__":
    main()