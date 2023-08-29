import glob
import subprocess
from os import path

import pandas as pd
from tqdm import tqdm

# NOTE: sm_52 reflects the T4 results

data_path = "data/"
compilation_command = "clang++-10 {} -pthread --cuda-path=/usr/lib/cuda/w -gencode=arch=compute_52,code=sm_52 -std=c++11 -fno-exceptions -stdlib=libstdc++ -S -emit-llvm -O3 --cuda-device-only -o {}"

if __name__ == '__main__':

    # Read CSV containing all kernels
    kernels = pd.read_csv(data_path+'kernel_list.csv')

    # Add paths of files to compile
    kernels['kernel_path'] = [data_path+"kernels/"+str(r["Repo"])+"/"+str(r["underdirectory"]) 
                            for _, r in kernels.iterrows()]
    kernels['main_file']  = kernels['kernel_path']+"/"+"main.cu"

    # Start compiling
    num_kernels = len(glob.glob(data_path+'kernels/*/*/main.cu'))
    errors = []

    for idx, row in tqdm(kernels.iterrows(), total=len(kernels)):
        main_file_path = row['main_file']
        function_name = row['function']
        output_file_path = main_file_path.replace('main.cu', f'{function_name}.ll')
        if not path.exists(output_file_path):
            result = subprocess.run(compilation_command.format(main_file_path, output_file_path), shell=True)
            if result.returncode != 0:
                errors.append(main_file_path)

    print(f"Compiled {num_kernels - len(errors)}/{num_kernels} LS-CAT kernels")
    print(f"{len(errors)} errors at {errors}")