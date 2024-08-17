import os
import argparse
import subprocess
import logging

from modal import Image, App, method, enter

app = App("colabfold")

image = (
    Image
    # .from_registry("nvidia/cuda:12.4.99-runtime-ubuntu22.04", add_python="3.11")
    .debian_slim(python_version="3.11")
    .micromamba(python_version="3.11")
    .apt_install("wget", "git", "curl")
    # .pip_install(
    #     "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold",
    #     gpu="a100",
    # )
    # .micromamba_install(
    #     "kalign2=2.04", "hhsuite=3.3.0", "pdbfixer", channels=["conda-forge", "bioconda"],
    #     gpu="a100",
    # )
    # .run_commands(
    #     'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
    #     gpu="a100",
    # )
    .run_commands('wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh')
    .run_commands('bash install_colabbatch_linux.sh', gpu="a100",)
    # .run_commands('export PATH="/localcolabfold/colabfold-conda/bin:$PATH"')
)



@app.cls(image=image, gpu='a100', timeout=2400)
class LocalColabFold:
    @enter()
    def setup(self):
        # Set up the environment when the container starts
        os.environ["PATH"] = "/localcolabfold/colabfold-conda/bin:" + os.environ["PATH"]

    @method()
    def fold(self, name: str, sequence: str, num_recycles: int = 1, **kwargs):
        input_fp = "/tmp/input.fasta"
        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        
        
        # Write name and sequence to a fasta file
        with open(input_fp, "w") as f:
            f.write(f">{name}\n{sequence}")
        
        cmd = [
            "colabfold_batch",
            "--num-recycle", str(num_recycles),
            "--model-type", "alphafold2_multimer_v3",
            "--zip",
        ]
        
        # Add additional kwargs to the command
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        cmd.extend([input_fp, out_dir])
        
        logging.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info(f"Command output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed with error: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise
        
        # Find and return the zip result
        try:
            zip_file = next(f for f in os.listdir(out_dir) if f.endswith(".zip"))
            with open(os.path.join(out_dir, zip_file), "rb") as g:
                return g.read()
        except StopIteration:
            logging.error(f"No zip file found in {out_dir}")
            logging.error(f"Directory contents: {os.listdir(out_dir)}")
            raise FileNotFoundError(f"No zip file found in {out_dir}")

@app.function()
def main(name: str, sequence: str, num_recycles: int = 1):
    lcf = LocalColabFold()
    return lcf.fold.remote(name, sequence, num_recycles)