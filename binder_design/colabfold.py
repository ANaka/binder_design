import os
import argparse
import subprocess
import logging
from typing import Dict
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
    def fold(self, sequences: Dict[str, str], **kwargs):
        input_fp = "/tmp/input.fasta"
        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        
        # Write all sequences to a single fasta file
        with open(input_fp, "w") as f:
            for name, sequence in sequences.items():
                f.write(f">{name}\n{sequence}\n")
        
        cmd = ["colabfold_batch"]
        
        # Handle arguments
        for key, value in kwargs.items():
            key = key.replace('_', '-')
            if isinstance(value, bool):
                # Handle flags
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                # Handle value arguments
                cmd.extend([f"--{key}", str(value)])
        
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
def main(
    sequences: Dict[str, str],
    num_recycles: int = 1,
    model_type: str = "alphafold2_multimer_v3",
    num_models: int = 5,
    max_msa: str = None,
    use_templates: bool = False,
    amber: bool = False,
    use_gpu_relax: bool = False,
    zip_results: bool = True,
    msa_mode: str = "mmseqs2_uniref_env",
    recycle_early_stop_tolerance: float = None,
    num_ensemble: int = 1,
    use_dropout: bool = False,
    relax_max_iterations: int = 2000,
    relax_tolerance: float = 2.39,
    relax_stiffness: float = 10.0,
    relax_max_outer_iterations: int = 3,
    rank: str = "auto",
    **kwargs
):
    lcf = LocalColabFold()
    return lcf.fold.remote(
        sequences=sequences,
        num_recycles=num_recycles,
        model_type=model_type,
        num_models=num_models,
        max_msa=max_msa,
        templates=use_templates,
        amber=amber,
        use_gpu_relax=use_gpu_relax,
        zip=zip_results,
        msa_mode=msa_mode,
        recycle_early_stop_tolerance=recycle_early_stop_tolerance,
        num_ensemble=num_ensemble,
        use_dropout=use_dropout,
        relax_max_iterations=relax_max_iterations,
        relax_tolerance=relax_tolerance,
        relax_stiffness=relax_stiffness,
        relax_max_outer_iterations=relax_max_outer_iterations,
        rank=rank,
        **kwargs
    )