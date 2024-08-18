import os
import argparse
import tempfile
import shutil
import subprocess
import logging
from binder_design import TEMPLATE_A3M_PATH
from modal import Image, App, method, enter, Dict

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
    # .pip_install("grpclib")
    # .run_commands('export PATH="/localcolabfold/colabfold-conda/bin:$PATH"')
)


def generate_a3m_files(binder_sequences, output_folder, template_a3m_path=TEMPLATE_A3M_PATH, target_sequence=None):
    os.makedirs(output_folder, exist_ok=True)
    
    with open(template_a3m_path, 'r') as template_file:
        template_lines = template_file.readlines()
    
    # Extract the target sequence from the template if target_sequence is None
    if target_sequence is None:
        template_sequence = template_lines[2].strip()
        binder_length = int(template_lines[0].split(',')[0].strip('#'))
        target_sequence = template_sequence[binder_length:]
    
    for name, binder_seq in binder_sequences.items():
        output_path = os.path.join(output_folder, f"{name}.a3m")
        
        with open(output_path, 'w') as output_file:
            # Write the header lines
            output_file.writelines(template_lines[:2])
            
            # Write the concatenated sequences
            output_file.write(f"{binder_seq}{target_sequence}\n")
            
            # Write the rest of the template file
            output_file.writelines(template_lines[3:])
    
    return output_folder

@app.cls(image=image, gpu='a100', timeout=2400)
class LocalColabFold:
    @enter()
    def setup(self):
        # Set up the environment when the container starts
        os.environ["PATH"] = "/localcolabfold/colabfold-conda/bin:" + os.environ["PATH"]

    @method()
    def fold(self, sequences=None, binder_sequences=None, template_a3m_path=None, target_sequence=None, **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            if template_a3m_path is None:
                # Sequence-based approach
                input_file = os.path.join(temp_dir, "input.fasta")
                with open(input_file, 'w') as f:
                    for name, seq in sequences.items():
                        f.write(f">{name}\n{seq}\n")
                input_path = input_file
            else:
                # A3M-based approach
                input_path = generate_a3m_files(
                    binder_sequences=binder_sequences,
                    output_folder=temp_dir,
                    template_a3m_path=template_a3m_path,
                    target_sequence=target_sequence
                )

            out_dir = os.path.join(temp_dir, "output")
            os.makedirs(out_dir, exist_ok=True)

            cmd = ["colabfold_batch", input_path, out_dir]
            
            # Handle arguments
            for key, value in kwargs.items():
                key = key.replace('_', '-')
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                elif value is not None:
                    cmd.extend([f"--{key}", str(value)])
            
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
def fold_sequences(
    sequences,
    num_recycle: int = 1,
    model_type: str = "alphafold2_multimer_v3",
    zip_results: bool = True,
    msa_mode: str = "mmseqs2_uniref_env",
    num_models: int = 3,
    max_msa: str = None,
    use_templates: bool = False,
    amber: bool = False,
    use_gpu_relax: bool = False,
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
        num_recycle=num_recycle,
        model_type=model_type,
        zip=zip_results,
        msa_mode=msa_mode,
        num_models=num_models,
        max_msa=max_msa,
        templates=use_templates,
        amber=amber,
        use_gpu_relax=use_gpu_relax,
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

@app.function()
def fold_a3m(
    binder_sequences: dict,
    template_a3m_path: str = TEMPLATE_A3M_PATH,
    target_sequence: str = None,
    num_recycle: int = 1,
    model_type: str = "alphafold2_multimer_v3",
    zip_results: bool = True,
    # msa_mode: str = "mmseqs2_uniref_env",
    # num_models: int = 3,
    # max_msa: str = None,
    # use_templates: bool = False,
    # amber: bool = False,
    # use_gpu_relax: bool = False,
    # recycle_early_stop_tolerance: float = None,
    # num_ensemble: int = 1,
    # use_dropout: bool = False,
    # relax_max_iterations: int = 2000,
    # relax_tolerance: float = 2.39,
    # relax_stiffness: float = 10.0,
    # relax_max_outer_iterations: int = 3,
    # rank: str = "auto",
    # **kwargs
):
    
    lcf = LocalColabFold()
    return lcf.fold.remote(
        binder_sequences=binder_sequences,
        template_a3m_path=template_a3m_path,
        target_sequence=target_sequence,
        num_recycle=num_recycle,
        model_type=model_type,
        zip=zip_results,
        # msa_mode=msa_mode,
        # num_models=num_models,
        # max_msa=max_msa,
        # templates=use_templates,
        # amber=amber,
        # use_gpu_relax=use_gpu_relax,
        # recycle_early_stop_tolerance=recycle_early_stop_tolerance,
        # num_ensemble=num_ensemble,
        # use_dropout=use_dropout,
        # relax_max_iterations=relax_max_iterations,
        # relax_tolerance=relax_tolerance,
        # relax_stiffness=relax_stiffness,
        # relax_max_outer_iterations=relax_max_outer_iterations,
        # rank=rank,
        # **kwargs
    )

@app.local_entrypoint()
def test():
    # Test sequence-based folding
    sequences = {
        'binder': 'NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWW',
        'target': 'EGFR_SEQUENCE_HERE'
    }
    results_seq = fold_sequences.remote(sequences=sequences)

    # Test a3m-based folding with provided target sequence
    template_a3m_path = TEMPLATE_A3M_PATH
    binder_sequences = {
        "binder1": "NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWW",
        "binder2": "NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRXX"
    }
    target_sequence = "EGFR_SEQUENCE_HERE"
    results_a3m_with_target = fold_a3m.remote(
        template_a3m_path=template_a3m_path,
        binder_sequences=binder_sequences,
        target_sequence=target_sequence
    )

    # Test a3m-based folding without provided target sequence
    results_a3m_without_target = fold_a3m.remote(
        template_a3m_path=template_a3m_path,
        binder_sequences=binder_sequences
    )