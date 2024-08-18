import os
import argparse
import tempfile
import shutil
import subprocess
import logging
from binder_design import TEMPLATE_A3M_PATH
from modal import Image, App, method, enter, Dict
import io
import zipfile
import re
import json

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
    .pip_install('biopython')
    .pip_install('pandas')
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


# Manual three-letter to one-letter amino acid code conversion
aa_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def three_to_one(three_letter_code):
    return aa_dict.get(three_letter_code, 'X')







with image.imports():
    from Bio import PDB
    import numpy as np
    import pandas as pd
    

@app.cls(image=image, gpu='a100', timeout=2400, concurrency_limit=20,)
class LocalColabFold:
    @enter()
    def setup(self):
        from Bio import PDB
        import numpy as np
        import pandas as pd
        
        # Set up the environment when the container starts
        os.environ["PATH"] = "/localcolabfold/colabfold-conda/bin:" + os.environ["PATH"]

    @method()
    def fold(self, sequences=None, binder_sequences=None, template_a3m_path=None, target_sequence=None, **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Created temporary directory: {temp_dir}")
            
            if template_a3m_path is None:
                # Sequence-based approach
                input_file = os.path.join(temp_dir, "input.fasta")
                with open(input_file, 'w') as f:
                    for name, seq in sequences.items():
                        f.write(f">{name}\n{seq}\n")
                input_path = input_file
                logging.info(f"Created input FASTA file: {input_file}")
            else:
                # A3M-based approach
                input_path = generate_a3m_files(
                    binder_sequences=binder_sequences,
                    output_folder=temp_dir,
                    template_a3m_path=template_a3m_path,
                    target_sequence=target_sequence
                )
                logging.info(f"Generated A3M files in: {input_path}")

            out_dir = "output"
            os.makedirs(out_dir, exist_ok=True)
            logging.info(f"Created output directory: {out_dir}")

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
                zip_path = os.path.join(out_dir, zip_file)
                logging.info(f"Found zip file: {zip_path}")
                
                # Extract metrics and PDBs here
                all_results, pdbs = self.extract_metrics_and_pdbs(zip_path, out_dir)
                
                logging.info(f"Extracted {len(all_results)} results and {len(pdbs)} PDB files")
                
                return {
                    'zip_content': open(zip_path, 'rb').read(),
                    'results': all_results,
                    'pdbs': pdbs
                }
            except StopIteration:
                logging.error(f"No zip file found in {out_dir}")
                logging.error(f"Directory contents: {os.listdir(out_dir)}")
                raise FileNotFoundError(f"No zip file found in {out_dir}")
            
    @staticmethod
    def extract_sequence_from_pdb(pdb_content):
        parser = PDB.PDBParser()
        structure = parser.get_structure("protein", io.StringIO(pdb_content.decode('utf-8')))
        
        sequences = {'A': '', 'B': ''}
        for model in structure:
            for chain in model:
                chain_id = chain.id
                if chain_id in sequences:
                    for residue in chain:
                        if PDB.is_aa(residue):
                            sequences[chain_id] += three_to_one(residue.resname)
        
        return {
            'binder': sequences['A'],
            'target': sequences['B'],
            'binder_length': len(sequences['A']),
            'target_length': len(sequences['B']),
        }
        
    @staticmethod
    def extract_sequences(zip_ref, seq_name):
        pdb_file = next(name for name in zip_ref.namelist() if name.startswith(seq_name) and name.endswith('.pdb'))
        with zip_ref.open(pdb_file) as file:
            pdb_content = file.read()
            sequences = LocalColabFold.extract_sequence_from_pdb(pdb_content)
        return sequences
    
    @staticmethod
    def extract_scores(zip_ref, seq_name, binder_length):
        pattern = r'model_(\d+)'
        score_jsons = [name for name in zip_ref.namelist() if seq_name in name and '_scores_' in name]
        
        results = []
        for json_file in score_jsons:
            match = re.search(pattern, json_file)
            if match:
                model_number = int(match.group(1))
                
            with zip_ref.open(json_file) as file:
                data = json.load(file)
                
                plddt_array = np.array(data['plddt'])
                pae_array = np.array(data['pae'])
                
                pae_interaction = (pae_array[binder_length:, :binder_length].mean() + pae_array[:binder_length, binder_length:].mean()) / 2
                binder_plddt = plddt_array[:binder_length].mean()
                binder_pae = pae_array[:binder_length, :binder_length].mean()
                
                result = {
                    'model_number': model_number,
                    'binder_plddt': float(binder_plddt),
                    'binder_pae': float(binder_pae),
                    'pae_interaction': float(pae_interaction),
                    'ptm': data['ptm'],
                }
            results.append(result)
        
        return results
    
    @staticmethod
    def extract_metrics_and_pdbs(zip_path, output_dir):
        logging.info(f"Extracting metrics and PDBs from {zip_path}")
        all_results = []
        pdbs = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            pdb_files = [name for name in zip_ref.namelist() if name.endswith('.pdb')]
            seq_names = list(set([n.split('_unrelaxed_rank')[0] for n in pdb_files]))
            
            logging.info(f"Found {len(seq_names)} sequence names in the zip file")
            
            for seq_name in seq_names:
                sequences = LocalColabFold.extract_sequences(zip_ref, seq_name)
                binder_length = sequences['binder_length']
                target_length = sequences['target_length']
                
                logging.info(f"Extracted sequences for {seq_name}: binder length {binder_length}, target length {target_length}")
                
                scores = LocalColabFold.extract_scores(zip_ref, seq_name, binder_length)
                
                logging.info(f"Extracted {len(scores)} scores for {seq_name}")
                
                for score in scores:
                    result = {
                        'seq_name': seq_name,
                        'binder_sequence': sequences['binder'],
                        'target_sequence': sequences['target'],
                        'binder_length': binder_length,
                        'target_length': target_length,
                        'model_number': score['model_number'],
                        'binder_plddt': score['binder_plddt'],
                        'binder_pae': score['binder_pae'],
                        'pae_interaction': score['pae_interaction'],
                        'ptm': score['ptm'],
                    }
                    all_results.append(result)
                    
                    # Find and extract PDB files
                    pdb_files = [name for name in zip_ref.namelist() if name.endswith('.pdb') and seq_name in name and f"rank_{score['model_number']:03d}" in name]
                    for pdb_filename in pdb_files:
                        with zip_ref.open(pdb_filename) as pdb_file:
                            pdb_content = pdb_file.read()
                            pdb_output_path = os.path.join(output_dir, pdb_filename)
                            with open(pdb_output_path, 'wb') as f:
                                f.write(pdb_content)
                            pdbs[f"{seq_name}_model_{score['model_number']}"] = pdb_output_path
                    
                    logging.info(f"Saved PDB file: {pdb_output_path}")
                    
                    result['pdb_path'] = pdb_output_path
        
        logging.info(f"Extracted {len(all_results)} total results and {len(pdbs)} PDB files")
        return all_results, pdbs
        

@app.function(timeout=4800, gpu='a100')
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

@app.function(timeout=4800, gpu='a100',)
def fold_a3m(
    binder_sequences: dict,
    template_a3m_path: str = TEMPLATE_A3M_PATH,
    target_sequence: str = None,
    num_recycle: int = 1,
    model_type: str = "alphafold2_multimer_v3",
    zip_results: bool = True,
    # msa_mode: str = "mmseqs2_uniref_env",
    num_models: int = 2,
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
        num_models=num_models,
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
    
@app.function(timeout=4800)
def fold_and_extract(
    binder_sequences: dict, 
    template_a3m_path: str=TEMPLATE_A3M_PATH, 
    target_sequence: str = None, 
    zip_results: bool = True,
    **kwargs):
    lcf = LocalColabFold()
    result = lcf.fold.remote(
        binder_sequences=binder_sequences,
        template_a3m_path=template_a3m_path,
        target_sequence=target_sequence,
        zip=zip_results,
        **kwargs
    )
    return result['results']

@app.function(timeout=4800)
def parallel_fold_and_extract(binder_sequences: dict, template_a3m_path: str=TEMPLATE_A3M_PATH, target_sequence: str = None, batch_size: int = 10, output_dir: str = 'output', **kwargs):
    all_results = []
    all_pdbs = {}
    
    # Prepare batches
    batches = []
    for i in range(0, len(binder_sequences), batch_size):
        batch = dict(list(binder_sequences.items())[i:i+batch_size])
        batches.append((batch, template_a3m_path, target_sequence, output_dir))


    all_results = []
    for result in fold_and_extract.starmap(batches, kwargs=kwargs):
        all_results.extend(result)
    
    return all_results

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