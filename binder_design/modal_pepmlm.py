
import os
from modal import App, Secret, gpu, Image, enter, method
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def download_models():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache
    
    logger.info("Starting model download")
    try:
        snapshot_download(repo_id="ChatterjeeLab/PepMLM-650M", repo_type="model", token=os.environ["HUGGINGFACE_TOKEN"])
        logger.info("Model download completed successfully")
        move_cache()
        logger.info("Cache moved successfully")
    except Exception as e:
        logger.error(f"Error during model download or cache move: {str(e)}")
        raise
    
image = (
    Image
    .debian_slim()
    .pip_install('uv')
    .run_commands("uv pip install  --system --compile-bytecode torch transformers==4.28 huggingface_hub pandas tqdm peft python-dotenv")
    .run_function(download_models, secrets=[Secret.from_dotenv()])
    )

app = App(name="pepmlm", image=image)

with image.imports():
    import torch
    import transformers
    import huggingface_hub
    import pandas
    import tqdm
    import peft
    from torch.distributions.categorical import Categorical
    import numpy as np
    import pandas as pd
  
@app.cls(
    container_idle_timeout=200,
    image=image,
    secrets=[Secret.from_dotenv()],
)
class PepMLM:

    @enter()
    def enter(self):
        logger.info("Initializing PepMLM")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info("Loading model and tokenizer")
        self.model = transformers.AutoModelForCausalLM.from_pretrained("ChatterjeeLab/PepMLM-650M").to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M").to(self.device)
        logger.info("Model and tokenizer loaded successfully")
        
    @method()
    def compute_pseudo_perplexity(self, target_seq, binder_seq):
        '''
        For alternative computation of PPL (in batch/matrix format), please check our github repo:
        https://github.com/programmablebio/pepmlm/blob/main/scripts/generation.py
        '''
        logger.info(f"Computing pseudo-perplexity for target sequence: {target_seq[:10]}... and binder sequence: {binder_seq}")
        sequence = target_seq + binder_seq
        tensor_input = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)
        total_loss = 0

        # Loop through each token in the binder sequence
        for i in range(-len(binder_seq)-1, -1):
            # Create a copy of the original tensor
            masked_input = tensor_input.clone()

            # Mask one token at a time
            masked_input[0, i] = self.tokenizer.mask_token_id
            # Create labels
            labels = torch.full(tensor_input.shape, -100).to(self.device)
            labels[0, i] = tensor_input[0, i]

            # Get model prediction and loss
            with torch.no_grad():
                outputs = self.model(masked_input, labels=labels)
                total_loss += outputs.loss.item()

        # Calculate the average loss
        avg_loss = total_loss / len(binder_seq)

        # Calculate pseudo perplexity
        pseudo_perplexity = np.exp(avg_loss)
        logger.info(f"Computed pseudo-perplexity: {pseudo_perplexity}")
        return pseudo_perplexity
        
    def generate_peptide_for_single_sequence(self, protein_seq, peptide_length=15, top_k=3, num_binders=4):
        logger.info(f"Generating peptides for sequence: {protein_seq[:10]}...")
        peptide_length = int(peptide_length)
        top_k = int(top_k)
        num_binders = int(num_binders)

        binders_with_ppl = []

        for i in range(num_binders):
            logger.info(f"Generating binder {i+1}/{num_binders}")
            # Generate binder
            masked_peptide = '<mask>' * peptide_length
            input_sequence = protein_seq + masked_peptide
            inputs = self.tokenizer(input_sequence, return_tensors="pt").to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
            mask_token_indices = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            logits_at_masks = logits[0, mask_token_indices]

            # Apply top-k sampling
            top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
            probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
            predicted_indices = Categorical(probabilities).sample()
            predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

            generated_binder = self.tokenizer.decode(predicted_token_ids, skip_special_tokens=True).replace(' ', '')
            logger.info(f"Generated binder: {generated_binder}")

            # Compute PPL for the generated binder
            ppl_value = self.compute_pseudo_perplexity(protein_seq, generated_binder)

            # Add the generated binder and its PPL to the results list
            binders_with_ppl.append([generated_binder, ppl_value])

        logger.info(f"Generated {num_binders} binders for the sequence")
        return binders_with_ppl

    @method()
    def generate_peptide(self, input_seqs, peptide_length=15, top_k=3, num_binders=4):
        logger.info("Starting peptide generation")
        if isinstance(input_seqs, str):  # Single sequence
            logger.info("Processing single sequence")
            binders = self.generate_peptide_for_single_sequence(input_seqs, peptide_length, top_k, num_binders)
            result = pd.DataFrame(binders, columns=['Binder', 'Pseudo Perplexity'])
            logger.info("Peptide generation completed for single sequence")
            return result

        elif isinstance(input_seqs, list):  # List of sequences
            logger.info(f"Processing list of {len(input_seqs)} sequences")
            results = []
            for i, seq in enumerate(input_seqs):
                logger.info(f"Processing sequence {i+1}/{len(input_seqs)}")
                binders = self.generate_peptide_for_single_sequence(seq, peptide_length, top_k, num_binders)
                for binder, ppl in binders:
                    results.append([seq, binder, ppl])
            result = pd.DataFrame(results, columns=['Input Sequence', 'Binder', 'Pseudo Perplexity'])
            logger.info("Peptide generation completed for all sequences")
            return result



@app.local_entrypoint()
def test():
    
    pepmlm = PepMLM()
    
    # Test compute_pseudo_perplexity
    target_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    binder_seq = "ACDEFGHIKLMNPQRS"
    ppl = pepmlm.compute_pseudo_perplexity.remote(target_seq, binder_seq)
    print(f"Pseudo-perplexity for the given sequence: {ppl}")

    # Test generate_peptide with a single sequence
    single_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    single_result = pepmlm.generate_peptide.remote(single_seq, peptide_length=15, top_k=3, num_binders=2)
    print("\nGenerated peptides for single sequence:")
    print(single_result)
