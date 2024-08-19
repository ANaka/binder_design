import os
from modal import App, Secret, gpu, Image, enter, method
import logging
from datetime import datetime
from binder_design import DATA_DIR, EGFS, EGFR
from binder_design.utils import get_mutation_diff, hash_seq
import pandas as pd
from binder_design.sampler import get_fold_results

import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    
image = (
    Image
    .debian_slim(python_version="3.10")
    .pip_install('uv')
    .run_commands("uv pip install --system --compile-bytecode torch evo_prot_grad", gpu="a10g")
    .run_commands("uv pip install --system --compile-bytecode pandas numpy scikit-learn", gpu="a10g")
    )

app = App(name="evo_prot_grad", image=image)

# with image.imports():
#     import torch
#     import numpy as np
#     import pandas as pd
#     import numpy as np
#     from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#     from sklearn.model_selection import KFold
    
  
@app.function(
    container_idle_timeout=150,
    image=image,
    gpu="a10g",
    # concurrency_limit=20,
    # timeout=9600,
)
def train_and_sample_evo_prot_grad(
    input_seqs: list,
    fold_results: pd.DataFrame,
    n_new_seqs_to_return:int=100,
    n_serial_chains_per_seq:int=20,
    n_steps:int=20,
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.model_selection import KFold
    from evo_prot_grad.models.downstream_cnn import OneHotCNN
    from evo_prot_grad.common.tokenizers import OneHotTokenizer
    from evo_prot_grad.common.utils import CANONICAL_ALPHABET
    from evo_prot_grad import get_expert
    import evo_prot_grad
    
    
    class ProteinDataset(Dataset):
        def __init__(self, sequences, properties):
            self.sequences = sequences
            self.properties = properties

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.properties[idx]
        
    df = fold_results.query('pae_interaction < 10').drop_duplicates(subset=['seq_id']).sort_values('pae_interaction')

    df['len'] = df['binder_sequence'].apply(len)
    df = df.query('len == 50')
    all_sequences = df['binder_sequence'].to_list()
    all_properties = (-df['pae_interaction']).to_list() # negative bc too lazy to edit evo_prot_grad
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cnn = OneHotCNN(vocab_size=20, kernel_size=8, input_size=64)
    cnn_tokenizer = OneHotTokenizer(alphabet=CANONICAL_ALPHABET)
   
    # Set up k-fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Hyperparameters
    batch_size = 32
    num_epochs = 300
    learning_rate = 3e-4

    # Lists to store results
    all_train_losses = []
    all_val_losses = []
    all_models = []  # List to store models

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_sequences)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Split data into train and validation sets
        train_seq = [all_sequences[i] for i in train_idx]
        train_prop = [all_properties[i] for i in train_idx]
        val_seq = [all_sequences[i] for i in val_idx]
        val_prop = [all_properties[i] for i in val_idx]
        
        # Create datasets and dataloaders
        train_dataset = ProteinDataset(train_seq, train_prop)
        val_dataset = ProteinDataset(val_seq, val_prop)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model, loss function, and optimizer
        model = OneHotCNN(vocab_size=20, kernel_size=8, input_size=64).to(device)
        tokenizer = OneHotTokenizer(alphabet=CANONICAL_ALPHABET)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            for sequences, properties in train_loader:
                inputs = tokenizer(sequences).to(device)
                properties = properties.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), properties.float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Validation
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for sequences, properties in val_loader:
                    inputs = tokenizer(sequences).to(device)
                    properties = properties.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), properties.float())
                    epoch_val_loss += loss.item()
            
            train_losses.append(epoch_train_loss / len(train_loader))
            val_losses.append(epoch_val_loss / len(val_loader))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_models.append(model)
        
        # Training set predictions
        model.eval()
        train_predictions = []
        train_actual_values = []
        with torch.no_grad():
            for sequences, properties in train_loader:
                inputs = tokenizer(sequences).to(device)
                outputs = model(inputs)
                train_predictions.extend(outputs.squeeze().cpu().tolist())
                train_actual_values.extend(properties.tolist())
        
        
        # Validation set predictions
        val_predictions = []
        val_actual_values = []
        with torch.no_grad():
            for sequences, properties in val_loader:
                inputs = tokenizer(sequences).to(device)
                outputs = model(inputs)
                val_predictions.extend(outputs.squeeze().cpu().tolist())
                val_actual_values.extend(properties.tolist())
        
        
        mse = mean_squared_error(val_actual_values, val_predictions)
        mae = mean_absolute_error(val_actual_values, val_predictions)
        r_squared = r2_score(val_actual_values, val_predictions)
        
        print(f"Validation set - MSE: {mse:.4f}, MAE: {mae:.4f}, R-squared: {r_squared:.4f}")
        
        experts = []
        for model in all_models:
            regression_expert = get_expert(
                'onehot_downstream_regression',
                temperature = 1.0,
                scoring_strategy = 'attribute_value',
                model = model
                )
            experts.append(regression_expert)

        all_preds = []
        for seq in input_seqs:
            try:
                input_pae_i = (df.query('binder_sequence == @seq')['pae_interaction']).values[0]
            except:
                continue
            for ii in range(n_serial_chains_per_seq):
                variants, scores = evo_prot_grad.DirectedEvolution(
                                wt_protein = seq,    # path to wild type fasta file
                                output = 'best',                # return best, last, all variants    
                                experts = experts,   # list of experts to compose
                                parallel_chains = 1,            # number of parallel chains to run
                                n_steps = n_steps,                   # number of MCMC steps per chain
                                max_mutations = 3,             # maximum number of mutations per variant
                                verbose = False                 # print debug info to command line
                )()
                prop_seqs = [''.join(v[0].split(' ')) for v in variants]
                pred_pae_i = -(scores - input_pae_i).ravel()
                preds = pd.DataFrame({'binder_sequence':prop_seqs, 'pred_pae_interaction': pred_pae_i})
                preds['chain'] = ii
                
                all_preds.append(preds)
                
        all_preds = pd.concat(all_preds).drop_duplicates(subset=['binder_sequence']).reset_index(drop=True)
        all_preds = all_preds.sort_values('pred_pae_interaction')
        return all_preds.head(n_new_seqs_to_return)

@app.local_entrypoint()
def test():
    fold_results = get_fold_results()
    df = fold_results.query('pae_interaction < 10').drop_duplicates(subset=['seq_id']).sort_values('pae_interaction').head(10)
    print(df)
    input_seqs = df['binder_sequence'].to_list()
    
    output = train_and_sample_evo_prot_grad.remote(
        input_seqs=input_seqs, 
        fold_results=fold_results
        )
    print(output)