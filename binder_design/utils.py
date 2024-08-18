import hashlib

def hash_seq(sequence):
    """
    Generate a hash for a given protein sequence.
    
    Args:
    sequence (str): The protein sequence to hash
    
    Returns:
    str: A hexadecimal string representation of the hash
    """
    # Remove any whitespace and convert to uppercase
    cleaned_sequence = ''.join(sequence.split()).upper()
    
    # Create a SHA256 hash object
    hasher = hashlib.sha256()
    
    # Update the hasher with the cleaned sequence encoded as UTF-8
    hasher.update(cleaned_sequence.encode('utf-8'))
    
    # Return the hexadecimal representation of the hash
    return hasher.hexdigest()[:6]


def get_mutation_diff(seq1, seq2):

    """
    Compare two sequences and return a string of mutations.
    
    Args:
    seq1 (str): The original sequence
    seq2 (str): The mutated sequence
    
    Returns:
    str: A comma-separated string of mutations in the format {original_aa}{position}{new_aa}
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    
    mutations = []
    for i, (aa1, aa2) in enumerate(zip(seq1, seq2)):
        if aa1 != aa2:
            mutations.append(f"{aa1}{i+1}{aa2}")
    
    return ",".join(mutations)