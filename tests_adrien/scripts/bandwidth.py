import numpy as np

def get_config_bandwidth(split, sample_rate, enc_dim):
    """
    returns a list of sub-bands lengths, so that sum(band_width) = enc_dim = number of frequency bins
    
    expects argument split of format [(delta_f, n_delta_f)] where 
    - delta_f is the width in real frequencies (Hz)
    - n_delta_f is the number of sub-bands of width delta_f
    """
    bandwidths = []
    for delta_f, n_delta_f in split:
        bw = int(np.floor(delta_f / (sample_rate / 2.0) * enc_dim))
        bandwidths.extend([bw] * n_delta_f)

    remaining = enc_dim - sum(bandwidths)
    if remaining > 0:
        bandwidths.append(remaining)

    return bandwidths


def get_bandwidth_from_split(frequency_split, sample_rate, enc_dim):

    if frequency_split == "low_frequency_narrow_split" or frequency_split is None:
        split = [(25, 40), (100, 10), (250, 8), (500, 8)] #66 subbands
    
    elif frequency_split == "normal_split": # pb with this one, inferior number of bins... 46 subbands
        split = [(50, 20), (100, 10), (250, 8), (500, 8)]
    
    elif frequency_split == "even_split":
        split = [(100, 66), (1400, 1)]
    
    elif frequency_split == "non_split": #321 subbands
        split = [(25, 321)]
    

    ## warning: the following require larger analysis windows for narower possible bands

    elif frequency_split == "low_f0_split":
        split = [(25, 3), (13, 9), (25, 14), (50, 9), (70, 11), (250, 12), (500, 6)]
    
    elif frequency_split == "high_f0_split":
        split = [(25, 8), (13, 16), (25, 12), (70, 10), (200, 12), (500, 8)]
    
    elif frequency_split == "f1_split":
        split = [(25, 16), (20, 20), (70, 10), (200, 12), (500, 8)]
    
    elif frequency_split == "f2_split":
        split = [(35, 24), (35, 22), (200, 12), (500, 8)]
    
    elif frequency_split == "f3_split":
        split = [(42, 36), (125, 20), (500, 8)]
    
    else:
        raise ValueError(f"Unknown frequency split scheme: {frequency_split}")

    return get_config_bandwidth(split, sample_rate, enc_dim)
