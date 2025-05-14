import numpy as np

def equal_interval_ece(probs, labels, num_bins=15):
    '''
    eqaul interval binning
    '''
    bins = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(labels[in_bin])
            bin_accs.append(avg_accuracy)
            bin_confs.append(avg_confidence)
            bin_counts.append(bin_count)
            ece += (bin_count / len(probs)) * np.abs(avg_accuracy - avg_confidence)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
    return ece, bins, bin_confs, bin_accs, bin_counts