import numpy as np
from sklearn.utils import resample

def compute_export_values_bootstrap(mass_content, n_iterations = 10000):

    num_years = len(np.unique(mass_content.time.dt.year))    
    
    # Container for bootstrap yearly exports
    bootstrap_yearly_exports = []
    
    # Perform bootstrap sampling
    for _ in range(n_iterations):
        # Resample with replacement
        boot_sample = resample(mass_content.values, replace=True)
        # Compute the average yearly export for the bootstrap sample
        boot_yearly_export = boot_sample.sum() / num_years
        
        bootstrap_yearly_exports.append(boot_yearly_export)
    
    # Calculate the mean of the bootstrap yearly exports
    bootstrap_mean = np.mean(bootstrap_yearly_exports)
    
    # Calculate the 95% confidence interval (2 stds)
    bootstrap_std = np.std(bootstrap_yearly_exports) * 2
    
    return bootstrap_mean, bootstrap_std