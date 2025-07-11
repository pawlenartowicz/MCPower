"""
Optimized design matrix generator for Monte Carlo Power Analysis with Numba compilation.
Generates correlated predictor variables with various distributions (normal, binary, skewed, etc.)
using lookup tables and Cholesky decomposition for enhanced performance.
"""

import os

if os.environ.get('NUMBA_AOT_BUILD'):
    from numba.pycc import CC
    cc = CC('mcpower_compiled')
    compile_function = lambda sig: lambda func: cc.export(func.__name__, sig)(func) # type: ignore
else:
    from numba import njit
    cc = None
    compile_function = lambda sig: njit(sig, cache=True)


import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple

# Constants
DIST_RESOLUTION = 2048
PERCENTILE_RANGE = (0.001, 0.999)
NORM_RANGE = (-6, 6)
SQRT3 = np.sqrt(3)
SKEW_MEAN = np.exp(0.5)
SKEW_STD = np.sqrt(np.exp(2) - np.exp(1))
NORM_SCALE  = (DIST_RESOLUTION - 1) / (NORM_RANGE[1] - NORM_RANGE[0])
PERC_SCALE = (DIST_RESOLUTION - 1) / (PERCENTILE_RANGE[1] - PERCENTILE_RANGE[0])
FLOAT_NEAR_ZERO = 1e-15 # for division by 0 protection

# Global tables
NORM_CDF_TABLE = None
T3_PPF_TABLE = None

def _init_tables():
    """Initialize distribution lookup tables."""
    global NORM_CDF_TABLE, T3_PPF_TABLE
    
    cache_file = os.path.join(os.path.dirname(__file__), '.generator_lookup_tables.pkl')
    
    try:
        with open(cache_file, 'rb') as f:
            NORM_CDF_TABLE, T3_PPF_TABLE = pickle.load(f)
    except (FileNotFoundError, pickle.PickleError):
        from scipy.stats import norm, t

        # Normal CDF table
        x_norm = np.linspace(*NORM_RANGE, DIST_RESOLUTION)
        NORM_CDF_TABLE = norm.cdf(x_norm)
        
        # T(3) PPF table
        percentile_points= np.linspace(*PERCENTILE_RANGE, DIST_RESOLUTION)
        T3_PPF_TABLE = t.ppf(percentile_points, 3) / np.sqrt(3)
        
        # Cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((NORM_CDF_TABLE, T3_PPF_TABLE), f)
        except:
            pass

_init_tables()


def create_uploaded_lookup_tables(data_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lookup tables for uploaded data variables.
    
    Args:
        data_matrix: (n_samples, n_vars) matrix of uploaded data
        
    Returns:
        (normal_values, uploaded_values): 2D arrays for numba compilation
        - normal_values: (n_vars, n_samples) normal quantiles  
        - uploaded_values: (n_vars, n_samples) corresponding uploaded values
    """
    from scipy.stats import norm
    
    n_samples, n_vars = data_matrix.shape
    
    normal_values = np.zeros((n_vars, n_samples))
    uploaded_values = np.zeros((n_vars, n_samples))
    
    for var_idx in range(n_vars):
        data = data_matrix[:, var_idx]
        
        # Normalize data
        normalized = (data - np.mean(data)) / np.std(data)
        
        # Sort uploaded values
        sorted_uploaded = np.sort(normalized)
        
        # Create corresponding normal quantiles
        percentiles = np.linspace(1/(n_samples+1), n_samples/(n_samples+1), n_samples)
        normal_quantiles = norm.ppf(percentiles)
        
        normal_values[var_idx] = normal_quantiles
        uploaded_values[var_idx] = sorted_uploaded
    
    return normal_values, uploaded_values

# Compiled X generation method
# i8 - sample_size
# i8 - n_vars  
# f8[:,:] - correlation_matrix (2D correlation matrix)
# i8[:] - var_types (variable type codes)
# f8[:] - var_params (variable parameters)
# f8[:] - norm_cdf_table (normal CDF lookup)
# f8[:] - t3_ppf_table (t-distribution PPF lookup)
# f8[:,:] - upload_normal_values (uploaded normal quantiles)
# f8[:,:] - upload_data_values (uploaded data values)
# i8 - seed
@compile_function('f8[:,:](i8, i8, f8[:,:], i8[:], f8[:], f8[:], f8[:], f8[:,:], f8[:,:], i8)')
def _generate_X_compiled(sample_size, n_vars, correlation_matrix, 
                         var_types, var_params, 
                         norm_cdf_table, t3_ppf_table, 
                         upload_normal_values, upload_data_values, 
                         sim_seed):
    """
    Generate X matrix with added correlations and transformed into specified distributions.
    """
    
    def _vectorized_norm_cdf_lookup(x_array):
        """Vectorized normal CDF lookup."""
        n = len(x_array)
        result = np.zeros(n)
        
        for i in range(n):
            x = x_array[i]
            if x < NORM_RANGE[0]:
                result[i] = 0.0
            elif x > NORM_RANGE[1]:
                result[i] = 1.0
            else:
                idx = (x - NORM_RANGE[0]) * NORM_SCALE
                idx_int = int(idx)
                
                if idx_int >= DIST_RESOLUTION - 1:
                    result[i] = norm_cdf_table[DIST_RESOLUTION - 1]
                else:
                    frac = idx - idx_int
                    result[i] = (norm_cdf_table[idx_int] * (1 - frac) + 
                               norm_cdf_table[idx_int + 1] * frac)
        return result

    def _vectorized_t3_ppf_lookup(percentile_array):
        """Vectorized t(3) PPF lookup."""
        n = len(percentile_array)
        result = np.zeros(n)
        
        for i in range(n):
            percentile = percentile_array[i]
            if percentile <= PERCENTILE_RANGE[0]:
                result[i] = t3_ppf_table[0]
            elif percentile >= PERCENTILE_RANGE[1]:
                result[i] = t3_ppf_table[DIST_RESOLUTION - 1]
            else:
                idx = (percentile - PERCENTILE_RANGE[0]) * PERC_SCALE
                idx_int = int(idx)
                
                if idx_int >= DIST_RESOLUTION - 1:
                    result[i] = t3_ppf_table[DIST_RESOLUTION - 1]
                else:
                    frac = idx - idx_int
                    result[i] = (t3_ppf_table[idx_int] * (1 - frac) + 
                               t3_ppf_table[idx_int + 1] * frac)
        return result

    def _vectorized_uploaded_lookup(normal_array, normal_vals, uploaded_vals):
        """Vectorized uploaded data lookup."""
        n_samples = len(normal_array)
        n_lookup = len(normal_vals)
        result = np.zeros(n_samples)
        
        for i in range(n_samples):
            normal_value = normal_array[i]
            
            # Handle edge cases
            if normal_value <= normal_vals[0]:
                result[i] = uploaded_vals[0]
            elif normal_value >= normal_vals[-1]:
                result[i] = uploaded_vals[-1]
            else:
                # Binary search
                left, right = 0, n_lookup - 1
                while left < right - 1:
                    mid = (left + right) // 2
                    if normal_vals[mid] <= normal_value:
                        left = mid
                    else:
                        right = mid
                
                # Linear interpolation
                frac = (normal_value - normal_vals[left]) / (normal_vals[right] - normal_vals[left])
                result[i] = uploaded_vals[left] * (1 - frac) + uploaded_vals[right] * frac
        
        return result

    def _cholesky_decomposition(corr_matrix):
        """Compute Cholesky decomposition with fallback."""
        try:
            return np.linalg.cholesky(corr_matrix)
        except:
            eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
            eigenvals = np.maximum(eigenvals, FLOAT_NEAR_ZERO)
            return eigenvecs @ np.diag(np.sqrt(eigenvals))

    def _transform_distribution(data, dist_type, param, var_idx):
        """Transform standard normal to target distribution - vectorized."""
        
        if dist_type == 0:  # normal
            return data.copy()
        elif dist_type == 1:  # binary
            percentiles = _vectorized_norm_cdf_lookup(data)
            binary_data = (percentiles < param).astype(np.float64)
            return binary_data - np.mean(binary_data)
        elif dist_type == 2:  # right_skewed
            percentiles = _vectorized_norm_cdf_lookup(data)
            return (-np.log(percentiles) - SKEW_MEAN) / SKEW_STD
        elif dist_type == 3:  # left_skewed
            percentiles = _vectorized_norm_cdf_lookup(data)
            return (np.log(1 - percentiles) + SKEW_MEAN) / SKEW_STD
        elif dist_type == 4:  # high_kurtosis
            percentiles = _vectorized_norm_cdf_lookup(data)
            return _vectorized_t3_ppf_lookup(percentiles)
        elif dist_type == 5:  # uniform
            percentiles = _vectorized_norm_cdf_lookup(data)
            return SQRT3 * (2 * percentiles - 1)
        elif dist_type == 99:  # uploaded_data
            # Safe check for uploaded data availability
            if var_idx < upload_normal_values.shape[0]:
                normal_vals = upload_normal_values[var_idx]
                uploaded_vals = upload_data_values[var_idx]
                return _vectorized_uploaded_lookup(data, normal_vals, uploaded_vals)
            else:
                return data.copy()  # Fallback to normal if no uploaded data
        else:
            return data.copy()
    
    # Create matrix with random data
    if sim_seed >= 0:
        np.random.seed(sim_seed)
    base_normal = np.random.standard_normal((sample_size, n_vars))
    
    # Create correlated data from random, and correlation matrix
    cholesky_matrix = _cholesky_decomposition(correlation_matrix)
    correlated_data = base_normal @ cholesky_matrix.T
    
    # Transform into specified distributions
    X = np.zeros((sample_size, n_vars))
    for j in range(n_vars):
        X[:, j] = _transform_distribution(correlated_data[:, j], var_types[j], var_params[j], j)
    
    return X

# wrapper for _generate_X_compiled
def _generate_X(sample_size: int, n_vars: int, correlation_matrix: Optional[np.ndarray] = None,
                      var_types: Optional[np.ndarray] = None, var_params: Optional[np.ndarray] = None,
                      normal_values: Optional[np.ndarray] = None, uploaded_values: Optional[np.ndarray] = None,
                      seed: Optional[int] = None) -> np.ndarray:
    """
    Generate X design matrix with specified distributions and correlations.
    Wrapper only prepares metadata - all computation in compiled function.
    """

    # backup correlation_matrix creation, probably unnecesary
    if correlation_matrix is None:
        correlation_matrix = np.eye(n_vars)
    
    # Prepare uploaded data tables (empty if None), probably unnecesary
    if normal_values is None:
        normal_values = np.zeros((2, 2))  # Small dummy size
    if uploaded_values is None:
        uploaded_values = np.zeros((2, 2))  # Small dummy size
    
    # ALL computation happens in compiled function
    X = _generate_X_compiled(
                             sample_size, n_vars, correlation_matrix, 
                             var_types, var_params, 
                             NORM_CDF_TABLE, T3_PPF_TABLE,
                             normal_values, uploaded_values,
                             seed if seed is not None else -1
    )
    
    return X