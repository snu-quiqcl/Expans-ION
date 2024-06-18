import numpy as np
import torch
import copy as cp

E_CHARGE = 1.6022*1e-19                # C
HBAR = 6.62607015*10**(-34)/(2*np.pi)  # J s
M_P = 1.67262192*10**(-27)             # kg
K_B = 1.380649*10**(-23)

def tensor_prod(tensor_list):
    out_tensor = tensor_list[0]
    for tensor in tensor_list[1:]:
        out_tensor = np.tensordot(out_tensor, tensor, axes=0)
    return out_tensor

def sum_product(N_MOD, rho_prev, K_mat, K_d_mat, J_mat_heating = list(), J_mat_dephasing = list(), K_einsum_str=str(), J_einsum_list=list(), rho_shift_plus_list=list(), rho_shift_minus_list=list()): 
    K_einsum_string = K_einsum_str
    J_einsum_string = J_einsum_list
    rho_shift_plus_tuples = rho_shift_plus_list
    rho_shift_minus_tuples = rho_shift_minus_list
    
    K_term = torch.einsum(K_einsum_string, K_mat, rho_prev)
    K_d_term = torch.einsum(K_einsum_string, rho_prev, K_d_mat)
    
    # Add to GPU memory
    rho_plus_list = list()
    rho_minus_list = list()
    J_term_heating_plus_list = list()
    J_term_heating_minus_list = list()
    J_term_dephasing_list = list()

    for i in range(N_MOD):
        rho_plus = torch.zeros_like(rho_prev, dtype=torch.complex128)
        rho_minus = torch.zeros_like(rho_prev, dtype=torch.complex128)
        rho_plus[rho_shift_plus_tuples[0][i]] = rho_prev.clone()[rho_shift_plus_tuples[1][i]]
        rho_minus[rho_shift_minus_tuples[0][i]] = rho_prev.clone()[rho_shift_minus_tuples[1][i]]
        rho_plus_list.append(rho_plus)
        rho_minus_list.append(rho_minus)

    for i in range(N_MOD):
        J_term_heating_plus = torch.einsum(J_einsum_string[i], J_mat_heating[0][i], rho_plus_list[i])
        J_term_heating_minus = torch.einsum(J_einsum_string[i], J_mat_heating[1][i], rho_minus_list[i])
        J_term_dephasing = torch.einsum(J_einsum_string[i], J_mat_dephasing[i], rho_prev)
        J_term_heating_plus_list.append(J_term_heating_plus)
        J_term_heating_minus_list.append(J_term_heating_minus)
        J_term_dephasing_list.append(J_term_dephasing)

    K_term_sum = K_term + K_d_term
    J_term_sum = torch.zeros_like(rho_prev, dtype=torch.complex128)
    for i in range(N_MOD):
        J_term_sum += J_term_heating_plus_list[i] + J_term_heating_minus_list[i] + J_term_dephasing_list[i]
        
    # Release from GPU memory
    del rho_plus_list, rho_minus_list
    
    return K_term_sum + J_term_sum

def generate_sum_product_einsum_string(tensor_shape, N_MOD):
    """
    Generate the einsum string for the sum product of K and J tensors with shape (a, a, b, b, c, c, ...).
    
    Args:
        tensor_shape (tuple): The shape of the input tensor.
        N_MOD: Dimension of the vibrational modes (used to limit the indices in the einsum of J tensors)
    Returns:
        str: Generated einsum strings.
    """
    einsum_range = 26*2
    num_pairs = len(tensor_shape) // 2
    K_einsum_index_len = 3
        
    a_list = list()
    A_list = list()
    for i in range(int(einsum_range/2)):
        a_list.append(chr(ord('a') + i))
        A_list.append(chr(ord('A') + i))
    einsum_index_list = a_list + A_list

    max_pairs = round(int(len(einsum_index_list)/K_einsum_index_len))
    if num_pairs >= max_pairs:
        raise ValueError("The current simulator supports einsum of tensors up to length {}".format(max_pairs))

    # K (propagator) sum product
    K_index_left = str()
    K_index_middle = str()
    K_index_right = str()
    K_einsum_index_list = cp.deepcopy(einsum_index_list)
    for i in range(num_pairs):
        if K_einsum_index_len > len(K_einsum_index_list):
            raise ValueError("{} is greater than the length of the list.".format(K_einsum_index_len))
        first_elements = K_einsum_index_list[:K_einsum_index_len]
        sub_K_index_left = first_elements[0] + first_elements[1]
        sub_K_index_middle = first_elements[1] + first_elements[2]
        sub_K_index_right = first_elements[0] + first_elements[2]
        K_index_left = K_index_left + sub_K_index_left
        K_index_middle = K_index_middle + sub_K_index_middle
        K_index_right = K_index_right + sub_K_index_right
        del K_einsum_index_list[:K_einsum_index_len]
    
    K_einsum_string = "{},{}->{}".format(K_index_left, K_index_middle, K_index_right)

    # J (dissipator) sum product
    J_einsum_string_list = list()
    J_einsum_index_list = cp.deepcopy(einsum_index_list)
    total_elements = J_einsum_index_list[:len(tensor_shape)]
    J_index_total = str()
    for i in range(len(tensor_shape)):
        J_index_total += total_elements[i]
    for i in range(N_MOD):
        sub_J_index_left = total_elements[0] + total_elements[1]
        sub_J_index_middle = J_index_total
        sub_J_index_right = J_index_total
        sub_J_einsum_string = "{},{}->{}".format(sub_J_index_left, sub_J_index_middle, sub_J_index_right)
        J_einsum_string_list.append(sub_J_einsum_string)
        del total_elements[:2]

    return K_einsum_string, J_einsum_string_list

def generate_rho_shift_tuples(N_MOD):
    shift_plus_list = [list() for i in range(2)]
    shift_minus_list = [list() for i in range(2)]

    for i in range(N_MOD):
        start = i * 2
        end = start + 2
        
        # Prepare the indices for the original array
        plus_left_indices = [slice(None)] * (2 * N_MOD)
        plus_right_indices = [slice(None)] * (2 * N_MOD)
        minus_left_indices = [slice(None)] * (2 * N_MOD)
        minus_right_indices = [slice(None)] * (2 * N_MOD)        
        
        # Modify the relevant slices for the original array
        plus_left_indices[start:end] = [slice(None, -1), slice(None, -1)]
        plus_right_indices[start:end] = [slice(1, None), slice(1, None)]
        minus_left_indices[start:end] = [slice(1, None), slice(1, None)]
        minus_right_indices[start:end] = [slice(None, -1), slice(None, -1)]
        
        shift_plus_list[0].append(plus_left_indices)
        shift_plus_list[1].append(plus_right_indices)
        shift_minus_list[0].append(minus_left_indices)
        shift_minus_list[1].append(minus_right_indices)

    return shift_plus_list, shift_minus_list  


class NatConst:
    def __init__(self):
        self.e_charge = 1.6022*1e-19                # C
        self.hbar = 6.62607015*10**(-34)/(2*np.pi)  # J s
        self.m_p = 1.67262192*10**(-27)             # kg
        self.k_B = 1.380649*10**(-23)
