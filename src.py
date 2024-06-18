import copy as cp
# import torch
# import numpy as np
from utils import *
from scipy import special
from tqdm.notebook import tqdm

E_CHARGE = 1.6022*1e-19                # C
HBAR = 6.62607015*10**(-34)/(2*np.pi)  # J s
M_P = 1.67262192*10**(-27)             # kg
K_B = 1.380649*10**(-23)

class ChainAPI:
    def __init__(self, N_ion, MOD_params_dict):
        """
        
        """
        self.N_ION = N_ion
        self.MOD_PARAMS_DICT = MOD_params_dict
        self.N_TLS, self.N_MOD = self.return_object_indices()
        self.base_index_array = self.return_index_array()
        self.tensor_index_array = self.return_tensor_index_array()
        self.BASE_sim_array = self.create_base_sim_array()
        self.CURR_sim_array = np.array([])
        self.TLS_init_array = np.array([])
        self.TLS_init_p_list = list()
        self.MOD_init_q_list = list()

    def return_object_indices(self):
        """
        
        """        
        if 2*self.N_ION != len(self.MOD_PARAMS_DICT["dim"]):
            print("Error: Number of ions does not match the mode dimension")
            del self
        N_tls = self.N_ION
        N_mod = 2*self.N_ION
        return N_tls, N_mod
    
    def return_index_array(self):
        """
        
        """
        dim_list = list()
        mod_params_dim = self.MOD_PARAMS_DICT["dim"]
        for i in range(len(mod_params_dim)):
            dim_list.append(mod_params_dim[i])
        for i in range(self.N_ION):
            dim_list.append(2)
        return np.array(dim_list)
    
    def return_tensor_index_array(self):
        """
        
        """
        return np.array([item for item in self.base_index_array for _ in range(2)])

    def create_base_sim_array(self):
        """
        
        """
        print("Base simulation array created: N_TLS = {}, N_MOD = {}, dimensions {}".format(self.N_TLS, self.N_MOD, self.tensor_index_array))
        return np.zeros(self.tensor_index_array, dtype=complex)
    
    def init_sim_array(self, init_dist_type, init_params_mod, init_params_tls):
        """
        
        """
        if self.N_ION != len(init_params_tls):
            print("Error: Number of ions does not match the TLS dimension")
            return                
        if self.N_ION != len(init_params_mod[0]):
            print("Error: Number of ions does not match the mode dimension")
            return
        
        self.CURR_sim_array = cp.deepcopy(self.BASE_sim_array)

        """
        Construct TLS base
        """
        self.TLS_init_p_list = init_params_tls
        sim_innermost_tls = np.array([[complex(self.TLS_init_p_list[-1][0]), complex(0)], [complex(0), complex(self.TLS_init_p_list[-1][1])]])
        self.TLS_init_array = sim_innermost_tls

        """
        Construct MOD base
        """
        init_params_mod = self.flatten_list(init_params_mod)
        mod_init_q = list()

        if init_dist_type == 'thermal':
            for i in range(self.N_MOD):
                n_list = list()
                for k in range(self.base_index_array[i]):
                    n_mean_init = init_params_mod[i]
                    p_n = (1/(n_mean_init + 1))*(n_mean_init/(n_mean_init + 1))**k
                    n_list.append(p_n)
                n_array = np.array(n_list)
                n_array /= n_array.sum()
                norm_n_list = n_array.tolist()
                mod_init_q.append(norm_n_list)

        elif init_dist_type == 'non-thermal':
            num_dist = 0
            for init_list in init_params_mod:
                num_dist += sum(isinstance(i, list) for i in init_list)
            if num_dist != self.N_MOD:
                print("Error: Define the initial distribution for each mode as lists")
                return
            for i in range(self.N_MOD):
                n_list = init_params_mod[i]
                mod_init_q.append(n_list) 
                n_array = np.array(n_list)
                n_array /= n_array.sum()
                norm_n_list = n_array.tolist()
                mod_init_q.append(norm_n_list)  
        else:
            print("Error: Select a proper intial distribution")
            return
        
        self.MOD_init_q_list = mod_init_q
        init_index = self.base_index_array[:-1]
        self.execute_nested_loops(init_index, self.init_population)

    def init_population(self, *indices):
        """
        
        """
        tensor_indices = [item for item in indices for _ in range(2)]
        p_val = 1
        # print(tensor_indices)
        for index, element in enumerate(indices):
            if index < self.N_MOD:
                p_val = p_val*self.MOD_init_q_list[index][element]
            else:
                p_val = p_val*self.TLS_init_p_list[int(len(indices) - index - 1)][element]
        self.CURR_sim_array[tuple(tensor_indices)] = p_val*self.TLS_init_array

    def flatten_list(self, nested_list):
        """
        Flattens a nested list.

        Parameters:
        nested_list (list): A list that can contain other lists as elements.

        Returns:
        list: A flattened list with all nested elements extracted.
        """
        flattened_list = []
        for element in nested_list:
            if isinstance(element, list):
                flattened_list.extend(self.flatten_list(element))
            else:
                flattened_list.append(element)
        return flattened_list

    def execute_nested_loops(self, tensor_index, function):
        """
        Execute nested for loops based on the provided dimensions and apply an action at the innermost level.

        Parameters:
        tensor_index: A list of integers representing the dimensions for the nested loops.
        function: A function to be executed at the innermost level of the nested loops.
        """    
        def recursive_loops(current_dim, indices):
            if current_dim == len(tensor_index):
                function(*indices)
            else:
                for i in range(tensor_index[current_dim]):
                    recursive_loops(current_dim + 1, indices + [i])

        recursive_loops(0, [])

    def partial_trace(self, array, indices_to_keep):
        """
        Perform a partial trace over a specified index in a multidimensional array.
        
        Parameters:
        array (numpy.ndarray): The input multidimensional array.
        index_to_keep (int): The index to keep, with all other dimensions traced out.
        
        Returns:
        numpy.ndarray: The reduced array with the specified index preserved.
        """
        # Get the shape of the array
        if isinstance(indices_to_keep, np.ndarray):
            indices_to_keep.tolist()
        shape = array.shape
        num_dims = len(shape)
        
        # Ensure the index_to_keep is valid
        for index_to_keep in indices_to_keep:
            if index_to_keep < 0 or index_to_keep >= num_dims // 2:
                raise ValueError("Invalid index_to_keep. It should be between 0 and half of the number of dimensions.")
        
        # Determine the dimensions to keep and trace out
        dims_to_keep = list()
        dims_to_trace_out = list()
        for index_to_keep in indices_to_keep:
            dims_to_keep += [2 * index_to_keep, 2 * index_to_keep + 1]
        dims_to_trace_out += [i for i in range(num_dims) if i not in dims_to_keep]
        # print(dims_to_keep)
        # print(dims_to_trace_out)

        # Move the dimensions to keep to the end of the shape tuple
        new_shape = tuple(shape[i] for i in dims_to_trace_out)
        for index_to_keep in indices_to_keep:
            new_shape += (shape[2 * index_to_keep], shape[2 * index_to_keep + 1])
        # print(new_shape)

        # Reshape the array to isolate the dimensions to keep
        moveaxis_list = list()
        for i in range(int(2 * len(indices_to_keep))):
            moveaxis_list.append(-2 * len(indices_to_keep) + i)
        # print(moveaxis_list)
        reshaped_array = np.moveaxis(array, dims_to_keep, moveaxis_list)
        reshaped_array = reshaped_array.reshape(new_shape)
        
        # Sum over the dimensions to trace out
        for dim in range(len(dims_to_trace_out) // 2):
            reshaped_array = np.trace(reshaped_array, axis1=0, axis2=1)
        
        return reshaped_array
    
    
class ExtSigAPI:
    def __init__(self, ion_index):
        """
        """
        self.ION_index = ion_index
        self.LASER_params_config = [355, "counter", 0*np.pi]   # wavelength (nm), angle w.r.t p-axis_1, unitless
        self.LASER_params_rabi = ["constant", [100E3, 100E3]]   # "constant" or "profile", beam_1, beam_2, units in Hz
        self.LASER_params_detuning = ["constant", [0E6, 0E6]]   # "constant" or "profile", beam_1, beam_2, units in Hz
        self.LASER_params_phase = ["constant", [0*np.pi, 0*np.pi]]   # "constant" or "profile", beam_1, beam_2, units in Hz
        self.LASER_params_stark = ["constant", [0E2, 0E2]]   # "constant" or "profile", beam_1, beam_2, units in Hz
        self.Efield_params = ["constant", [0E2, 0E2]]   # "constant" or "profile", p-axis_1, p-axis_2
        self.set_laser_params()
        
    def set_laser_params(self):
        if self.LASER_params_config[1] == 'counter':
            beam_factor = 2
        elif self.LASER_params_config[1] == 'perpendicular':
            beam_factor = np.sqrt(2)
        self.LASER_params_beam_factor = beam_factor*2*np.pi/(self.LASER_params_config[0]*1e-9)   # units in 1/m

        beam_theta = self.LASER_params_config[2]
        self.LASER_params_delta_k = -np.array([np.cos(beam_theta), np.sin(beam_theta)])

    def set_laser_params_config(self, values_list=list()):
        self.LASER_params_config = values_list
        self.set_laser_params()

    def set_laser_params_rabi(self, values_list=list()):
        self.LASER_params_rabi = values_list

    def set_laser_params_detuning(self, values_list=list()):
        self.LASER_params_detuning = values_list
        
class RunSim:
    def __init__(self):
        self.ION_CHAIN = None
        self.EXT_SIG = None
        self.dev = 'cpu'

    def construct_ion_chain(self, ion_chain_instance=None):
        self.ION_CHAIN = ion_chain_instance

    def set_external_signal(self, ext_sig_instance=None):
        self.EXT_SIG = ext_sig_instance
        
    def set_device(self, dev):
        if dev not in ['cuda', 'cpu']:
            raise Exception("Device should be cuda or cpu")

        if torch.cuda.is_available():
            if dev == 'cpu':
                self.dev = 'cpu'
                self.dev_name = 'cpu'
            elif dev == 'cuda':
                print('Use ' + torch.cuda.get_device_name())
                self.dev = 'cuda'        
                self.dev_name = torch.cuda.get_device_name()
        else:
            print('No GPU device is available')
            self.dev = 'cpu'
            self.dev_name = 'cpu'

    def construct_propagator_K(self):
        if self.ION_CHAIN == None or self.EXT_SIG == None:
            raise Exception("Not defined Ion chain or External signal")
        self._construct_laguerre_list()
        self._construct_LD_params()
        self._construct_K_mat()
        
    def construct_dissipator_J(self):
        if self.ION_CHAIN == None or self.EXT_SIG == None:
            raise Exception("Not defined Ion chain or External signal")
        self.J_dict = {}
        J_heat_list = []
        J_d_heat_list = []
        J_deph_list = []
        for mode_idx in range(self.ION_CHAIN.N_MOD):
            n_mod = self.ION_CHAIN.base_index_array[mode_idx]
            J_mat_heating = np.zeros((n_mod, n_mod), dtype=np.complex128)
            J_d_mat_heating = np.zeros((n_mod, n_mod), dtype=np.complex128)
            J_mat_dephasing = np.zeros((n_mod, n_mod), dtype=np.complex128)
            for m in range(n_mod):
                for n in range(n_mod):
                    Gamma_heating = self.ION_CHAIN.MOD_PARAMS_DICT['heating'][mode_idx]
                    Gamma_dephasing = self.ION_CHAIN.MOD_PARAMS_DICT['dephasing'][mode_idx]
                    bath_mean_n = self.ION_CHAIN.MOD_PARAMS_DICT['mean_n'][mode_idx]
                    J_mat_heating[m, n] = Gamma_heating*(bath_mean_n + 1)*(((m + 1)*(n + 1))**(1/2))
                    J_d_mat_heating[m, n] = Gamma_heating*bath_mean_n*((m*n)**(1/2))
                    J_mat_dephasing[m,n] = Gamma_dephasing*((bath_mean_n + 1)*((m + 1)*(n + 1)) + bath_mean_n*(m*n))
            J_heat_list.append(torch.from_numpy(J_mat_heating).to(self.dev))
            J_d_heat_list.append(torch.from_numpy(J_d_mat_heating).to(self.dev))
            J_deph_list.append(torch.from_numpy(J_mat_dephasing).to(self.dev))
            self.J_dict['J_heating'] = [J_heat_list, J_d_heat_list]
            self.J_dict['J_dephasing'] = J_deph_list

    def _construct_K_mat(self):
        self.K_dict = {}

        for ion_idx in range(self.ION_CHAIN.N_ION):
            self.K_dict[ion_idx] = {}
            
            spin_eye_mat = [np.eye(2, dtype=np.complex128) for _ in range(self.ION_CHAIN.N_ION)]
            spin_up_mat = cp.deepcopy(spin_eye_mat)
            spin_down_mat = cp.deepcopy(spin_eye_mat)
            spin_X_mat = cp.deepcopy(spin_eye_mat)
            spin_Y_mat = cp.deepcopy(spin_eye_mat)
                        
            for laser_idx in range(2):
                self.K_dict[ion_idx][laser_idx] = {}    
                    
                LD_phase = -np.sum(self.LD_params[ion_idx][laser_idx]**2) / 2
                laser_phase = self.EXT_SIG[ion_idx].LASER_params_phase[1][laser_idx]
                rabi = 2*np.pi*self.EXT_SIG[ion_idx].LASER_params_rabi[1][laser_idx]
                detun = 2*np.pi*self.EXT_SIG[ion_idx].LASER_params_detuning[1][laser_idx]
                
                spin_up_mat[ion_idx] = rabi*np.exp(-1j*laser_phase)*np.exp(LD_phase)*np.array([[0,1],[0,0]], dtype=np.complex128)
                spin_down_mat[ion_idx] = rabi*np.exp(1j*laser_phase)*np.exp(LD_phase)*np.array([[0,0],[1,0]], dtype=np.complex128)
                spin_X_mat[ion_idx] = np.array([[0,1j],[1j,0]], dtype=np.complex128)
                spin_Y_mat[ion_idx] = np.array([[0,1j],[-1j,0]], dtype=np.complex128)
                
                mode_mat_list = []
                mode_mat_d_list = []
                mode_index_array = self.ION_CHAIN.tensor_index_array[:-2*self.ION_CHAIN.N_ION]
                mode_mat_evolve = np.zeros(mode_index_array)
                mode_mat_evolve_detun = np.ones(mode_index_array) * detun
                # mode_mat_diag_list = []
            
                for mode_idx in range(self.ION_CHAIN.N_MOD):
                    n_mod = self.ION_CHAIN.base_index_array[mode_idx]
                    LD_mode = self.LD_params[ion_idx, laser_idx, mode_idx]
                    secular = 2*np.pi*self.ION_CHAIN.MOD_PARAMS_DICT['secular'][mode_idx]
                    
                    mode_mat = np.zeros((n_mod,n_mod), dtype=np.complex128)
                    mode_mat_d = np.zeros((n_mod,n_mod), dtype=np.complex128)
                    # mode_mat_diag = np.zeros((n_mod,n_mod), dtype=np.complex128)
                    
                    mode_evolve_axis = 2*mode_idx         
                    for m in range(n_mod):
                        for n in range(n_mod):
                            
                            mode_mat[m][n] = self._f_laguerre(m, n, LD_mode)
                            mode_mat_d[m][n] = self._f_laguerre(m, n, -LD_mode)
                            
                            slc = [slice(None)] * (2*self.ION_CHAIN.N_MOD)
                            slc[mode_evolve_axis] = slice(m, m+1)
                            slc[mode_evolve_axis+1] = slice(n, n+1)
                            mode_mat_evolve[tuple(slc)] += (m-n)*secular
                            # mode_mat_evolve_detun[tuple(slc)] += detun
                            
                    mode_mat_list.append(mode_mat)
                    mode_mat_d_list.append(mode_mat_d)
                                
                K_mat_const = tensor_prod(mode_mat_list + spin_up_mat) + tensor_prod(mode_mat_d_list + spin_down_mat)
                K_const = -(1j/2) * torch.from_numpy(K_mat_const).to(self.dev)
                self.K_dict[ion_idx][laser_idx]['K_const'] = K_const
                self.K_dict[ion_idx][laser_idx]['K_d_const'] = -K_const
                
                K_mat_evolve_sum = tensor_prod([mode_mat_evolve] + spin_X_mat) + tensor_prod([mode_mat_evolve_detun] + spin_Y_mat)
                K_mat_evolve = torch.from_numpy(K_mat_evolve_sum).to(self.dev)
                                
                self.K_dict[ion_idx][laser_idx]['K_evolve'] = K_mat_evolve
    
    def _construct_LD_params(self):
        secular_freqs = np.array(self.ION_CHAIN.MOD_PARAMS_DICT["secular"]).reshape(-1)
        rho_modes = np.sqrt(HBAR / (2*171*M_P*2*np.pi*secular_freqs))

        x_mode_vecs, y_mode_vecs = np.split(np.array(self.ION_CHAIN.MOD_PARAMS_DICT["n-mode"]), 2, axis=1)
        
        rho_ions = np.zeros((self.ION_CHAIN.N_ION, self.ION_CHAIN.N_MOD, 2)) # x / y value for each ion & mode
        for ion_idx in range(self.ION_CHAIN.N_ION):
            for mode_idx in range(self.ION_CHAIN.N_MOD):
                rho_ions[ion_idx][mode_idx][0] = x_mode_vecs[mode_idx,ion_idx] * rho_modes[mode_idx]
                rho_ions[ion_idx][mode_idx][1] = y_mode_vecs[mode_idx,ion_idx] * rho_modes[mode_idx]
        
        
        self.LD_params = np.zeros((self.ION_CHAIN.N_ION, 2, self.ION_CHAIN.N_MOD)) # 2 lasers for each ion
        for ion_idx in range(self.ION_CHAIN.N_ION):
            laser = self.EXT_SIG[ion_idx]
            beam_delta_k = laser.LASER_params_delta_k
            beam_factor = laser.LASER_params_beam_factor
            for laser_idx in range(2):
                for mode_idx in range(self.ION_CHAIN.N_MOD):
                    self.LD_params[ion_idx][laser_idx][mode_idx] = \
                    np.dot(rho_ions[ion_idx][mode_idx], beam_delta_k) * beam_factor
            
    def _construct_laguerre_list(self):
        max_dim = np.max(self.ION_CHAIN.MOD_PARAMS_DICT['dim'])
        self.laguerre_list = {}
        for n1 in range(max_dim):
            self.laguerre_list[n1] = {}
            for n2 in range(max_dim):
                self.laguerre_list[n1][n2] = special.genlaguerre(n1, n2)
            
    # returns the motional Rabi frequency for different vibrational number state transitions
    # factor_down: |0><1|, factor_up: |1><0|
    def _f_laguerre(self, n_f, n_i, LD_param):
        n_diff = int(np.abs(n_f - n_i))
        min_n = int(np.min([n_i,n_f]))
        max_n = int(np.max([n_i,n_f]))
        laguerre_eval = self.laguerre_list[min_n][n_diff](LD_param**2)
        laguerre_factor = ((1j*LD_param)**n_diff)*np.sqrt(special.factorial(min_n)/special.factorial(max_n))*laguerre_eval
        # laguerre_factor = 1
        return laguerre_factor
                    
    def _update_K(self, t_curr):
        K_mat_dt = torch.zeros(tuple(self.ION_CHAIN.tensor_index_array), dtype=torch.complex128).to(self.dev)
        K_d_mat_dt = torch.zeros(tuple(self.ION_CHAIN.tensor_index_array), dtype=torch.complex128).to(self.dev)
        
        for ion_idx in range(self.ION_CHAIN.N_ION):
            for laser_idx in range(2):
         
                K_evolve_mat = self.K_dict[ion_idx][laser_idx]['K_evolve']
                K_evolve_mat_curr = torch.exp(K_evolve_mat*t_curr)
                
                K_mat_dt += K_evolve_mat_curr * self.K_dict[ion_idx][laser_idx]['K_const']
                K_d_mat_dt += K_evolve_mat_curr * self.K_dict[ion_idx][laser_idx]['K_d_const']
                
        return K_mat_dt, K_d_mat_dt
                
    def execute_evolution(self, t_tot, dt, dev='cpu'):
        self.set_device(dev)
        print('Set device to {}'.format(self.dev_name))     
        
        print('Construct K matrix')
        self.construct_propagator_K()
        print('Construct J matrix')
        self.construct_dissipator_J()
        
        sim_timeline = {'time_step': list(), 'density': list()}
        
        print('Start simulation')
        Jmats_heat, Jmats_dephase = self.J_dict['J_heating'], self.J_dict['J_dephasing']
        rho_prev = torch.from_numpy(self.ION_CHAIN.CURR_sim_array).to(self.dev)
        rho_next = torch.from_numpy(self.ION_CHAIN.CURR_sim_array).to(self.dev)

        sim_timeline['time_step'].append(0)
        sim_timeline['density'].append(rho_prev.detach().cpu().numpy())
        
        for iter in tqdm(range(round(t_tot/dt-1))):
            # Generate the einsum string
            K_einsum_string, J_einsum_string_list = generate_sum_product_einsum_string(self.ION_CHAIN.tensor_index_array, self.ION_CHAIN.N_MOD)
            rho_shift_plus, rho_shift_minus = generate_rho_shift_tuples(self.ION_CHAIN.N_MOD)
            
            # step 1
            # Update K matrix
            t_curr = iter*dt
            K_matrix, K_d_matrix = self._update_K(t_curr)
            # Compute k1
            k1= sum_product(self.ION_CHAIN.N_MOD, rho_prev, K_matrix, K_d_matrix, Jmats_heat, Jmats_dephase, 
                            K_einsum_string, J_einsum_string_list,
                            rho_shift_plus, rho_shift_minus)
            
            # step 2-3
            # Update K matrix & Compute k2, k3
            t_curr = iter*dt + dt/2
            K_matrix, K_d_matrix = self._update_K(t_curr)
            k2 = sum_product(self.ION_CHAIN.N_MOD, rho_prev + dt*k1/2, K_matrix, K_d_matrix, Jmats_heat, Jmats_dephase, 
                            K_einsum_string, J_einsum_string_list,
                            rho_shift_plus, rho_shift_minus)
            k3 = sum_product(self.ION_CHAIN.N_MOD, rho_prev + dt*k2/2, K_matrix, K_d_matrix, Jmats_heat, Jmats_dephase, 
                            K_einsum_string, J_einsum_string_list,
                            rho_shift_plus, rho_shift_minus)

            # step 4
            # Update K matrix & Compute k4
            t_curr = iter*dt + dt
            K_matrix, K_d_matrix = self._update_K(t_curr)
            k4 = sum_product(self.ION_CHAIN.N_MOD, rho_prev + dt*k3, K_matrix, K_d_matrix, Jmats_heat, Jmats_dephase, 
                            K_einsum_string, J_einsum_string_list,
                            rho_shift_plus, rho_shift_minus)
            
            rho_next = rho_prev + (1/6)*(k1 + 2*k2 + 2*k3 + k4)*dt  
            sim_timeline['time_step'].append(iter*dt)
            sim_timeline['density'].append(rho_next.detach().cpu().numpy())
            
            # Release from GPU memory
            del K_matrix, K_d_matrix
            rho_prev = rho_next.clone()
        
        self.sim_timeline = sim_timeline
            
