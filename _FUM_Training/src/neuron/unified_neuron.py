import torch
import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# Attempt to import SIE
try:
    from model.sie import SelfImprovementEngine
    _sie_import_success = True
except ImportError as e:
    print(f"Warning: Could not import SelfImprovementEngine: {e}. SIE functionality will be disabled.")
    _sie_import_success = False

# Dynamically add the 'src' directory to sys.path for direct execution
# Assumes this script is in src/neuron/
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Attempt to import the config loader utility using path relative to src_dir
_config_import_success = False
try:
    from utils.config_loader import get_compute_backend, get_hardware_config
    _config_import_success = True
except ImportError as e:
    print(f"Error: Could not import config loader: {e}. Ensure _FUM_Training/src/utils/config_loader.py and __init__.py exist.")
    # Set default values directly if import fails
    COMPUTE_BACKEND = 'cpu'
    HARDWARE_CONFIG = {'compute_backend': 'cpu'}
    print(f"Using fallback compute backend: {COMPUTE_BACKEND}")
    print(f"Using fallback hardware config: {HARDWARE_CONFIG}")


# --- Global Configuration & Device Setup ---
# Ensure _config_import_success is checked before using COMPUTE_BACKEND etc.
if _config_import_success:
    # Get config normally if import succeeded
    COMPUTE_BACKEND = get_compute_backend()
    HARDWARE_CONFIG = get_hardware_config()
# DEVICE initialization happens later in initialize_device()
DEVICE = None

def initialize_device():
    """Sets the global DEVICE based on the compute backend."""
    global DEVICE, COMPUTE_BACKEND # Declare modification intent for global COMPUTE_BACKEND too
    if DEVICE is not None:
        return DEVICE # Already initialized

    if COMPUTE_BACKEND == 'nvidia_cuda':
        if torch.cuda.is_available():
            device_id = HARDWARE_CONFIG.get('cuda_device_id', 0)
            DEVICE = torch.device(f'cuda:{device_id}')
            print(f"Using NVIDIA CUDA backend on device: {DEVICE}")
        else:
            print("Warning: nvidia_cuda backend selected but CUDA is not available. Falling back to CPU.")
            DEVICE = torch.device('cpu')
            COMPUTE_BACKEND = 'cpu' # Update global state
    elif COMPUTE_BACKEND == 'amd_hip':
        # Placeholder: Add ROCm/HIP device selection logic here
        # Requires torch compiled with ROCm support
        # Example (needs verification):
        # if torch.hip.is_available():
        #     device_id = HARDWARE_CONFIG.get('hip_device_id', 0)
        #     DEVICE = torch.device(f'hip:{device_id}')
        #     print(f"Using AMD HIP backend on device: {DEVICE}")
        # else:
        print("Warning: amd_hip backend selected but HIP support is not fully implemented/available. Falling back to CPU.")
        DEVICE = torch.device('cpu')
        COMPUTE_BACKEND = 'cpu' # Update global state
    else: # Default to CPU
        DEVICE = torch.device('cpu')
        print("Using CPU backend.")

    return DEVICE

# Initialize the device when the module is loaded
DEVICE = initialize_device()

# --- Unified Neuron Class ---

class UnifiedNeuronModel:
    """
    Represents the core FUM neuron model, handling state updates (LIF),
    plasticity (STDP), intrinsic plasticity, and SIE interactions.
    Designed to be backend-agnostic via conditional logic.
    Ref: How_It_Works/2_Core_Architecture_Components/2A_Spiking_Neurons.md
    """
    def __init__(self,
                 num_neurons: int,
                 config: Dict[str, Any], # General config containing neuron, stdp, sie params
                 input_dim: int, # Needed for SIE novelty calculation
                 num_clusters: int, # Needed for SIE TD state representation
                 initial_weights: Optional[torch.Tensor] = None,
                 dt: float = 1.0):
        self.num_neurons = num_neurons
        self.config = config # Store the main config
        self.dt = dt # Simulation timestep in ms
        self.device = DEVICE # Use the globally initialized device
        self.input_dim = input_dim # Store input dim for SIE
        self.num_clusters = num_clusters # Store num clusters for SIE

        print(f"Initializing UnifiedNeuronModel with {num_neurons} neurons on device: {self.device}")

        # --- Initialize Heterogeneous Parameters (Based on Docs A.5.i) ---
        # Using torch.float32 for compatibility with typical PyTorch operations
        self.v_reset = -70.0 # Fixed reset potential (mV)
        self.tau = torch.normal(mean=20.0, std=2.0, size=(num_neurons,), device=self.device, dtype=torch.float32) # Membrane time constant (ms)
        self.v_th = torch.normal(mean=-55.0, std=2.0, size=(num_neurons,), device=self.device, dtype=torch.float32) # Threshold potential (mV)

        # --- Intrinsic Plasticity Parameters (Based on Docs A.6.i) ---
        self.ip_target_rate_low = 0.1 # Hz
        self.ip_target_rate_high = 0.5 # Hz
        self.ip_v_th_adj = 0.1 # mV adjustment per check
        self.ip_tau_adj = 0.1 # ms adjustment per check
        self.ip_v_th_bounds = [-60.0, -50.0] # mV bounds
        self.ip_tau_bounds = [15.0, 25.0] # ms bounds
        self.ip_update_interval = 50 # Apply every 50 timesteps (ms)
        self.spike_counts = torch.zeros(num_neurons, device=self.device, dtype=torch.int32)
        self.steps_since_ip_update = 0

        # --- Membrane Resistance (Not explicitly in A.5.i, using example value) ---
        # TODO: Confirm source/value for r_mem. Using 10.0 Mohm for now.
        self.r_mem = 10.0

        # --- Resting Potential (Not explicitly in A.5.i, assuming standard -65mV) ---
        # TODO: Confirm source/value for v_rest. Using -65.0 mV for now.
        self.v_rest = -65.0

        # --- Backend-Specific State Initialization ---
        if COMPUTE_BACKEND == 'nvidia_cuda' or COMPUTE_BACKEND == 'cpu':
            # Initialize voltage state
            self.voltage = torch.full((num_neurons,), self.v_reset, device=self.device, dtype=torch.float32)
            # Add other state variables if needed (e.g., refractory counters)

            # Initialize weights
            if initial_weights is not None:
                self.weights = initial_weights.to(self.device)
            else:
                self.weights = None # Needs proper initialization (e.g., sparse)

            print("Using PyTorch backend for neuron states and weights.")

        elif COMPUTE_BACKEND == 'amd_hip':
            print("Placeholder: Add AMD HIP specific state initialization here.")
            # Fallback to CPU tensors for now if HIP logic isn't implemented
            self.voltage = torch.full((num_neurons,), self.v_reset, device=self.device, dtype=torch.float32)
            self.weights = None
            print("Warning: AMD HIP backend selected but initialization not implemented. Using CPU tensors.")
        else:
             pass # CPU handled above

        # Initialize STDP handler
        self.stdp_handler = self._initialize_stdp_handler()

        # Initialize SIE handler
        self.sie = self._initialize_sie_handler()


    def _initialize_stdp_handler(self):
        """Initializes the appropriate STDP handler based on backend and config."""
        stdp_config = self.config.get('stdp_params', {})
        if not stdp_config:
             print("Warning: STDP parameters not found in config. Using default values.")
             # Define default STDP parameters here if needed, or rely on handler defaults
             stdp_config = {
                 'eta': 0.01, 'a_plus': 0.1, 'a_minus': 0.05,
                 'tau_plus': 20.0, 'tau_minus': 20.0, 'tau_trace': 25.0,
                 'gamma_min': 0.50, 'gamma_max': 0.95,
                 'plv_k': 10.0, 'plv_mid': 0.5,
                 'w_min': -1.0, 'w_max': 1.0
             }

        # Currently, only the Tensor implementation exists for PyTorch backend
        if COMPUTE_BACKEND == 'nvidia_cuda' or COMPUTE_BACKEND == 'cpu':
            try:
                from model.resonance_enhanced_stdp import ResonanceEnhancedSTDP_TraceModulation_Tensor
                print("Initializing ResonanceEnhancedSTDP_TraceModulation_Tensor handler.")
                # Assuming fully connected for now (num_pre=num_post=num_neurons)
                return ResonanceEnhancedSTDP_TraceModulation_Tensor(
                    num_pre=self.num_neurons,
                    num_post=self.num_neurons,
                    device=self.device,
                    **stdp_config # Pass loaded/default config
                )
            except ImportError as e:
                print(f"Error importing Tensor STDP model: {e}. STDP handler not initialized.")
                return None
        elif COMPUTE_BACKEND == 'amd_hip':
            print("Placeholder: Initialize AMD HIP specific STDP handler.")
            return None
        else:
             print("Warning: Unknown compute backend for STDP handler initialization.")
             return None

    def _initialize_sie_handler(self):
        """Initializes the SelfImprovementEngine handler."""
        if not _sie_import_success:
             print("SIE import failed, SIE handler not initialized.")
             return None

        sie_config = self.config.get('sie_params', {})
        if not sie_config:
             print("Warning: SIE parameters not found in config. Using default values.")
             # Define default SIE parameters here if needed, or rely on handler defaults
             sie_config = {
                 'gamma': 0.9, 'td_alpha': 0.1, 'novelty_history_size': 100,
                 'habituation_decay': 0.95, 'target_variance': 0.05,
                 'reward_sigmoid_scale': 1.0
             }

        try:
            print("Initializing SelfImprovementEngine handler.")
            return SelfImprovementEngine(
                config=sie_config,
                device=self.device,
                input_dim=self.input_dim,
                num_clusters=self.num_clusters
            )
        except Exception as e:
            print(f"Error initializing SelfImprovementEngine: {e}. SIE handler not initialized.")
            return None

    def update_state(self, input_currents: torch.Tensor) -> torch.Tensor:
        global COMPUTE_BACKEND # Declare potentially modified global at function start
        """
        Updates neuron states (LIF voltage) based on input currents for one timestep (self.dt).
        Includes spike generation and reset.
        Ref: Docs A.3.i, A.4
        """
        # global COMPUTE_BACKEND # Correct placement: Declare potentially modified global at function start <-- REMOVED DUPLICATE
        if self.device != input_currents.device:
             input_currents = input_currents.to(self.device)
        if input_currents.shape[0] != self.num_neurons:
             raise ValueError(f"Input currents shape ({input_currents.shape}) does not match num_neurons ({self.num_neurons})")

        spikes = torch.zeros_like(self.voltage, dtype=torch.bool) # Initialize spikes tensor for this step

        if COMPUTE_BACKEND == 'nvidia_cuda' or COMPUTE_BACKEND == 'cpu':
            # --- PyTorch LIF Implementation ---
            # dV/dt = (-(V - V_rest) + R*I) / tau
            # Note: Using self.v_rest, self.r_mem, self.tau which are now class attributes
            dV = (-(self.voltage - self.v_rest) + self.r_mem * input_currents) / self.tau * self.dt
            self.voltage += dV

            # Spike generation (using heterogeneous self.v_th)
            spikes = self.voltage >= self.v_th
            self.voltage[spikes] = self.v_reset # Reset voltage for neurons that spiked

            # Update spike counts for intrinsic plasticity
            self.spike_counts[spikes] += 1

        elif COMPUTE_BACKEND == 'amd_hip':
            # --- AMD HIP Kernel Call ---
            try:
                # Assuming 'lib' is loaded globally or accessible here.
                # Need to ensure kernel path is correct relative to execution.
                # TODO: Refactor kernel loading to be more robust.
                # For now, assume fum.py's loading works if run from project root.
                import ctypes # Ensure ctypes is imported
                # Attempt to load if not already loaded (basic check)
                if 'lib' not in globals():
                     # Adjust path relative to this file's location?
                     # Assuming kernel is in _FUM_Training/src/gpu/
                     kernel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gpu', 'lif_kernel.so'))
                     if os.path.exists(kernel_path):
                         lib = ctypes.CDLL(kernel_path)
                         lib.launch_lif_kernel.argtypes = [
                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # V, spikes, I
                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # tau, v_th, v_reset (as tensors/pointers)
                             ctypes.c_int, ctypes.c_float # num_neurons, dt
                         ]
                         lib.launch_lif_kernel.restype = None # Kernel likely modifies tensors in place
                         globals()['lib'] = lib # Store globally for reuse
                     else:
                         raise FileNotFoundError(f"HIP Kernel not found at {kernel_path}")

                # Prepare arguments for the kernel
                # Kernel expects pointers to GPU data.
                # Need v_reset as a tensor if kernel expects it.
                v_reset_tensor = torch.full_like(self.voltage, self.v_reset)
                # Ensure input_currents is on the correct device
                input_currents_dev = input_currents.to(self.device)

                # Allocate output spikes tensor (kernel modifies in place)
                spikes_out_tensor = torch.zeros_like(self.voltage, dtype=torch.bool) # Use bool or match kernel type

                # Call the kernel
                lib.launch_lif_kernel(
                    self.voltage.data_ptr(),
                    spikes_out_tensor.data_ptr(), # Kernel writes spikes here
                    input_currents_dev.data_ptr(),
                    self.tau.data_ptr(),
                    self.v_th.data_ptr(),
                    v_reset_tensor.data_ptr(),
                    self.num_neurons,
                    self.dt
                    # Original fum.py call had extra args (0.01, 1000) - unclear purpose, omitting based on argtypes
                )
                spikes = spikes_out_tensor # Use the output tensor

                # Update spike counts based on kernel output
                self.spike_counts[spikes] += 1
                print("Executed AMD HIP kernel for neuron state update.")

            except Exception as e:
                print(f"ERROR executing AMD HIP kernel: {e}. Falling back to CPU.")
                # Fallback to CPU logic if kernel fails
                dV = (-(self.voltage - self.v_rest) + self.r_mem * input_currents) / self.tau * self.dt
                self.voltage += dV
                spikes = self.voltage >= self.v_th
                self.voltage[spikes] = self.v_reset
                self.spike_counts[spikes] += 1
                # Update global state if fallback occurs
                # global COMPUTE_BACKEND # Removed from here
                COMPUTE_BACKEND = 'cpu' # Now modifies the global declared at function start
                self.device = torch.device('cpu')
                # Move tensors back to CPU if necessary
                self.voltage = self.voltage.to(self.device)
                self.tau = self.tau.to(self.device)
                self.v_th = self.v_th.to(self.device)
                self.spike_counts = self.spike_counts.to(self.device)


        # --- Intrinsic Plasticity Check ---
        self.steps_since_ip_update += 1
        if self.steps_since_ip_update >= self.ip_update_interval:
            self.apply_intrinsic_plasticity()
            self.steps_since_ip_update = 0 # Reset counter

        return spikes # Return boolean tensor indicating which neurons spiked in this step


    def apply_intrinsic_plasticity(self):
        """
        Adjusts neuron thresholds (v_th) and time constants (tau) based on recent firing rates.
        Ref: Docs A.6.i
        """
        if COMPUTE_BACKEND == 'nvidia_cuda' or COMPUTE_BACKEND == 'cpu':
            # Calculate firing rate in Hz for the interval
            current_rate = self.spike_counts / (self.ip_update_interval * self.dt / 1000.0) # Convert interval dt (ms) to s

            # Identify hyperactive and hypoactive neurons
            hyperactive_mask = current_rate > self.ip_target_rate_high
            hypoactive_mask = current_rate < self.ip_target_rate_low

            # Adjust parameters
            self.v_th[hyperactive_mask] += self.ip_v_th_adj
            self.tau[hyperactive_mask] -= self.ip_tau_adj # Decrease tau -> less leaky -> more excitable (counter-intuitive? Check docs/logic) -> Doc says decrease tau to reduce excitability. OK.

            self.v_th[hypoactive_mask] -= self.ip_v_th_adj
            self.tau[hypoactive_mask] += self.ip_tau_adj # Increase tau -> more leaky -> less excitable -> Doc says increase tau to increase excitability. OK.

            # Clamp parameters to bounds
            self.v_th.clamp_(min=self.ip_v_th_bounds[0], max=self.ip_v_th_bounds[1])
            self.tau.clamp_(min=self.ip_tau_bounds[0], max=self.ip_tau_bounds[1])

            # Reset spike counts for the next interval
            self.spike_counts.zero_()

            # print(f"IP Applied: Hyper={hyperactive_mask.sum()}, Hypo={hypoactive_mask.sum()}") # Debug
            # print(f"  New v_th mean: {self.v_th.mean():.2f}, New tau mean: {self.tau.mean():.2f}") # Debug

        elif COMPUTE_BACKEND == 'amd_hip':
            print("Placeholder: Apply intrinsic plasticity using HIP backend (potentially within neuron_kernel.hip or separate kernel).")
            # Need to manage spike counts and parameter updates on GPU if kernel handles IP
            pass


    def apply_stdp(self,
                   pre_spikes_t: torch.Tensor,
                   post_spikes_t: torch.Tensor,
                   plv: float,
                   # SIE-related inputs (placeholders, assuming passed from simulation loop)
                   current_input_encoding: Optional[torch.Tensor] = None,
                   current_avg_spike_rate: Optional[float] = None,
                   current_cluster_id: Optional[int] = None,
                   next_cluster_id: Optional[int] = None,
                   external_reward: float = 0.0,
                   hints: Optional[torch.Tensor] = None, # Added hints
                   hint_factor: float = 0.1): # Added hint_factor
        """
        Calculates SIE reward (if available) and applies modulated STDP updates, potentially biased by hints.

        Args:
            pre_spikes_t (torch.Tensor): Boolean tensor of pre-synaptic spikes at current time t.
            post_spikes_t (torch.Tensor): Boolean tensor of post-synaptic spikes at current time t.
            plv (float): Phase-locking value for this update context.
            current_input_encoding (Optional[torch.Tensor]): Encoded input for SIE novelty.
            current_avg_spike_rate (Optional[float]): Average network spike rate for SIE self-benefit.
            current_cluster_id (Optional[int]): Current cluster ID for SIE TD learning.
            next_cluster_id (Optional[int]): Next cluster ID for SIE TD learning.
            external_reward (float): External reward signal.
            hints (Optional[torch.Tensor]): Optional tensor of hints biasing weight changes.
            hint_factor (float): Scaling factor for the influence of hints.
        """
        if self.stdp_handler is None:
            print("Warning: STDP handler not initialized. Skipping STDP update.")
            return
        if self.weights is None:
            print("Warning: Weights not initialized. Initializing dense random weights for STDP test.")
            self.weights = torch.rand((self.num_neurons, self.num_neurons), device=self.device) * 0.1 - 0.05
            self.weights.fill_diagonal_(0) # No self-connections - Corrected: Call as tensor method

        # --- Calculate SIE Reward and Modulation ---
        total_reward = torch.tensor(external_reward, device=self.device) # Start with external reward
        mod_factor = torch.tensor(0.0, device=self.device) # Default modulation (no change to eta)

        if self.sie is not None and current_input_encoding is not None and \
           current_avg_spike_rate is not None and current_cluster_id is not None and \
           next_cluster_id is not None:
            try:
                total_reward = self.sie.calculate_total_reward(
                    current_input_encoding,
                    current_avg_spike_rate,
                    current_cluster_id,
                    next_cluster_id,
                    external_reward
                )
                mod_factor = self.sie.get_modulation_factor(total_reward)
            except Exception as e:
                print(f"Error during SIE calculation: {e}. Using external reward only.")
                # Reset to external reward and default modulation
                total_reward = torch.tensor(external_reward, device=self.device)
                mod_factor = torch.tensor(0.0, device=self.device)
        elif self.sie is not None:
             print("Warning: Skipping SIE calculation due to missing inputs (encoding, rate, or cluster IDs). Using external reward only.")


        # --- Apply Modulated STDP ---
        if COMPUTE_BACKEND == 'nvidia_cuda' or COMPUTE_BACKEND == 'cpu':
            if pre_spikes_t.shape[0] != self.num_neurons or post_spikes_t.shape[0] != self.num_neurons:
                 print(f"Warning: Spike tensor shapes ({pre_spikes_t.shape}, {post_spikes_t.shape}) mismatch expected ({self.num_neurons}). Skipping STDP.")
                 return

            # Calculate effective learning rate based on modulation factor
            # Ensure stdp_handler.eta exists and is accessible
            base_eta = getattr(self.stdp_handler, 'eta', 0.01) # Get base eta or default
            eta_effective = base_eta * (1.0 + mod_factor)

            # Calculate the final reward value to pass to the STDP update rule
            # Rule: Δw = eta_effective * total_reward * eligibility_trace
            # The handler's update likely implements: Δw = total_reward * eligibility_trace
            # So, we compute modulated_reward_signal = eta_effective * total_reward
            modulated_reward_signal = eta_effective * total_reward

            self.weights = self.stdp_handler.update(
                pre_spikes_t=pre_spikes_t,
                post_spikes_t=post_spikes_t,
                weights=self.weights,
                plv=plv,
                total_reward=modulated_reward_signal,  # Pass the combined modulated reward
                hints=hints, # Pass hints
                hint_factor=hint_factor # Pass hint_factor
            )

        elif COMPUTE_BACKEND == 'amd_hip':
            print("Placeholder: Call AMD HIP kernel for STDP update with modulated reward.")
            pass


# --- Example Usage ---
if __name__ == '__main__':
    print(f"\n--- Unified Neuron Test ---")
    print(f"Selected Backend: {COMPUTE_BACKEND}")
    print(f"Using Device: {DEVICE}")

    # --- Mock Configuration ---
    mock_cvonfig = {
        'neuron_params': {}, # Add if needed
        'stdp_params': { # Example STDP params
            'eta': 0.01, 'a_plus': 0.1, 'a_minus': 0.05, 'tau_plus': 20.0,
            'tau_minus': 20.0, 'tau_trace': 25.0, 'gamma_min': 0.5, 'gamma_max': 0.95,
            'plv_k': 10.0, 'plv_mid': 0.5, 'w_min': -1.0, 'w_max': 1.0
        },
        'sie_params': { # Example SIE params
            'gamma': 0.9, 'td_alpha': 0.1, 'novelty_history_size': 5, # Smaller for test
            'habituation_decay': 0.9, 'target_variance': 0.05,
            'reward_sigmoid_scale': 1.0
        }
    }
    num_test_neurons = 10
    mock_input_dim = 5 # Example input dimension for SIE
    mock_num_clusters = 3 # Example number of clusters for SIE

    # --- Instantiate Model ---
    model = UnifiedNeuronModel(
        num_neurons=num_test_neurons,
        config=mock_config,
        input_dim=mock_input_dim,
        num_clusters=mock_num_clusters,
        dt=1.0
    )

    # --- Simulation Setup ---
    # Input current needs to be scaled appropriately for the LIF equation
    # A constant current that reliably causes spiking for testing:
    # If V_th=-55, V_rest=-65, R=10, tau=20, steady state V = V_rest + R*I
    # To reach -55 from -65, need R*I = 10mV. So I = 1 nA (if R is Mohm, I is nA)
    input_current = torch.full((num_test_neurons,), 1.1, device=DEVICE) # Slightly above threshold current

    # Simulate steps, checking for intrinsic plasticity application
    print("\nSimulating steps...")
    num_steps = 100
    ip_interval = model.ip_update_interval
    for t in range(num_steps):
        spikes = model.update_state(input_current) # dt is handled internally now
        if (t + 1) % ip_interval == 0:
             print(f"Step {t+1}: Spikes = {spikes.sum().item()} (IP Applied)")
             # print(f"  v_th mean: {model.v_th.mean():.2f}, tau mean: {model.tau.mean():.2f}")
        elif t < 5 or t > num_steps - 5: # Print first/last few steps
             print(f"Step {t+1}: Spikes = {spikes.sum().item()}")

    # Basic STDP call test (using current spikes as both pre/post for simplicity)
    if model.stdp_handler is not None and model.weights is not None and model.sie is not None:
         # --- Mock SIE Inputs ---
         mock_encoding = torch.randn(mock_input_dim, device=DEVICE)
         mock_avg_rate = spikes.float().mean().item() # Use current step's avg rate
         mock_current_cluster = t % mock_num_clusters # Cycle through clusters
         mock_next_cluster = (t + 1) % mock_num_clusters
         mock_external_reward = 1.0 if (t % 10 == 0) else 0.0 # Example external reward

         model.apply_stdp(
             pre_spikes_t=spikes,
             post_spikes_t=spikes,
             plv=0.5, # Example PLV
             current_input_encoding=mock_encoding,
             current_avg_spike_rate=mock_avg_rate,
             current_cluster_id=mock_current_cluster,
             next_cluster_id=mock_next_cluster,
             external_reward=mock_external_reward,
             hints=None, # Pass None for hints in basic test
             hint_factor=0.1 # Default hint factor
         )
         # Optional: Print SIE state or weight changes for verification
         # if t == num_steps - 1:
         #     print(f"Final weights mean: {model.weights.mean().item()}")
         #     print(f"Final SIE V_states: {model.sie.V_states.cpu().numpy()}")

    print("--- Test Complete ---")
