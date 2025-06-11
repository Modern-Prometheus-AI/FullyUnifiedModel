import unittest
import torch
import sys
import os
from typing import Dict, Any

# --- Adjust sys.path to find project modules ---
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Attempt Imports ---
try:
    from neuron.unified_neuron import UnifiedNeuronModel, initialize_device
    _imports_ok = True
except ImportError as e:
    print(f"Error importing necessary modules for test_knowledge_graph: {e}")
    _imports_ok = False

# Determine device
try:
    DEVICE = initialize_device()
except NameError:
     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     print(f"Warning: Could not use initialize_device from unified_neuron. Defaulting test device to {DEVICE}")


@unittest.skipIf(not _imports_ok, "Skipping knowledge graph tests due to import errors.")
class TestKnowledgeGraphHints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up common parameters and mock data for tests."""
        cls.device = DEVICE
        cls.num_neurons = 10 # Smaller size for focused testing
        cls.input_dim = 5
        cls.num_clusters = 2
        cls.mock_config = {
            'stdp_params': {
                'eta': 0.01, 'a_plus': 0.1, 'a_minus': 0.05, 'tau_plus': 20.0,
                'tau_minus': 20.0, 'tau_trace': 25.0, 'gamma_min': 0.5, 'gamma_max': 0.95,
                'plv_k': 10.0, 'plv_mid': 0.5, 'w_min': -1.0, 'w_max': 1.0
            },
            'sie_params': { # Need SIE params for model init, but won't focus on SIE reward here
                'gamma': 0.9, 'td_alpha': 0.1, 'novelty_history_size': 5,
                'habituation_decay': 0.9, 'target_variance': 0.05,
                'reward_sigmoid_scale': 1.0
            }
        }
        # Define neuron groups for testing hints
        cls.group1_indices = torch.tensor([0, 1, 2], dtype=torch.long)
        cls.group2_indices = torch.tensor([3, 4, 5], dtype=torch.long)
        cls.other_indices = torch.tensor([6, 7, 8, 9], dtype=torch.long)

    def test_stdp_with_hints(self):
        """Test if STDP weight updates are biased by the hints tensor."""
        model = UnifiedNeuronModel(
            num_neurons=self.num_neurons,
            config=self.mock_config,
            input_dim=self.input_dim,
            num_clusters=self.num_clusters,
            dt=1.0
        )
        # Ensure STDP handler is initialized
        self.assertIsNotNone(model.stdp_handler)

        # Initialize weights near zero for clear observation of changes
        model.weights = torch.zeros((self.num_neurons, self.num_neurons), device=self.device)

        # Create hints tensor: positive hints within group1, negative between group1 and group2
        hints = torch.zeros_like(model.weights)
        # Positive hints within group 1 (potentiation encouraged)
        for i in self.group1_indices:
            for j in self.group1_indices:
                if i != j: hints[i, j] = 1.0
        # Negative hints from group 1 to group 2 (depression encouraged)
        for i in self.group1_indices:
            for j in self.group2_indices:
                hints[i, j] = -1.0
        # Positive hints from group 2 to group 1 (potentiation encouraged)
        for i in self.group2_indices:
             for j in self.group1_indices:
                 hints[i, j] = 0.5 # Smaller positive hint

        initial_weights = model.weights.clone()

        # Simulate correlated firing within group 1 (pre before post -> potentiation expected)
        # Use a positive external reward to enable learning
        num_updates = 20
        pre_spikes = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)
        post_spikes = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)
        pre_spikes[self.group1_indices[0]] = True # Neuron 0 fires
        post_spikes[self.group1_indices[1]] = True # Neuron 1 fires shortly after (simulated by applying STDP)

        print("\nApplying STDP updates with hints...")
        for i in range(num_updates):
            # Apply STDP with hints and positive reward
            # We use a fixed positive reward directly for simplicity, bypassing SIE calculation
            # The total_reward passed to stdp_handler.update should be eta_eff * total_reward
            # Here, we simulate a positive total_reward and mod_factor=0 for simplicity
            # So, modulated_reward_signal = base_eta * (1.0 + 0.0) * 1.0 = base_eta
            base_eta = self.mock_config['stdp_params']['eta']
            simulated_modulated_reward = torch.tensor(base_eta * 1.0, device=self.device) # Simulate positive reward effect

            # Need to manually update traces before calling stdp_handler.update directly for test control
            model.stdp_handler.pre_trace *= model.stdp_handler.decay_trace
            model.stdp_handler.post_trace *= model.stdp_handler.decay_trace
            model.stdp_handler.pre_trace[pre_spikes] = 1.0
            model.stdp_handler.post_trace[post_spikes] = 1.0
            delta_eligibility_pot = model.stdp_handler.a_plus * model.stdp_handler.pre_trace.unsqueeze(1) * post_spikes.unsqueeze(0).float()
            delta_eligibility_dep = -model.stdp_handler.a_minus * model.stdp_handler.post_trace.unsqueeze(0) * pre_spikes.unsqueeze(1).float()
            delta_eligibility = delta_eligibility_pot + delta_eligibility_dep
            current_gamma = model.stdp_handler._calculate_gamma(plv=0.5) # Assume neutral PLV
            model.stdp_handler.eligibility_traces *= current_gamma
            model.stdp_handler.eligibility_traces += delta_eligibility

            # Now call the update method which applies hints
            model.weights = model.stdp_handler.update(
                pre_spikes_t=pre_spikes, # Pass spikes mainly for shape checks now
                post_spikes_t=post_spikes,
                weights=model.weights,
                plv=0.5, # Assume neutral PLV
                total_reward=simulated_modulated_reward,  # Pass the simulated modulated reward
                hints=hints,
                hint_factor=0.1 # Use a noticeable hint factor
            )
            # print(f"Update {i+1}: w[0,1]={model.weights[0,1].item():.4f}, w[0,3]={model.weights[0,3].item():.4f}, w[3,0]={model.weights[3,0].item():.4f}") # Debug

        # --- Assertions ---
        # Connection 0 -> 1 (within group 1, positive hint) should be strongly potentiated
        self.assertGreater(model.weights[0, 1].item(), initial_weights[0, 1].item() + 0.01, # Expect significant increase
                           "Weight [0,1] (positive hint) did not potentiate significantly.")

        # Connection 0 -> 3 (group 1 to group 2, negative hint) should be depressed or weakly potentiated
        # Note: Base STDP might still cause some potentiation, but hint should counteract it.
        # Check if it's significantly *less* potentiated than the hinted one.
        self.assertLess(model.weights[0, 3].item(), model.weights[0, 1].item() * 0.5, # Should be much less than hinted connection
                        "Weight [0,3] (negative hint) was not significantly less potentiated than [0,1].")
        # It might even become negative depending on parameters
        # self.assertLess(model.weights[0, 3].item(), initial_weights[0, 3].item() + 0.001,
        #                 "Weight [0,3] (negative hint) potentiated unexpectedly.")

        # Connection 3 -> 0 (group 2 to group 1, smaller positive hint) should be potentiated, but less than [0,1]
        # Need to simulate firing 3 -> 0
        model.weights = initial_weights.clone() # Reset weights
        model.stdp_handler.eligibility_traces.zero_() # Reset traces
        pre_spikes.zero_(); post_spikes.zero_()
        pre_spikes[self.group2_indices[0]] = True # Neuron 3 fires
        post_spikes[self.group1_indices[0]] = True # Neuron 0 fires
        for _ in range(num_updates):
            # Manually update traces
            model.stdp_handler.pre_trace *= model.stdp_handler.decay_trace
            model.stdp_handler.post_trace *= model.stdp_handler.decay_trace
            model.stdp_handler.pre_trace[pre_spikes] = 1.0
            model.stdp_handler.post_trace[post_spikes] = 1.0
            delta_eligibility_pot = model.stdp_handler.a_plus * model.stdp_handler.pre_trace.unsqueeze(1) * post_spikes.unsqueeze(0).float()
            delta_eligibility_dep = -model.stdp_handler.a_minus * model.stdp_handler.post_trace.unsqueeze(0) * pre_spikes.unsqueeze(1).float()
            delta_eligibility = delta_eligibility_pot + delta_eligibility_dep
            current_gamma = model.stdp_handler._calculate_gamma(plv=0.5)
            model.stdp_handler.eligibility_traces *= current_gamma
            model.stdp_handler.eligibility_traces += delta_eligibility
            # Apply update
            model.weights = model.stdp_handler.update(
                pre_spikes, post_spikes, model.weights, 0.5,
                total_reward=simulated_modulated_reward,
                hints=hints, hint_factor=0.1
            )

        self.assertGreater(model.weights[3, 0].item(), initial_weights[3, 0].item() + 0.005, # Expect some potentiation
                           "Weight [3,0] (smaller positive hint) did not potentiate.")
        # Compare [3,0] with the first test's [0,1] - need to re-run [0,1] potentiation for fair comparison or store result.
        # For simplicity, just check it's positive.

        # Connection 6 -> 7 (no hint) should show baseline STDP potentiation (likely small positive)
        model.weights = initial_weights.clone() # Reset weights
        model.stdp_handler.eligibility_traces.zero_() # Reset traces
        pre_spikes.zero_(); post_spikes.zero_()
        pre_spikes[self.other_indices[0]] = True # Neuron 6 fires
        post_spikes[self.other_indices[1]] = True # Neuron 7 fires
        baseline_weight_change = 0.0
        for _ in range(num_updates):
            # Manually update traces
            model.stdp_handler.pre_trace *= model.stdp_handler.decay_trace
            model.stdp_handler.post_trace *= model.stdp_handler.decay_trace
            model.stdp_handler.pre_trace[pre_spikes] = 1.0
            model.stdp_handler.post_trace[post_spikes] = 1.0
            delta_eligibility_pot = model.stdp_handler.a_plus * model.stdp_handler.pre_trace.unsqueeze(1) * post_spikes.unsqueeze(0).float()
            delta_eligibility_dep = -model.stdp_handler.a_minus * model.stdp_handler.post_trace.unsqueeze(0) * pre_spikes.unsqueeze(1).float()
            delta_eligibility = delta_eligibility_pot + delta_eligibility_dep
            current_gamma = model.stdp_handler._calculate_gamma(plv=0.5)
            model.stdp_handler.eligibility_traces *= current_gamma
            model.stdp_handler.eligibility_traces += delta_eligibility
            # Apply update WITHOUT hints
            model.weights = model.stdp_handler.update(
                pre_spikes, post_spikes, model.weights, 0.5,
                total_reward=simulated_modulated_reward, hints=None)
            if _ == num_updates - 1: baseline_weight_change = model.weights[6, 7].item()

        self.assertGreater(baseline_weight_change, initial_weights[6, 7].item(),
                           "Weight [6,7] (no hint) did not show baseline potentiation.")

        # Re-run [0,1] to compare against baseline
        model.weights = initial_weights.clone(); model.stdp_handler.eligibility_traces.zero_()
        pre_spikes.zero_(); post_spikes.zero_()
        pre_spikes[self.group1_indices[0]] = True; post_spikes[self.group1_indices[1]] = True
        hinted_positive_weight_change = 0.0
        for _ in range(num_updates):
            model.stdp_handler.pre_trace *= model.stdp_handler.decay_trace; model.stdp_handler.post_trace *= model.stdp_handler.decay_trace
            model.stdp_handler.pre_trace[pre_spikes] = 1.0; model.stdp_handler.post_trace[post_spikes] = 1.0
            delta_eligibility_pot = model.stdp_handler.a_plus * model.stdp_handler.pre_trace.unsqueeze(1) * post_spikes.unsqueeze(0).float()
            delta_eligibility_dep = -model.stdp_handler.a_minus * model.stdp_handler.post_trace.unsqueeze(0) * pre_spikes.unsqueeze(1).float()
            delta_eligibility = delta_eligibility_pot + delta_eligibility_dep
            current_gamma = model.stdp_handler._calculate_gamma(plv=0.5)
            model.stdp_handler.eligibility_traces *= current_gamma; model.stdp_handler.eligibility_traces += delta_eligibility
            model.weights = model.stdp_handler.update(
                pre_spikes, post_spikes, model.weights, 0.5,
                total_reward=simulated_modulated_reward,
                hints=hints, hint_factor=0.1)
            if _ == num_updates - 1: hinted_positive_weight_change = model.weights[0, 1].item()

        self.assertGreater(hinted_positive_weight_change, baseline_weight_change * 1.1, # Expect hint to boost potentiation
                           "Weight [0,1] (positive hint) was not significantly greater than baseline [6,7].")


if __name__ == '__main__':
    if not _imports_ok:
        print("Cannot run tests due to import errors.")
    else:
        print(f"Running Knowledge Graph tests on device: {DEVICE}")
        unittest.main()
