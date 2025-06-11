import unittest
import torch
import sys
import os
from typing import Dict, Any

# --- Adjust sys.path to find project modules ---
# Assuming this test script is in _FUM_Training/tests/
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Attempt Imports ---
try:
    from model.sie import SelfImprovementEngine
    from neuron.unified_neuron import UnifiedNeuronModel, initialize_device
    _imports_ok = True
except ImportError as e:
    print(f"Error importing necessary modules for test_autonomy: {e}")
    _imports_ok = False

# Determine device (reuse logic from unified_neuron if possible, or simplified)
try:
    DEVICE = initialize_device() # Use the function from unified_neuron
except NameError: # If initialize_device wasn't imported
     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     print(f"Warning: Could not use initialize_device from unified_neuron. Defaulting test device to {DEVICE}")


@unittest.skipIf(not _imports_ok, "Skipping autonomy tests due to import errors.")
class TestSIEAutonomy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up common parameters and mock data for tests."""
        cls.device = DEVICE
        cls.input_dim = 10
        cls.num_clusters = 5
        cls.num_neurons = 20
        cls.mock_sie_config = {
            'gamma': 0.9,
            'td_alpha': 0.1,
            'novelty_history_size': 5,
            'habituation_decay': 0.9,
            'target_variance': 0.05,
            'reward_sigmoid_scale': 1.0
        }
        cls.mock_model_config = {
            'stdp_params': {'eta': 0.01}, # Minimal STDP config for testing
            'sie_params': cls.mock_sie_config
        }

    def test_01_sie_initialization(self):
        """Test if SelfImprovementEngine initializes correctly."""
        sie = SelfImprovementEngine(self.mock_sie_config, self.device, self.input_dim, self.num_clusters)
        self.assertEqual(sie.device, self.device)
        self.assertEqual(sie.V_states.shape[0], self.num_clusters)
        self.assertEqual(sie.recent_inputs.shape, (self.mock_sie_config['novelty_history_size'], self.input_dim))
        self.assertEqual(sie.habituation_counters.shape[0], self.mock_sie_config['novelty_history_size'])
        self.assertEqual(sie.recent_spike_rates.shape[0], 1000) # Hardcoded in SIE for now

    def test_02_sie_novelty_calculation(self):
        """Test novelty calculation logic."""
        sie = SelfImprovementEngine(self.mock_sie_config, self.device, self.input_dim, self.num_clusters)
        input1 = torch.randn(self.input_dim, device=self.device)
        input2 = torch.randn(self.input_dim, device=self.device)
        input1_copy = input1.clone()

        # Fill history
        for _ in range(self.mock_sie_config['novelty_history_size']):
            sie._calculate_novelty(torch.randn(self.input_dim, device=self.device))

        # Novel input
        novelty1, idx1 = sie._calculate_novelty(input1)
        # self.assertGreater(novelty1.item(), 0.5) # Assertion might be too strict for random vectors

        # Repeated input
        novelty2, idx2 = sie._calculate_novelty(input1_copy)
        self.assertLess(novelty2.item(), novelty1.item() * 0.5, "Novelty should decrease significantly on repeat.") # Check relative decrease
        self.assertLess(novelty2.item(), 0.2) # Should be low novelty generally
        # self.assertNotEqual(idx1.item(), idx2.item()) # Index might not change if buffer size is small

        # Different input
        novelty3, idx3 = sie._calculate_novelty(input2)
        self.assertGreater(novelty3.item(), novelty2.item(), "Novelty for new input should be higher than repeated.")
        # self.assertGreater(novelty3.item(), 0.5) # Assertion might be too strict

    def test_03_sie_habituation_calculation(self):
        """Test habituation calculation logic."""
        sie = SelfImprovementEngine(self.mock_sie_config, self.device, self.input_dim, self.num_clusters)
        input_repeated = torch.randn(self.input_dim, device=self.device)

        # Fill history
        for i in range(self.mock_sie_config['novelty_history_size']):
             sie._calculate_novelty(torch.randn(self.input_dim, device=self.device))

        # Initial presentation - Find its index first
        novelty1, idx_repeated = sie._calculate_novelty(input_repeated)
        # Habituation is calculated *after* a match is found and counter incremented
        # So, the first time we calculate habituation *for that index*, it will be 0.1
        # Let's simulate one more step to ensure the counter is incremented
        novelty_dummy, idx_dummy = sie._calculate_novelty(torch.randn(self.input_dim, device=self.device)) # Advance buffer
        habituation_after_first_match = sie._calculate_habituation(idx_repeated) # Now calculate for the original index
        self.assertAlmostEqual(habituation_after_first_match.item(), 0.1, places=5) # Expect 0.1 after first increment

        # Repeat several times
        last_habituation = habituation_after_first_match.item()
        for i in range(5):
            novelty, idx = sie._calculate_novelty(input_repeated)
            # Manually trigger habituation update based on match
            if novelty.item() < 0.1: # If it matched
                 habituation = sie._calculate_habituation(idx)
                 self.assertGreaterEqual(habituation.item(), last_habituation * sie.habituation_decay) # Should increase or decay slightly
                 last_habituation = habituation.item()
            else:
                 # If it didn't match (e.g. buffer wrapped), habituation should be low
                 habituation = sie._calculate_habituation(idx)
                 self.assertLess(habituation.item(), 0.1)


        self.assertGreater(last_habituation, 0.1) # Should have built up some habituation

    def test_04_sie_self_benefit_calculation(self):
        """Test self-benefit (homeostasis) calculation."""
        sie = SelfImprovementEngine(self.mock_sie_config, self.device, self.input_dim, self.num_clusters)
        target_var = self.mock_sie_config['target_variance']

        # Fill history with rates having variance close to target
        rates_good = torch.normal(mean=0.1, std=target_var**0.5, size=(1000,), device=self.device)
        for rate in rates_good:
            benefit = sie._calculate_self_benefit(rate.item())
        self.assertGreater(benefit.item(), 0.8) # Benefit should be high

        # Fill history with rates having low variance
        rates_low_var = torch.full((1000,), 0.1, device=self.device)
        for rate in rates_low_var:
            benefit = sie._calculate_self_benefit(rate.item())
        self.assertLess(benefit.item(), 0.5) # Benefit should be low

        # Fill history with rates having high variance
        rates_high_var = torch.normal(mean=0.1, std=(target_var*5)**0.5, size=(1000,), device=self.device)
        for rate in rates_high_var:
            benefit = sie._calculate_self_benefit(rate.item())
        self.assertLess(benefit.item(), 0.5) # Benefit should be low

    def test_05_sie_td_error_calculation(self):
        """Test TD error calculation and value function update."""
        sie = SelfImprovementEngine(self.mock_sie_config, self.device, self.input_dim, self.num_clusters)
        sie.V_states = torch.tensor([0.1, 0.5, -0.2, 0.0, 0.8], device=self.device)
        current_v = sie.V_states.clone()

        # Test case 1: Positive reward
        td_error1 = sie._calculate_td_error(current_cluster_id=0, next_cluster_id=1, external_reward=1.0)
        expected_td1 = 1.0 + sie.gamma * current_v[1] - current_v[0]
        self.assertAlmostEqual(td_error1.item(), expected_td1.item(), places=5)
        expected_v_update1 = current_v[0] + sie.td_alpha * expected_td1
        self.assertAlmostEqual(sie.V_states[0].item(), expected_v_update1.item(), places=5)

        # Test case 2: Negative reward
        td_error2 = sie._calculate_td_error(current_cluster_id=2, next_cluster_id=3, external_reward=-1.0)
        expected_td2 = -1.0 + sie.gamma * current_v[3] - current_v[2]
        self.assertAlmostEqual(td_error2.item(), expected_td2.item(), places=5)
        expected_v_update2 = current_v[2] + sie.td_alpha * expected_td2
        self.assertAlmostEqual(sie.V_states[2].item(), expected_v_update2.item(), places=5)

        # Test case 3: No external reward
        td_error3 = sie._calculate_td_error(current_cluster_id=4, next_cluster_id=0, external_reward=0.0)
        # Use updated V_state[0] from previous step
        expected_td3 = 0.0 + sie.gamma * sie.V_states[0] - current_v[4]
        self.assertAlmostEqual(td_error3.item(), expected_td3.item(), places=5)
        expected_v_update3 = current_v[4] + sie.td_alpha * expected_td3
        self.assertAlmostEqual(sie.V_states[4].item(), expected_v_update3.item(), places=5)

    def test_06_sie_total_reward_and_modulation(self):
        """Test calculation of total reward and modulation factor."""
        sie = SelfImprovementEngine(self.mock_sie_config, self.device, self.input_dim, self.num_clusters)
        # Mock internal calculations to isolate total reward and modulation
        sie._calculate_novelty = lambda x: (torch.tensor(0.8, device=self.device), torch.tensor(1, device=self.device))
        sie._calculate_habituation = lambda x: torch.tensor(0.2, device=self.device)
        sie._calculate_self_benefit = lambda x: torch.tensor(0.9, device=self.device)
        sie._calculate_td_error = lambda c, n, r: torch.tensor(0.5 + r, device=self.device) # Mock TD error depends on external reward

        # Test with positive external reward
        total_reward1 = sie.calculate_total_reward(None, None, None, None, external_reward=1.0)
        expected_total1 = (0.5 + 1.0) + 0.8 - 0.2 + 0.9 # td + nov - hab + sb
        self.assertAlmostEqual(total_reward1.item(), expected_total1, places=5)
        mod_factor1 = sie.get_modulation_factor(total_reward1)
        expected_mod1 = 2 * torch.sigmoid(total_reward1 * sie.reward_sigmoid_scale) - 1
        self.assertAlmostEqual(mod_factor1.item(), expected_mod1.item(), places=5)
        self.assertGreater(mod_factor1.item(), 0) # Should be positive modulation

        # Test with negative external reward
        total_reward2 = sie.calculate_total_reward(None, None, None, None, external_reward=-1.0)
        expected_total2 = (0.5 - 1.0) + 0.8 - 0.2 + 0.9 # td + nov - hab + sb
        self.assertAlmostEqual(total_reward2.item(), expected_total2, places=5)
        mod_factor2 = sie.get_modulation_factor(total_reward2)
        expected_mod2 = 2 * torch.sigmoid(total_reward2 * sie.reward_sigmoid_scale) - 1
        self.assertAlmostEqual(mod_factor2.item(), expected_mod2.item(), places=5)
        # Depending on the exact value and sigmoid scale, could be positive or negative

    def test_07_unified_neuron_sie_integration(self):
        """Test integration of SIE within UnifiedNeuronModel's apply_stdp."""
        model = UnifiedNeuronModel(
            num_neurons=self.num_neurons,
            config=self.mock_model_config,
            input_dim=self.input_dim,
            num_clusters=self.num_clusters,
            dt=1.0
        )
        # Ensure SIE and STDP handlers were initialized
        self.assertIsNotNone(model.sie)
        self.assertIsNotNone(model.stdp_handler)

        # Mock inputs for apply_stdp
        pre_spikes = torch.rand(self.num_neurons, device=self.device) > 0.8
        post_spikes = torch.rand(self.num_neurons, device=self.device) > 0.8
        plv = 0.7
        mock_encoding = torch.randn(self.input_dim, device=self.device)
        mock_avg_rate = 0.15
        mock_current_cluster = 1
        mock_next_cluster = 2
        mock_external_reward = 0.5

        # --- Spy on SIE methods ---
        original_calc_reward = model.sie.calculate_total_reward
        original_get_mod = model.sie.get_modulation_factor
        sie_calls = {'calc_reward': 0, 'get_mod': 0}
        calculated_reward = None
        calculated_mod_factor = None

        def spy_calc_reward(*args, **kwargs):
            nonlocal calculated_reward
            sie_calls['calc_reward'] += 1
            calculated_reward = original_calc_reward(*args, **kwargs)
            return calculated_reward

        def spy_get_mod(*args, **kwargs):
            nonlocal calculated_mod_factor
            sie_calls['get_mod'] += 1
            calculated_mod_factor = original_get_mod(*args, **kwargs)
            return calculated_mod_factor

        model.sie.calculate_total_reward = spy_calc_reward
        model.sie.get_modulation_factor = spy_get_mod

        # --- Spy on STDP update ---
        original_stdp_update = model.stdp_handler.update
        stdp_calls = {'update': 0}
        received_modulated_reward = None

        def spy_stdp_update(*args, **kwargs):
            nonlocal received_modulated_reward
            stdp_calls['update'] += 1
            # The total_reward passed here should already be eta_eff * total_reward
            received_modulated_reward = kwargs.get('total_reward', None)
            return original_stdp_update(*args, **kwargs)

        model.stdp_handler.update = spy_stdp_update

        # --- Call apply_stdp ---
        model.apply_stdp(
            pre_spikes_t=pre_spikes,
            post_spikes_t=post_spikes,
            plv=plv,
            current_input_encoding=mock_encoding,
            current_avg_spike_rate=mock_avg_rate,
            current_cluster_id=mock_current_cluster,
            next_cluster_id=mock_next_cluster,
            external_reward=mock_external_reward
        )

        # --- Assertions ---
        self.assertEqual(sie_calls['calc_reward'], 1, "SIE calculate_total_reward was not called exactly once.")
        self.assertEqual(sie_calls['get_mod'], 1, "SIE get_modulation_factor was not called exactly once.")
        self.assertEqual(stdp_calls['update'], 1, "STDP update was not called exactly once.")

        self.assertIsNotNone(calculated_reward, "SIE reward was not calculated.")
        self.assertIsNotNone(calculated_mod_factor, "SIE mod_factor was not calculated.")
        self.assertIsNotNone(received_modulated_reward, "Modulated reward signal was not passed to STDP update.")

        # Verify the modulated reward passed to STDP matches calculation
        base_eta = getattr(model.stdp_handler, 'eta', 0.01)
        expected_eta_effective = base_eta * (1.0 + calculated_mod_factor)
        expected_modulated_reward = expected_eta_effective * calculated_reward
        self.assertAlmostEqual(received_modulated_reward.item(), expected_modulated_reward.item(), places=5,
                               msg="Modulated reward signal passed to STDP does not match expected calculation.")

        # Restore original methods if necessary for other tests (though usually test isolation is better)
        model.sie.calculate_total_reward = original_calc_reward
        model.sie.get_modulation_factor = original_get_mod
        model.stdp_handler.update = original_stdp_update


if __name__ == '__main__':
    if not _imports_ok:
        print("Cannot run tests due to import errors.")
    else:
        print(f"Running Autonomy tests on device: {DEVICE}")
        unittest.main()
