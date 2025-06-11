import unittest
import torch
import numpy as np
import sys
import os
from typing import Dict, List

# Add src directory to path to allow importing model components
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the class to be tested and device initializer
try:
    from model.resonance_enhanced_stdp import ResonanceEnhancedSTDP_TraceModulation_Tensor
    from neuron.unified_neuron import initialize_device # Use the same device initializer
    TEST_DEVICE = initialize_device()
    print(f"STDP Test device initialized to: {TEST_DEVICE}")
    _stdp_import_success = True
except ImportError as e:
    print(f"ERROR: Failed to import ResonanceEnhancedSTDP_TraceModulation_Tensor or initialize_device: {e}")
    _stdp_import_success = False
    # Dummy class if import fails
    class ResonanceEnhancedSTDP_TraceModulation_Tensor:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): return torch.tensor([])
        def apply_synaptic_scaling(self, *args, **kwargs): return torch.tensor([])
    TEST_DEVICE = torch.device('cpu')

@unittest.skipIf(not _stdp_import_success, "ResonanceEnhancedSTDP_TraceModulation_Tensor could not be imported.")
class TestResonanceEnhancedSTDP(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for STDP tests."""
        self.num_pre = 5
        self.num_post = 4
        self.device = TEST_DEVICE
        self.stdp_handler = ResonanceEnhancedSTDP_TraceModulation_Tensor(
            num_pre=self.num_pre,
            num_post=self.num_post,
            device=self.device,
            eta=0.01,
            a_plus=0.1, # Mean A+
            a_minus=0.05,
            tau_plus=20.0, # Mean Tau+
            tau_minus=20.0,
            tau_trace=25.0,
            gamma_min=0.5,
            gamma_max=0.95,
            target_rate=0.3,
            temporal_filter_width=4,
            w_min=-1.0,
            w_max=1.0
        )
        # Initialize weights for testing update method
        self.weights = torch.rand((self.num_pre, self.num_post), device=self.device) * 0.2 - 0.1

    def test_initialization(self):
        """Test initialization of parameters and internal states."""
        self.assertEqual(self.stdp_handler.num_pre, self.num_pre)
        self.assertEqual(self.stdp_handler.num_post, self.num_post)
        self.assertEqual(self.stdp_handler.device, self.device)
        # Check shapes of internal tensors
        self.assertEqual(self.stdp_handler.eligibility_traces.shape, (self.num_pre, self.num_post))
        self.assertEqual(self.stdp_handler.pre_trace.shape, (self.num_pre,))
        self.assertEqual(self.stdp_handler.post_trace.shape, (self.num_post,))
        self.assertEqual(self.stdp_handler.a_plus_base.shape, (self.num_pre, self.num_post))
        self.assertEqual(self.stdp_handler.tau_plus_base.shape, (self.num_pre, self.num_post))
        self.assertEqual(self.stdp_handler.tag_history.shape, (self.num_pre, self.num_post, self.stdp_handler.stc_consolidation_duration))
        # Check parameter ranges
        self.assertTrue(torch.all(self.stdp_handler.a_plus_base >= 0.05))
        self.assertTrue(torch.all(self.stdp_handler.a_plus_base <= 0.15))
        self.assertTrue(torch.all(self.stdp_handler.tau_plus_base >= 15.0))
        self.assertTrue(torch.all(self.stdp_handler.tau_plus_base <= 25.0))

    def test_trace_update(self):
        """Test update of pre and post synaptic traces."""
        pre_spikes = torch.tensor([True, False, True, False, False], device=self.device)
        post_spikes = torch.tensor([False, True, False, True], device=self.device)

        initial_pre_trace = self.stdp_handler.pre_trace.clone()
        initial_post_trace = self.stdp_handler.post_trace.clone()

        # Simulate one update step (ignore weight changes for this test)
        _ = self.stdp_handler.update(pre_spikes, post_spikes, self.weights.clone(), plv=0.5, total_reward=0.0) # Pass total_reward

        # Check decay
        self.assertTrue(torch.all(self.stdp_handler.pre_trace <= initial_pre_trace * self.stdp_handler.decay_trace + 1.0)) # Allow for addition
        self.assertTrue(torch.all(self.stdp_handler.post_trace <= initial_post_trace * self.stdp_handler.decay_trace + 1.0))
        # Check reset/increase for spiking neurons
        self.assertEqual(self.stdp_handler.pre_trace[0].item(), 1.0)
        self.assertEqual(self.stdp_handler.pre_trace[2].item(), 1.0)
        self.assertEqual(self.stdp_handler.post_trace[1].item(), 1.0)
        self.assertEqual(self.stdp_handler.post_trace[3].item(), 1.0)
        # Check non-spiking neurons decayed (or stayed zero)
        self.assertLess(self.stdp_handler.pre_trace[1].item(), 1.0)
        self.assertLess(self.stdp_handler.post_trace[0].item(), 1.0)

    def test_eligibility_trace_update(self):
        """Test update of eligibility traces based on spike pairs."""
        # Simulate pre-before-post (potentiation)
        pre_spikes = torch.tensor([True, False, False, False, False], device=self.device)
        post_spikes = torch.tensor([False, True, False, False], device=self.device)
        self.stdp_handler.pre_trace[0] = 0.8 # Assume pre-trace value
        self.stdp_handler.post_trace[1] = 0.0
        initial_eligibility = self.stdp_handler.eligibility_traces.clone()
        _ = self.stdp_handler.update(pre_spikes, post_spikes, self.weights.clone(), plv=0.5, total_reward=0.0) # Pass total_reward
        # Expect potentiation at (0, 1)
        self.assertGreater(self.stdp_handler.eligibility_traces[0, 1].item(), initial_eligibility[0, 1].item())

        # Simulate post-before-pre (depression)
        pre_spikes = torch.tensor([False, False, True, False, False], device=self.device)
        post_spikes = torch.tensor([False, False, False, True], device=self.device)
        self.stdp_handler.pre_trace[2] = 0.0
        self.stdp_handler.post_trace[3] = 0.7 # Assume post-trace value
        initial_eligibility = self.stdp_handler.eligibility_traces.clone()
        _ = self.stdp_handler.update(pre_spikes, post_spikes, self.weights.clone(), plv=0.5, total_reward=0.0) # Pass total_reward
        # Expect depression at (2, 3)
        self.assertLess(self.stdp_handler.eligibility_traces[2, 3].item(), initial_eligibility[2, 3].item())

    def test_weight_update_and_clipping(self):
        """Test the final weight update step including reward and clipping."""
        self.stdp_handler.eligibility_traces[0, 0] = 0.5 # Positive eligibility
        self.stdp_handler.eligibility_traces[1, 1] = -0.3 # Negative eligibility
        weights_before = self.weights.clone()

        # Test positive reward signal
        total_reward_pos = 1.0 # Example positive total reward
        updated_weights_pos = self.stdp_handler.update(
            torch.zeros_like(self.stdp_handler.pre_trace).bool(), # No new spikes
            torch.zeros_like(self.stdp_handler.post_trace).bool(),
            weights_before.clone(), plv=0.5, total_reward=total_reward_pos
        )
        # Expect weight increase at (0,0), decrease at (1,1) (ignoring noise)
        self.assertGreater(updated_weights_pos[0, 0].item(), weights_before[0, 0].item())
        self.assertLess(updated_weights_pos[1, 1].item(), weights_before[1, 1].item())

        # Test negative reward value (should flip the effect of eligibility trace)
        # Note: The implementation uses the total_reward argument directly after
        # modulation. A positive total_reward always reinforces the eligibility
        # trace direction. Here we test a negative value.
        total_reward_neg = -1.0 # Example negative total reward
        updated_weights_neg = self.stdp_handler.update(
            torch.zeros_like(self.stdp_handler.pre_trace).bool(),
            torch.zeros_like(self.stdp_handler.post_trace).bool(),
            weights_before.clone(), plv=0.5, total_reward=total_reward_neg
        )
        # Expect weight decrease at (0,0), increase at (1,1) (ignoring noise)
        self.assertLess(updated_weights_neg[0, 0].item(), weights_before[0, 0].item())
        self.assertGreater(updated_weights_neg[1, 1].item(), weights_before[1, 1].item())

        # Test clipping
        self.weights[0, 0] = self.stdp_handler.w_max - 0.01
        self.stdp_handler.eligibility_traces[0, 0] = 1.0 # Strong potentiation eligibility
        updated_weights_clip = self.stdp_handler.update(
            torch.zeros_like(self.stdp_handler.pre_trace).bool(),
            torch.zeros_like(self.stdp_handler.post_trace).bool(),
            self.weights.clone(), plv=0.5, total_reward=total_reward_pos # Use positive reward
        )
        self.assertAlmostEqual(updated_weights_clip[0, 0].item(), self.stdp_handler.w_max, delta=1e-6)

    def test_diversity_modulation(self):
        """Test the effect of rate and SIE reward modulation on A+."""
        # Need mock inputs for modulation
        pre_rates = torch.ones(self.num_pre, device=self.device) * self.stdp_handler.target_rate * 1.5 # 50% above target
        post_clusters = torch.zeros(self.num_post, device=self.device, dtype=torch.long) # All in cluster 0
        cluster_rewards = torch.tensor([0.5], device=self.device) # Cluster 0 gets 50% reward

        # Manually calculate expected effective A+ for a synapse
        # Base A+ is variable, use mean for estimation
        expected_a_plus = self.stdp_handler.a_plus_base.mean().item()
        expected_a_plus *= 0.5 # Reward modulation
        expected_a_plus *= 1.5 # Rate modulation (clamped rate_factor is 1.5)

        # Need to access a_plus_effective inside update - requires refactoring or a test hook
        # For now, check if potentiation delta changes as expected
        pre_spikes = torch.tensor([True, False, False, False, False], device=self.device)
        post_spikes = torch.tensor([False, True, False, False], device=self.device)
        self.stdp_handler.pre_trace[0] = 1.0
        self.stdp_handler.post_trace[1] = 0.0
        initial_eligibility = self.stdp_handler.eligibility_traces.clone()

        # Run update WITH modulation inputs
        _ = self.stdp_handler.update(
            pre_spikes, post_spikes, self.weights.clone(), plv=0.5, total_reward=0.0, # Pass total_reward
            pre_spike_rates=pre_rates,
            cluster_assignments=post_clusters,
            cluster_rewards=cluster_rewards
        )
        delta_with_mod = self.stdp_handler.eligibility_traces[0, 1] - initial_eligibility[0, 1]

        # Reset and run update WITHOUT modulation inputs
        self.setUp() # Re-init handler
        self.stdp_handler.pre_trace[0] = 1.0
        self.stdp_handler.post_trace[1] = 0.0
        initial_eligibility = self.stdp_handler.eligibility_traces.clone()
        _ = self.stdp_handler.update(
            pre_spikes, post_spikes, self.weights.clone(), plv=0.5, total_reward=0.0 # Pass total_reward
            # No modulation inputs provided
        )
        delta_without_mod = self.stdp_handler.eligibility_traces[0, 1] - initial_eligibility[0, 1]

        # Expect potentiation delta to be different (likely smaller with 0.5 reward mod)
        self.assertNotAlmostEqual(delta_with_mod.item(), delta_without_mod.item(), delta=1e-6)
        # Check if roughly matches expectation (0.5 * 1.5 = 0.75 scaling)
        # Note: This is approximate due to base variability
        self.assertLess(delta_with_mod.item(), delta_without_mod.item())

    def test_stc_tagging(self):
        """Test the STC tagging logic."""
        pre_spikes = torch.tensor([True, False], device=self.device)
        post_spikes = torch.tensor([False, True], device=self.device)
        self.stdp_handler.pre_trace[0] = 1.0 # Ensure high pre-trace for potentiation
        self.stdp_handler.post_trace[1] = 0.0

        # Calculate potentiation delta manually (approx, using mean base A+)
        mean_a_plus_base = 0.1 # From init args
        # Assume no other modulations for simplicity
        delta_pot_expected = mean_a_plus_base * 1.0 * 1.0 # a_plus * pre_trace * post_spike

        # Run update
        _ = self.stdp_handler.update(pre_spikes, post_spikes, self.weights.clone(), plv=0.5, total_reward=0.0) # Pass total_reward

        # Check tag history for the synapse (0, 1) that should have potentiated
        current_tag = self.stdp_handler.tag_history[0, 1, self.stdp_handler.tag_history_ptr - 1] # Check last entry

        if delta_pot_expected > self.stdp_handler.stc_tag_threshold:
            self.assertTrue(current_tag.item(), "Synapse should be tagged after strong potentiation.")
        else:
            self.assertFalse(current_tag.item(), "Synapse should not be tagged if potentiation is below threshold.")

    def test_stc_consolidation(self):
        """Test the STC consolidation bonus logic (conceptual)."""
        # Manually set a persistent tag history for synapse (0,0)
        self.stdp_handler.tag_history[0, 0, :] = True
        self.stdp_handler.eligibility_traces[0, 0] = 0.1 # Some eligibility
        weights_before = self.weights.clone()

        # Run update with positive reward signal
        total_reward_pos = 1.0
        updated_weights = self.stdp_handler.update(
            torch.zeros_like(self.stdp_handler.pre_trace).bool(),
            torch.zeros_like(self.stdp_handler.post_trace).bool(),
            weights_before.clone(), plv=0.5, total_reward=total_reward_pos
        )

        # Check if weight at (0,0) increased by more than just eligibility*reward (due to bonus)
        # Expected base change = reward * eligibility = 1.0 * 0.1 = 0.1
        # Expected bonus = 0.1
        # Total expected delta (ignoring noise) = 0.1 + 0.1 = 0.2
        actual_delta = updated_weights[0, 0] - weights_before[0, 0]
        # Allow for stochastic noise added
        self.assertAlmostEqual(actual_delta.item(), 0.2, delta=0.05, msg="Consolidation bonus not applied correctly.")

    def test_exploration_noise(self):
        """Test that exploration noise is added."""
        weights_before = self.weights.clone()
        # Run update with zero eligibility to isolate noise effect
        self.stdp_handler.eligibility_traces.zero_()
        updated_weights = self.stdp_handler.update(
            torch.zeros_like(self.stdp_handler.pre_trace).bool(),
            torch.zeros_like(self.stdp_handler.post_trace).bool(),
            weights_before.clone(), plv=0.5, total_reward=0.0 # Zero reward
        )
        # Weights should change slightly due to stochastic noise
        self.assertFalse(torch.equal(weights_before, updated_weights), "Weights did not change, exploration noise might be missing.")
        delta_w_mean_abs = torch.mean(torch.abs(updated_weights - weights_before)).item()
        # Expect mean absolute change around the noise level (0.01 * std_normal ~ 0.01 * 0.8)
        self.assertAlmostEqual(delta_w_mean_abs, 0.01 * np.sqrt(2/np.pi), delta=0.005) # Mean abs value of N(0, 0.01^2)

    def test_synaptic_scaling_placeholder(self):
        """Test the placeholder synaptic scaling method."""
        # Create a scenario where scaling should occur
        weights_before = torch.ones((self.num_pre, self.num_post), device=self.device) * 0.5
        is_excitatory = torch.ones(self.num_pre, dtype=torch.bool, device=self.device)
        # Total exc input = 5 * 0.5 = 2.5, which is > 1.0 (target)

        weights_after = self.stdp_handler.apply_synaptic_scaling(weights_before.clone(), is_excitatory)

        # Check that weights were reduced
        self.assertTrue(torch.all(weights_after <= weights_before))
        # Check that the sum is now closer to 1.0
        total_exc_after = torch.sum(weights_after[is_excitatory, 0][weights_after[is_excitatory, 0] > 0])
        self.assertAlmostEqual(total_exc_after.item(), 1.0, delta=1e-6)

    def test_jitter_mitigation_scaling(self):
        """Test latency-aware scaling."""
        self.stdp_handler.eligibility_traces[0, 0] = 1.0
        weights_before = self.weights.clone()

        # Run with zero latency error (no scaling)
        updated_weights_0 = self.stdp_handler.update(
            torch.zeros_like(self.stdp_handler.pre_trace).bool(),
            torch.zeros_like(self.stdp_handler.post_trace).bool(),
            weights_before.clone(), plv=0.5, total_reward=1.0, max_latency=10.0, latency_error=0.0 # Positive reward
        )
        delta_0 = updated_weights_0[0,0] - weights_before[0,0]

        # Run with high latency error (should scale down delta)
        updated_weights_err = self.stdp_handler.update(
            torch.zeros_like(self.stdp_handler.pre_trace).bool(),
            torch.zeros_like(self.stdp_handler.post_trace).bool(),
            weights_before.clone(), plv=0.5, total_reward=1.0, max_latency=10.0, latency_error=5.0 # 50% error -> scale factor 0.5
        )
        delta_err = updated_weights_err[0,0] - weights_before[0,0]

        # Expect delta_err to be smaller than delta_0 (approx half, considering noise)
        self.assertLess(delta_err.item(), delta_0.item())
        self.assertAlmostEqual(delta_err.item(), delta_0.item() * 0.5, delta=0.05) # Allow for noise

    def test_fail_graceful_update(self):
        """Test that update returns original weights on error."""
        weights_before = self.weights.clone()
        # Force an error by providing incompatible shapes
        invalid_spikes = torch.tensor([True], device=self.device)

        updated_weights = self.stdp_handler.update(
            invalid_spikes, # Incorrect shape
            torch.zeros_like(self.stdp_handler.post_trace).bool(),
            weights_before.clone(), plv=0.5, total_reward=0.0
        )
        # Should return original weights due to caught exception
        self.assertTrue(torch.equal(weights_before, updated_weights))

        # Test NaN/Inf check
        self.stdp_handler.eligibility_traces[0,0] = 1.0
        weights_nan = weights_before.clone()
        weights_nan[0,0] = torch.nan
        updated_weights_nan = self.stdp_handler.update(
             torch.zeros_like(self.stdp_handler.pre_trace).bool(),
             torch.zeros_like(self.stdp_handler.post_trace).bool(),
             weights_nan, # Pass weights that already contain NaN
             plv=0.5, total_reward=1.0 # Positive reward
        )
         # Should still return original weights passed in (which had NaN)
         # The check prevents *new* NaNs from being introduced by the update itself
         # Let's modify to test if update *introduces* NaN
        self.stdp_handler.eligibility_traces[0,0] = torch.inf # Inf eligibility
        weights_clean = weights_before.clone()
        updated_weights_inf = self.stdp_handler.update(
             torch.zeros_like(self.stdp_handler.pre_trace).bool(),
             torch.zeros_like(self.stdp_handler.post_trace).bool(),
             weights_clean.clone(),
             plv=0.5, total_reward=1.0 # Positive reward
        )
        # Should return the clean original weights because the update would create Inf
        self.assertTrue(torch.equal(weights_clean, updated_weights_inf))


    # TODO: Add tests for synaptic scaling placeholder
    # TODO: Add tests for fail-graceful logic (e.g., by forcing NaNs)

if __name__ == '__main__':
    if not _stdp_import_success:
        print("Skipping STDP tests as ResonanceEnhancedSTDP_TraceModulation_Tensor could not be imported.")
    else:
        print(f"Running STDP tests on device: {TEST_DEVICE}")
        unittest.main()
