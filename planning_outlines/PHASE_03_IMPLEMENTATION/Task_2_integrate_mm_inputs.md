### Task 2: Integrate Multimodal Inputs
- [ ] **Success Statement**: I will know I achieved success on this task when enhanced encoders/decoders support autonomous interpretation and generation across modalities, validated through testing.
- **Current Status**: *Partial*—Basic encoders/decoder implemented and tested. Advanced features, integration, and streaming pending.

- [ ] **Step 1: Implement Enhanced Multimodal Encoders**
  - [ ] **Validation Check**: Encoders process text, image, video, audio using hierarchical and temporal schemes effectively (Ref: 3A.2).
  - [ ] **Success Statement**: I will know I achieved success on this step when all modalities stream information-rich spikes continuously, capturing sufficient information (~2255-8460 bits/input) (Ref: 3A.2.iv).
  - [ ] **Actionable Substeps**:
    - [x] Code basic `encoder.py` sub-encoders for different modalities (Ref: 3A).
    - [ ] Implement Poisson spike generation logic (Ref: 3A.3.i):
        - [ ] Calculate spike probability `p = f * dt` (where `dt=1ms`).
        - [ ] Generate spike if `torch.rand(1) < p`.
    - [ ] Implement 5ms refractory period for input neurons (Ref: 3A.3.ii):
        - [ ] Maintain `refractory[i]` counter.
        - [ ] Set `refractory[i]=5` after spike.
        - [ ] Only generate spike if `refractory[i]==0`. Decrement counter each step.
    - [ ] Implement hierarchical encoding enhancements (Ref: 3A.2.ii):
        - [ ] Define hierarchical layers for each modality (e.g., text: char/word/sentence; image: pixel/edge/object).
        - [ ] Implement logic to encode features at different layers with varying spike rates (e.g., `encode_hierarchy(input, layers=3)`).
    - [ ] Implement spike pattern encoding enhancements (Ref: 3A.2.iii):
        - [ ] Implement logic to encode features via precise timing of spikes within a defined window (e.g., 50ms) (`encode_pattern(input, max_rate=10Hz, duration=50ms)`).
    - [ ] Implement encoding robustness: Apply low-pass filter (moving average over 5 steps) to input frequencies before Poisson generation (Ref: 5E.5.ii).
    - [x] Test: Unit test basic encoders (`test_io.py`).
    - [ ] Test: Unit test Poisson spike generation with refractory period implementation.
    - [ ] Test: Unit test enhanced encoding schemes (hierarchical layers logic, pattern generation logic).
    - [ ] Test: Unit test encoding robustness filter application.
    - [ ] Integrate encoders with `run_phase3.py` script for continuous/streaming operation.
    - [ ] Test: Validate handling of multi-domain inputs (temporal separation, sequential cluster activation, inhibitory suppression) (Ref: 2D.4.iv).
  - [ ] **Test: Unit test Step 1 functionalities.**

- [ ] **Step 2: Implement Flexible Decoder**
  - [ ] **Validation Check**: Decoder outputs varied formats (text, structured, silent) based on context/SIE state accurately (Ref: 3B).
  - [ ] **Success Statement**: I will know I achieved success on this step when FUM signals goals flexibly, appropriately, and accurately across different output modes.
  - [ ] **Actionable Substeps**:
    - [x] Code basic rate decoder (`decode_text_rate`) in `decoder.py` (Ref: 3B.2.i).
        - [ ] Implement rate calculation: `rate = sum(spike_history[output_neuron]) / T_window`.
        - [ ] Implement mapping from rate to symbol (e.g., classification: `argmax(rate)`; numerical: `int(rate * scale)`).
    - [ ] Implement temporal decoding for structured output (Ref: 3B.2.ii):
        - [ ] Define sequential time windows for decoding steps.
        - [ ] Implement logic to interpret firing rates within each window.
        - [ ] Implement mapping from windowed rates to tokens/symbols using a lookup table (`token = lookup[rate]`).
    - [ ] Implement mode selection logic (choose text, voice, visual, silent based on SIE state / context) (Ref: Roadmap/3B).
    - [ ] Implement decoder execution on CPU after retrieving spike history/rates from GPU (Ref: 3B.4.i).
    - [ ] Implement logging of decoded outputs to SSD (e.g., `torch.save(outputs, 'outputs.pt')`) (Ref: 3B.4.i).
    - [x] Test: Unit test basic text decoding via rate (`test_io.py`).
    - [ ] Test: Unit test temporal decoding logic (windowing, rate interpretation, token mapping).
    - [ ] Test: Unit test mode selection logic.
    - [ ] Test: Unit test voice/visual output generation (placeholder/basic implementation).
    - [ ] Test: Unit test CPU execution and logging mechanism.
  - [ ] **Test: Unit test Step 2 functionalities.**

- [ ] **Test: Integration test Task 2 components (Encoders, Decoders, interaction with core SNN).**
- [ ] **Test: Unit tests for Task 2.**
