# Fully Unified Model (FUM)

## Overview

The Fully Unified Model (FUM) is an advanced, brain-inspired artificial intelligence architecture designed for efficient learning, autonomous operation, and emergent intelligence. It integrates spiking neural networks, spike-timing dependent plasticity (STDP), structural plasticity, and a self-improvement engine (SIE) to learn complex patterns and behaviors from minimal data inputs.

FUM involves significant technical depth. This repository contains comprehensive and detailed documentation reflecting the current stage of active research and development. Explore the key resources linked below to help navigate the material and understand the architecture, concepts, and progress.

An option you have to learn more without navigating this repository is to send your email to justin@neuroca.dev with FUM in the subject. You will gain access to the NotebookLM notebook and gain the ability to ask questions directly to much of the repository content.

## Project Documentation

**Note:** For the most comprehensive and potentially up-to-date understanding, including technical explanations integrated with mathematical details, please explore the documentation within the [`How_It_Works/`](./How_It_Works/) directory (start with the [Table of Contents](./How_It_Works/0_Table_of_Contents.md)). The [`Fully_Unified_Model.pdf`](./Fully_Unified_Model.pdf) provides a focused, consolidated reference specifically for the mathematical formalisms used. To see the novel math used in the development process of the Fully Unified Model, please check out [My Advanced Mathematics Tools](https://github.com/justinlietz93/AdvancedMath/tree/main)


Explore the FUM architecture, concepts, and implementation details through the following resources:

*   **FUM NotebookLM:** [View Interactive Notebook](https://notebooklm.google.com/notebook/3b718c75-c204-40ad-96ea-6bcb7a510f18)
*   **NotebookLM Audio Explanation:** [Listen to Explanation](https://notebooklm.google.com/notebook/3b718c75-c204-40ad-96ea-6bcb7a510f18/audio)
*   (Please send your email to justin@neuroca.dev with FUM in the subject to gain access to the NotebookLM notebook. Currently there is no option to make this available publicly.

  
*   **Technical Whitepaper:** [`Fully_Unified_Model.pdf`](./Fully_Unified_Model.pdf) - Detailed paper on the architecture and theoretical foundations.
*   **Implementation Documentation:** [`How_It_Works/`](./How_It_Works/) - Comprehensive guide covering core components, I/O, emergent behaviors, training, scaling, and practical considerations.

## Quick Start Guides for Different Audiences

**For Investors / Potential Affiliates:**
* Understand the core vision, potential impact, and key differentiators: [High-Level Concept](./How_It_Works/1_High_Level_Concept.md)
* Review the project rationale, feasibility assessment, and high-level roadmap: [Feasibility and Rationale Summary](./How_It_Works/6_Feasibility_and_Rationale_Summary.md)
* See key achieved milestones and current validation status: [Initial Landmarks](./How_It_Works/02_Initial_Landmarks.md) & [Key Validation Highlights](#key-validation-highlights-fum---early-stage-as-of-april-2025) (section above)

**For Technical Reviewers / HPC Access:**
* Start with the overall concept and feasibility: [High-Level Concept](./How_It_Works/1_High_Level_Concept.md) & [Feasibility and Rationale Summary](./How_It_Works/6_Feasibility_and_Rationale_Summary.md)
* Dive into the architecture via the Table of Contents: [How It Works - ToC](./How_It_Works/0_Table_of_Contents.md) (Especially [Core Architecture Components](./How_It_Works/2_Core_Architecture_Components/))
* Understand scaling strategy and resource needs: [Scaling Strategy](./How_It_Works/5_Training_and_Scaling/5D_Scaling_Strategy.md) & [Practical Considerations (Efficiency Section)](./How_It_Works/5_Training_and_Scaling/5E_Practical_Considerations.md#e3-computational-cost-of-overhead-components--net-efficiency)
* Examine mathematical foundations and component analyses: [Mathematical Frameworks Overview](./mathematical_frameworks/) & [PDF Whitepaper (Math Focus)](./Fully_Unified_Model.pdf)
* Review the detailed development/validation plans: [Planning Outlines](./planning_outlines/)

**For Developers (Potential Contributors / Integrators):**
* Review the core concepts and architecture first: [High-Level Concept](./How_It_Works/1_High_Level_Concept.md) & [How It Works - ToC](./How_It_Works/0_Table_of_Contents.md)
* Explore the core codebase structure: [`src/`](./_FUM_Training/src/)
* Understand key component implementations: See relevant sections in [Core Architecture Components](./How_It_Works/2_Core_Architecture_Components/)
* Review testing setup: [`tests/`](./_FUM_Training/tests/)
* Check configuration structure: [`config/`](./_FUM_Training/config/)

**For AI Enthusiasts / Interested Learners:**
* Start with the main README Overview & [Key Differentiating Features](#key-differentiating-features) above.
* Grasp the core ideas: [High-Level Concept](./How_It_Works/1_High_Level_Concept.md)
* Explore the visual overview: [Mind Map](./How_It_Works/FUM_Mind_Map.png)
* Understand the project goals and feasibility: [Feasibility/Rationale Summary](./How_It_Works/6_Feasibility_and_Rationale_Summary.md)
* See the validation approach and highlights: [Key Validation Highlights](#key-validation-highlights-fum---early-stage-as-of-april-2025) & [Section on Emergent Validation](./How_It_Works/...#...)
* (Optional) Ask specific questions via the [NotebookLM Interface](#project-documentation) (requires email access request).

  
## Key Validation Highlights (FUM - Early Stage as of April 2025)

* **Foundational Component Analysis:** Key mechanisms are undergoing rigorous analysis and preliminary validation. This includes:
    * SIE stability analysis via detailed simulation, demonstrating stable dynamics under tested conditions primarily via homeostatic mechanisms. [See Stability Analysis in Mathematical Frameworks]
    * Development and testing of a Topological Data Analysis (TDA) framework on synthetic data, showing strong correlations between proposed metrics and graph properties like efficiency/pathology. [See TDA Framework in Mathematical Frameworks]
* *(Note: Large-scale integrated system testing and benchmarking are planned for upcoming phases as outlined in the roadmap. Results from the predecessor AMN prototype are detailed separately in the relevant documentation.)*


## Conceptual Overview

The following mind map provides a visual representation of the FUM's core concepts and architecture:

![FUM Mind Map](./How_It_Works/FUM_Mind_Map.png)

## Key Differentiating Features

FUM distinguishes itself through a unique synthesis of advanced concepts:

*   **Integrated Multi-Scale Plasticity:** Combines synaptic, intrinsic, and structural plasticity across timescales.
*   **Self-Improvement Engine (SIE):** Autonomous learning driven by internal multi-objective rewards (RL, novelty, homeostasis).
*   **Emergent Knowledge Graph (KG):** Self-organizing structure and pathways arising from local interactions.
*   **Extreme Data Efficiency:** Designed for expert performance from minimal initial data (target: 80-300 inputs).
*   **Adaptive Clustering for State & Plasticity:** Dynamic clustering of neural activity guides RL states and structural changes.
*   **Hybrid Neural-Evolutionary Dynamics:** Blends directed learning (STDP+SIE) with undirected variation for exploration.
*   **Active Criticality Management:** Predictive control maintains network near Self-Organized Criticality (SOC).
*   **Advanced Temporal Credit Assignment:** Uses eligibility traces and synaptic tagging for learning from delays.
*   **Universal Spike-Based Multimodal Processing:** Unified spiking representation handles diverse data types seamlessly.

## Repository Structure

This repository contains the core components, documentation, training infrastructure, and mathematical analysis for the FUM project:

*   `Fully_Unified_Model.pdf`: The main technical whitepaper.
*   `How_It_Works/`: Detailed documentation on the FUM's architecture and functionality.
*   `_FUM_Training/`: Scripts, configurations, data, and source code for training the FUM model across different phases.
    *   `src/`: Core model implementation (neurons, STDP, SIE, I/O, etc.).
    *   `scripts/`: Executable scripts for running training phases and benchmarks.
    *   `config/`: Configuration files for training, hardware, and model parameters.
    *   `data/`: Sample data, seed corpus, and benchmark datasets.
    *   `logs/`: Training logs.
    *   `tests/`: Unit and integration tests.
*   `mathematical_frameworks/`: Analysis and implementation of the mathematical underpinnings, including Knowledge Graph and SIE stability analysis.
*   `planning_outlines/`: Development roadmaps and planning documents for different project phases.

## Theoretical Validation & Mathematical Foundations

The FUM project emphasizes rigorous theoretical validation alongside empirical testing. The [`mathematical_frameworks/`](./mathematical_frameworks/) directory contains dedicated analyses and simulations exploring the properties of core components:

*   **Emergent Knowledge Graph (KG) Dynamics:** Quantifying structure, dynamics, and computation using advanced mathematical techniques like TDA and information geometry. *(Partially Validated)*
*   **Self-Improvement Engine (SIE) Stability & Control:** Formalizing the multi-objective, non-linear neuromodulated control system, analyzing convergence and stability under complex reward structures. *(Partially Validated)*

Future planned theoretical validation and formalization includes:

*   **Integrated Multi-Scale Plasticity:**
    *   *Need:* Ensure stable learning and adaptation when multiple plasticity types (synaptic, intrinsic, structural) interact across different timescales, a core FUM feature not addressed by standard models.
    *   *Discovery:* Analysis of FUM's unique combination of plasticity rules (`How_It_Works/2B`, `2A`) revealed potential stability challenges.
    *   *Plan:* Develop unified models and apply advanced stability analysis (e.g., Lyapunov for hybrid systems) to determine conditions for stable co-adaptation.

*   **Adaptive Clustering for RL & Plasticity:**
    *   *Need:* Validate the novel feedback loop where emergent clusters define RL states (`V(s)`) and guide plasticity, requiring extensions to RL theory for dynamic state spaces.
    *   *Discovery:* Identified as a unique mechanism (`How_It_Works/2F`, `2C`) coupling learning, state representation, and structural change.
    *   *Plan:* Extend TD learning convergence proofs for dynamic states; analyze the stability of the full feedback loop using coupled dynamical systems.

*   **Minimal Data Learning & Generalization:**
    *   *Need:* Provide rigorous justification for FUM's core claim of extreme data efficiency, linking information theory to SNN generalization.
    *   *Discovery:* Stemming from FUM's foundational goal (`How_It_Works/1A`) and unique validation strategies.
    *   *Plan:* Develop SNN-specific information theory measures; adapt statistical learning theory (e.g., PAC bounds) for FUM's STDP/SIE mechanisms; formalize primitive formation.

*   **Heterogeneous & Modulated STDP:**
    *   *Need:* Understand the computational effects of FUM's specific STDP implementation (constrained parameter distributions, multi-factor modulation) beyond standard STDP analysis.
    *   *Discovery:* Analysis of detailed STDP rules (`How_It_Works/2B`, `FUM_SNN_math.md`) highlighted deviations from simpler models.
    *   *Plan:* Extend mean-field/Fokker-Planck methods to incorporate heterogeneity and modulation; analyze impact on stability, memory capacity, and learning dynamics.

*   **Multi-Mechanism Temporal Credit Assignment:**
    *   *Need:* Formalize how FUM's combination of eligibility traces, synaptic tagging analogues, variable decay, and reward gating achieves robust credit assignment over long delays.
    *   *Discovery:* Analysis of FUM's specific mechanisms (`How_It_Works/2B`, `2C`) aimed at overcoming standard RL limitations.
    *   *Plan:* Develop extended RL theoretical models incorporating tagging/consolidation dynamics; analyze convergence and interference mitigation properties.

*   **Active SOC Management:**
    *   *Need:* Validate the stability and effectiveness of FUM's predictive control system designed to maintain beneficial network criticality (near the edge of chaos).
    *   *Discovery:* Arising from the goal of harnessing computational benefits of criticality (`How_It_Works/4`, `2H`) while ensuring stability.
    *   *Plan:* Apply advanced control theory (robust, adaptive, MPC) to the coupled network-controller system; use bifurcation analysis and formal methods to assess stability and performance.

*   **Hybrid Neural-Evolutionary Dynamics:**
    *   *Need:* Analyze the learning dynamics resulting from combining directed, reward-driven learning (STDP+SIE) with undirected, evolutionary-like variation (stochasticity, recombination).
    *   *Discovery:* Identified in FUM's plasticity rules (`How_It_Works/2B`, `2C`) as a potential mechanism for enhanced exploration and innovation.
    *   *Plan:* Develop hybrid dynamical models; apply fitness landscape analysis from evolutionary computation; theoretically compare hybrid vs. pure learning strategies.

*   **Scaling Dynamics & Resource Management:**
    *   *Need:* Create predictive models for FUM's performance, resource use (compute, memory, energy), and emergent behavior at massive scale (trillions of parameters).
    *   *Discovery:* Essential for planning FUM's long-term development and deployment (`How_It_Works/5D`, `5E`, `2I`).
    *   *Plan:* Develop complexity analyses and scaling laws specific to FUM's hybrid architecture; use performance modeling and statistical mechanics of growing networks.

*(Note: Spiking Neuron Dynamics and Multimodal I/O Fidelity are implicitly covered within the analyses of Multi-Scale Plasticity, Heterogeneous STDP, Minimal Data Learning, and Scaling Dynamics.)*

This ongoing work aims to provide a solid mathematical foundation for the FUM architecture and ensure its stability, predictability, and efficiency during scaling and autonomous operation.

## Development Roadmap

FUM development follows a structured, phased approach:

1.  **Phase 1: Foundation Seeding:** Initial network formation and learning from a small seed corpus.
2.  **Phase 2: Competence Scaling:** Refining and strengthening domain-specific pathways using scaled datasets and complexity.
3.  **Phase 3: Continuous Autonomy:** Enabling long-term self-learning, adaptation, and mastery.

Refer to the [`planning_outlines/`](./planning_outlines/) directory for detailed phase plans and the overall project roadmap.

## Getting Started

To understand the FUM architecture and its implementation, please consult the documentation available in the [`How_It_Works/`](./How_It_Works/) directory and the [`Fully_Unified_Model.pdf`](./Fully_Unified_Model.pdf) whitepaper. For details on training the model, refer to the contents within the [`_FUM_Training/`](./_FUM_Training/) directory.
