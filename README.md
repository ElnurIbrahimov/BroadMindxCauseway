# BroadMind x Causeway

## Integration of Causal Counterfactual Reasoning with Latent Program Execution

| Component | Repository |
|---|---|
| **Causeway** | [github.com/ElnurIbrahimov/causeway](https://github.com/ElnurIbrahimov/causeway) |
| **BroadMind** | [github.com/ElnurIbrahimov/BroadMind](https://github.com/ElnurIbrahimov/BroadMind) |

---

## 1. What Each System Does

### [Causeway](https://github.com/ElnurIbrahimov/causeway) — "What would change if I did X?"

A parameter-efficient causal adapter (794K params) that bolts onto frozen Transformers. Given a Transformer's hidden state and a candidate action, it returns a structured delta vector describing what would change under intervention.

```
StateEncoder -> CausalGraphLayer -> InterventionEngine -> DeltaPredictor
```

- **Input**: Transformer hidden state `h` (768-dim for GPT-2) + candidate action (768-dim)
- **Output**: 5-dimensional structured delta with per-dimension confidence

| Output Dimension | Meaning |
|---|---|
| `risk_shift` | How much risk increases/decreases |
| `goal_progress` | Change in progress toward goals |
| `constraint_violation` | Probability of violating constraints |
| `resource_cost` | Change in resource consumption |
| `success_probability` | Change in likelihood of success |

Key techniques: Pearl's do-operator (abduction-action-prediction), learned sparse DAG via NOTEARS + Gumbel-sigmoid gating, near-orthogonal rotation matrix for causal variable extraction.

Results: 0.9494 correlation on GPT-2 hidden states.

### [BroadMind](https://github.com/ElnurIbrahimov/BroadMind) — "How do I execute this program?"

A neural program executor (447K params) that generates and executes its own internal programs at runtime. Not a Transformer — it synthesizes latent instructions during forward execution and runs them to produce outputs.

```
WisdomMatcher -> [LatentGenerator -> RecursionRouter -> LatentExecutor -> Halter] (loop)
```

- **Input**: Initial state (x, y, z) + program operations
- **Output**: Final state after execution

Key techniques: Latent program induction (96-dim runtime-generated instructions), wisdom distillation (48-dim compressed task knowledge), adaptive compute (halter), Mixture of Recursions (1-4 inner depth), elastic inference (25-100% width).

Results: 100% accuracy at full width, 98.5% at 50% width, ~447K parameters.

---

## 2. Why They're Complementary

The two systems solve different halves of the same problem:

| Capability | Causeway | BroadMind |
|---|---|---|
| Causal reasoning | Learned sparse DAG, Pearl's do-operator | None |
| Program execution | None | Latent program induction |
| Consequence prediction | Structured delta vectors | None |
| Adaptive compute depth | Fixed 3-step propagation | Halter + Mixture of Recursions |
| Knowledge compression | Causal graph structure | Wisdom distillation (48-dim codes) |
| Backbone dependency | Bolts onto frozen Transformers | Standalone |

Neither duplicates the other. Causeway knows *what will change*; BroadMind knows *how to compute*.

### Dimensional Coincidence

Both systems use 48-dimensional compressed representations:
- Causeway: `d_causal = 48` (number of causal variables / graph nodes)
- BroadMind: `d_wisdom = 48` (compressed task knowledge codes)

This makes the bridge between them natural — no dimension adapter needed.

---

## 3. Integration Architecture

### Location

```
causeway/integration/broadmind_bridge.py
```

### Three Integration Modules

#### Module 1: CausalWisdomBridge

Converts Causeway's learned causal graph into BroadMind-compatible wisdom codes.

```
adjacency (48x48) + causal_vars (batch, 48)
    |
    [RowEncoder] -> mean-pool -> graph summary (64-dim)
    [VarEncoder] -> variable state summary (64-dim)
    [GraphStats] -> degree, density, spectral gap (4-dim)
    |
    [Fusion MLP] -> wisdom code (48-dim)
    |
    Output: wisdom compatible with BroadMind's WisdomBank
```

The bridge answers: "Given what the causal graph says about how variables relate, what procedural knowledge should guide program execution?"

#### Module 2: AdaptiveInterventionEngine

Replaces Causeway's fixed `propagation_steps=3` with BroadMind-style adaptive compute.

```
z + action + adjacency
    |
    [ActionEncoder] -> mask + values (same as Causeway)
    [Abduction]     -> epsilon (exogenous noise)
    [Intervention]  -> do-operator applied
    |
    For step 0..max_depth:
        [Propagate one step through causal graph]
        [Halter] -> should we stop?
            if converged: BREAK
    |
    Output: z_counterfactual + steps_used
```

Simple interventions (leaf variables) converge in 1 step. Complex causal chains with long paths use up to `max_depth` steps. No wasted compute.

#### Module 3: CausalProgramExecutor (The Joint System)

The main integration module that wires everything together.

```
Transformer hidden state h + candidate action
    |
    ============= CAUSEWAY SIDE =============
    |
    [Causeway.StateEncoder]     -> causal variables z (48-dim)
    [Causeway.CausalGraphLayer] -> refined z + learned adjacency
    [Causeway.InterventionEngine + DeltaPredictor]
        -> delta vector (5 values + 5 confidence)
    |
    ============= BRIDGE =============
    |
    [CausalWisdomBridge]  -> graph_wisdom (48-dim)
    [DeltaToWisdom MLP]   -> delta_wisdom (48-dim)
    causal_wisdom = graph_wisdom + delta_wisdom
    |
    ============= FUSION =============
    |
    [BroadMind.WisdomMatcher] -> matched_wisdom (48-dim)
    [GatedFusion]
        gate = sigmoid(Linear(causal_wisdom || matched_wisdom))
        fused = gate * causal_wisdom + (1-gate) * matched_wisdom
    |
    ============= BROADMIND SIDE =============
    |
    [BroadMind.ElasticSolver] (using fused wisdom)
        For each program step:
            [LatentGenerator(state, op, step, fused_wisdom)]
            [RecursionRouter -> depth selection]
            [LatentExecutor] x depth -> state update
            [Halter] -> stop check
    |
    Output: CausalExecutionResult
        - predictions (batch, n_steps, 3)   -- BroadMind state predictions
        - delta_values (batch, 5)           -- Causeway structured deltas
        - delta_confidence (batch, 5)       -- per-dimension confidence
        - causal_wisdom (batch, 48)         -- wisdom from causal graph
        - matched_wisdom (batch, 48)        -- BroadMind's own wisdom
        - fused_wisdom (batch, 48)          -- blended result
        - compute_cost                      -- BroadMind compute cost
```

### Wisdom Fusion Modes

| Mode | Behavior | Use Case |
|---|---|---|
| `gate` (default) | Learned sigmoid gate blends causal and matched wisdom | Most flexible; lets the model decide how much to trust each source |
| `add` | Simple addition of both wisdom signals | When both signals are always useful |
| `replace` | Use only causal wisdom, ignore BroadMind's matcher | When causal structure dominates task knowledge |

---

## 4. Parameter Budget

| Component | Parameters | % of Total |
|---|---|---|
| Causeway | 794,170 | 61.1% |
| BroadMind | 446,613 | 34.3% |
| CausalWisdomBridge | 52,528 | 4.0% |
| DeltaToWisdom MLP | 2,880 | 0.2% |
| Fusion Gate | 4,656 | 0.4% |
| **Total** | **1,300,847** | **100%** |

The bridge overhead is 60,064 parameters (4.6% of the combined system). The total 1.3M is still lightweight enough for edge deployment on Raspberry Pi or phone.

---

## 5. Verified Tensor Shapes

All shapes verified through end-to-end forward pass (batch_size=4):

```
CausalWisdomBridge:
  adj(48, 48) + z(4, 48)  ->  wisdom(4, 48)

AdaptiveInterventionEngine:
  z(4, 48) + action(4, 768) + adj(48, 48)  ->  z_cf(4, 48), steps=1..6

CausalProgramExecutor:
  h(4, 768) + action(4, 768) + programs(4, 4) + states(4, 3)
    ->  predictions(4, 4, 3)
        delta_values(4, 5)
        delta_confidence(4, 5)
        causal_wisdom(4, 48)
        matched_wisdom(4, 48)
        fused_wisdom(4, 48)
```

---

## 6. Training Strategy

### Joint Training Loss

`CausalExecutorLoss` combines four terms:

```
L_total = lambda_task * L_task
        + lambda_delta * L_delta
        + lambda_align * L_alignment
        + lambda_struct * L_structural
```

| Term | What It Measures | Default Weight |
|---|---|---|
| `L_task` | BroadMind state prediction MSE | 1.0 |
| `L_delta` | Causeway counterfactual accuracy | 1.0 |
| `L_alignment` | Cosine similarity between fused and matched wisdom (penalizes them being identical, forcing the bridge to contribute) | 0.1 |
| `L_structural` | Causeway's DAG acyclicity + sparsity + orthogonality | 0.01 |

### Recommended Training Schedule

| Phase | What | Details |
|---|---|---|
| 1 | Train Causeway independently | On Transformer hidden states (existing `train_on_transformer.py`) |
| 2 | Train BroadMind independently | 8-phase curriculum (existing `BroadMind_v077_elastic.py`) |
| 3 | Freeze both, train bridge only | CausalWisdomBridge + DeltaToWisdom + Gate (~60K params) |
| 4 | Fine-tune end-to-end | Low learning rate, all components unfrozen |

Phase 3 is the critical one — it teaches the bridge to translate between Causeway's causal space and BroadMind's wisdom space without disrupting either pretrained system.

---

## 7. Usage

### Building from Checkpoints

```python
from integration.broadmind_bridge import build_causal_executor

executor = build_causal_executor(
    causeway_checkpoint="causeway_gpt2.pt",
    broadmind_checkpoint="../BroadMind/broadmind_v077_elastic.pt",
    d_model=768,
    fusion_mode="gate",
    device="cuda",
)
```

### Full Forward Pass

```python
import torch

h = torch.randn(4, 768)                      # Transformer hidden states
action = torch.randn(4, 768)                  # Candidate action
programs = torch.randint(0, 12, (4, 4))       # 4-step program
initial_states = torch.rand(4, 3) * 10        # (x, y, z) initial values

result = executor(h, action, programs, initial_states)

# Causeway's causal predictions
print(result.delta_values)       # (4, 5) - what would change
print(result.delta_confidence)   # (4, 5) - how confident

# BroadMind's execution results
print(result.predictions)        # (4, 4, 3) - predicted states per step

# Wisdom inspection
print(result.causal_wisdom)      # (4, 48) - from causal graph
print(result.fused_wisdom)       # (4, 48) - blended with BroadMind's
```

### Ablation: Causal-Only or BroadMind-Only

```python
# Just causal reasoning (no program execution)
causal_out = executor.forward_causal_only(h, action)

# Just program execution (no causal input)
bm_out = executor.forward_broadmind_only(programs, initial_states)
```

### Standalone Modules

```python
from integration.broadmind_bridge import CausalWisdomBridge, AdaptiveInterventionEngine

# Use the bridge independently
bridge = CausalWisdomBridge(d_causal=48, d_wisdom=48)
wisdom = bridge(adjacency_matrix, causal_variables)

# Use adaptive intervention independently
engine = AdaptiveInterventionEngine(d_causal=48, d_action=768, max_depth=6)
z_cf, mask, epsilon, steps_used = engine(z, action, adjacency)
```

---

## 8. What the Integration Enables

### Before Integration

| System | Can Answer | Cannot Answer |
|---|---|---|
| Causeway | "If I increase deploy load, risk goes up 0.23" | "How do I execute a sequence of operations to minimize risk?" |
| BroadMind | "Given program [ACC_X, TRANSFER_XY], final state is (4, 7, 2)" | "Is this program safe? What are the causal consequences?" |

### After Integration

The combined system can answer: **"Given the causal structure of this domain, generate and execute a program that achieves the desired outcome while understanding the consequences."**

Concrete capabilities:
1. **Causally-informed execution**: BroadMind's latent programs are steered by causal knowledge. It doesn't just execute blindly — it knows what will change.
2. **Adaptive intervention depth**: Causal effect propagation uses only as many steps as needed, matching BroadMind's "no wasted compute" philosophy.
3. **Unified wisdom**: Compressed task knowledge (BroadMind's wisdom) is enriched with structural causal knowledge (Causeway's graph), producing programs that respect causal constraints.
4. **Ablation-ready**: Either side can be disabled for controlled experiments measuring the contribution of causal reasoning vs procedural execution.

---

## 9. Technical Compatibility

| Dimension | Causeway | BroadMind | Compatible? |
|---|---|---|---|
| Framework | PyTorch | PyTorch | Identical |
| Compressed repr dim | 48 (d_causal) | 48 (d_wisdom) | Identical |
| Training paradigm | Phased curriculum | 8-phase curriculum | Same pattern |
| Parameter scale | 794K | 447K | Both sub-1M |
| Edge deployment | Not yet | ONNX + INT8 (v0.78) | BroadMind path exists |
| Backbone | Frozen Transformer | Standalone | Complementary |
| Adaptive compute | Fixed depth | Halter + MoR | BroadMind's is richer |
| DAG learning | NOTEARS + Gumbel-sigmoid | None | Causeway's contribution |
| Knowledge compression | Causal graph structure | Wisdom distillation | Both present |

---

## 10. File Structure

```
causeway/
    integration/
        broadmind_bridge.py        <-- THE INTEGRATION MODULE
        transformer_bridge.py      (existing: Causeway <-> Transformer)
    causeway/
        causeway_module.py         (main Causeway nn.Module)
        state_encoder.py           (h -> causal variables z)
        causal_graph.py            (sparse DAG learning)
        intervention_engine.py     (Pearl's do-operator)
        delta_predictor.py         (z_pre/z_post -> structured delta)
        losses.py                  (training losses)
    train.py / train_on_transformer.py / evaluate.py / demo_gpt2.py

BroadMind/
    BroadMind_v077_elastic.py      (full BroadMind v0.77 system)
    BroadMind_v078_edge.py         (ONNX export + INT8 quantization)
    broadmind_v077_elastic.pt      (trained checkpoint)
```

---

## 11. Future Directions

### Near-Term
- **Shared domain**: Define a domain where both systems operate on the same variables. Currently Causeway models software deployment (8 variables) and BroadMind models 3-variable integer arithmetic. A shared domain (e.g., resource allocation with causal dependencies) would enable true joint training.
- **Causal program search**: Use Causeway's delta predictions to search over BroadMind programs, selecting the one with the best predicted causal outcome before execution.
- **Bridge distillation**: After joint training, distill the CausalWisdomBridge into BroadMind's WisdomBank directly, removing the bridge overhead at inference.

### Medium-Term
- **BroadMind as intervention engine**: Replace Causeway's fixed InterventionEngine entirely with BroadMind's solver. The causal graph becomes the "program" and BroadMind executes interventions through it adaptively.
- **Causal program induction**: BroadMind's latent programs could learn to represent causal interventions directly — each latent instruction becomes a do-operator application.
- **Joint ONNX export**: Extend BroadMind v0.78's edge deployment pipeline to export the combined system.

### Long-Term
- **Self-improving causal reasoning**: BroadMind's latent program induction discovers new causal relationships that Causeway's DAG hasn't captured, feeding back to update the causal graph.
- **Causal world model**: The combined system becomes a full causal world model — it understands structure (Causeway), executes programs (BroadMind), and predicts consequences (both together).

---

*Integration designed February 2026*
*Causeway v0.1.0 + BroadMind v0.77 (Elastic Inference)*
*Bridge overhead: 60,064 parameters (4.6% of combined 1.3M total)*
