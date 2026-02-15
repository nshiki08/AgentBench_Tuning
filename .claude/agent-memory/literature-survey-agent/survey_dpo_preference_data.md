# Literature Survey: DPO Preference Data Construction for AgentBench

## Survey Metadata
- Date: 2026-02-15
- Focus: DPO preference pair construction for DB Bench and ALFWorld tasks
- Model Scale: Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507
- Constraint: AgentBench data/environment cannot be used for training
- Knowledge Cutoff: May 2025 (recommend verifying latest papers on arXiv)

---

## Key Findings Summary

1. **Best-of-N Rejection Sampling** is the most practical and well-validated approach for constructing DPO pairs in agent settings. Generating K=8-16 completions per prompt and selecting highest/lowest scored pairs is standard practice (Touvron et al. 2023, Yuan et al. 2024).

2. **Execution-based rewards (DB tasks)** provide the strongest signal for DPO pair construction -- SQL execution match against gold is deterministic and unambiguous, making it ideal for preference labeling.

3. **Teacher-Student gap pairs** using a strong model (e.g., Qwen2.5-72B) for chosen and the training model (Qwen2.5-7B) for rejected is an effective complementary strategy, especially when self-play produces pairs with insufficient score margin.

4. **Iterative DPO (online DPO / Iterative RLHF)** with 2-3 rounds of pair generation and training significantly outperforms single-round DPO. Each round uses the latest policy to generate new pairs.

5. **Multi-turn DPO** remains challenging -- the DPO loss was designed for single-turn preference. The most effective approach for multi-turn agent trajectories is to apply DPO at the trajectory level (treating the entire agent response sequence as a single "completion") or to use step-level DPO (S-DPO).

---

## 1. DPO Preference Data Construction Methods

### 1.1 Best-of-N Rejection Sampling (Standard Approach)

**Core Method**: Generate N completions per prompt using the current policy model, score each with a reward function, and select the best/worst as chosen/rejected.

**Key Papers**:

#### RAFT: Reward rAnked FineTuning (Dong et al., 2023)
- **arXiv**: 2304.06767
- **Method**: Generate K responses, rank by reward, select top-1 for SFT (can be extended to DPO by selecting top-1 and bottom-1)
- **Results**: Competitive with PPO on summarization/dialogue tasks
- **Relevance**: High -- the exact framework the current `03_generate_dpo_pairs_v1.py` implements

#### SPIN: Self-Play Fine-Tuning (Chen et al., 2024)
- **arXiv**: 2401.01335
- **Method**: Uses the model's own generations as rejected and ground-truth as chosen. Iterates: each round the model improves, making the "rejected" harder to distinguish from "chosen."
- **Results**: On MT-Bench, improves Zephyr-7B by ~1 point over 3 iterations
- **Relevance**: Medium -- requires ground-truth responses (available for DB tasks via gold SQL, harder for Env)

#### Self-Rewarding LLMs (Yuan et al., 2024)
- **arXiv**: 2401.10020
- **Method**: The model itself generates reward scores (LLM-as-Judge) to construct preference pairs. No external reward model needed.
- **Results**: Llama-2-70B-Chat outperforms Claude and GPT-4 on certain benchmarks after iterative self-rewarding
- **Relevance**: Medium -- could use Qwen2.5-72B as the judge model (it is on the whitelist)

#### Statistical Rejection Sampling (RSO, Liu et al., 2024)
- **arXiv**: 2309.06657
- **Method**: Instead of best/worst, samples pairs from the optimal policy distribution using importance sampling. More statistically grounded pair selection.
- **Relevance**: Low-Medium -- more complex implementation, marginal gains

### 1.2 Teacher-Student Distillation Pairs

**Core Method**: Use a strong teacher model to generate chosen responses and the weaker student model for rejected responses.

#### Zephyr (Tunstall et al., 2023)
- **arXiv**: 2310.16944
- **Method**: dSFT (distilled SFT) followed by dDPO. Uses GPT-4 responses as chosen and the SFT model's responses as rejected. Applied UltraFeedback dataset.
- **Results**: Zephyr-7B-beta achieved 90.6% on AlpacaEval, competitive with much larger models
- **Relevance**: High -- directly applicable. Use Qwen2.5-72B or GPT-OSS-120B as teacher

#### Constitutional AI / RLAIF (Bai et al., 2022; Lee et al., 2023)
- **arXiv**: 2212.08073, 2309.00267
- **Method**: Use AI feedback instead of human feedback. The model generates multiple responses, and a separate (or same) model judges which is better.
- **Relevance**: High -- whitelist models can serve as judges

### 1.3 LLM-as-Judge for Preference Labeling

**Core Method**: Instead of using a scalar reward function, use a strong LLM to compare two completions and decide which is better.

#### Judging LLM-as-a-Judge (Zheng et al., 2023)
- **arXiv**: 2306.05685
- **Method**: GPT-4/Claude compare pairs of responses, provide ratings (1-10) or pairwise preferences
- **Results**: Strong agreement with human preference (>80% on MT-Bench)
- **Relevance**: High -- whitelist model (Qwen2.5-72B, Qwen3-32B) can serve as judge

**Practical Implementation for This Project**:
```
For each prompt:
  1. Generate 4-8 completions from SFT model (temperature=0.8)
  2. Score with task-specific reward (SQL execution / ReAct format)
  3. Optionally: have Qwen2.5-72B rank the top-2 candidates
  4. Select chosen (highest composite score) / rejected (lowest)
```

### 1.4 Contrastive Pair Construction

#### Kahneman-Tversky Optimization (KTO, Ethayarajh et al., 2024)
- **arXiv**: 2402.01306
- **Method**: Does not require paired data. Only needs binary labels (good/bad) per response.
- **Results**: Competitive with DPO while requiring less structured data
- **Relevance**: Medium -- fallback if pair construction proves difficult for Env tasks

#### IPO: A General Theoretical Paradigm (Azar et al., 2024)
- **arXiv**: 2310.12036
- **Method**: More theoretically grounded variant of DPO. Uses a different loss function that avoids over-optimization.
- **Relevance**: Low-Medium -- minor implementation change in trl (loss_type="ipo")

---

## 2. Agent-Task-Specific DPO Research

### 2.1 Multi-Turn Agent DPO

#### RLHF for Multi-Turn Dialogue (Jang et al., 2023)
- **Core Insight**: Standard DPO treats the entire conversation as a single unit. For agent tasks, this is problematic because the reward signal is sparse (only at the end).
- **Solution**: Apply DPO at the step level -- compare individual (Thought, Action) steps rather than entire trajectories.

#### Step-DPO (Lai et al., 2024)
- **arXiv**: 2406.18629
- **Method**: Apply DPO at each reasoning step rather than the final answer. Significantly improves mathematical reasoning (MATH benchmark).
- **Results**: +2-3% on MATH over standard DPO
- **Relevance**: High for agent tasks -- each agent turn (Thought+Action) can be treated as a "step"

#### Turn-Level DPO for Agents
- **Concept** (synthesized from multiple papers): For multi-turn agent interactions, construct pairs at the turn level:
  - Same conversation history up to turn t
  - chosen: Turn t that leads to successful subsequent trajectory
  - rejected: Turn t that leads to failure or suboptimal outcome
- **Implementation**: Requires rollout from each turn to evaluate downstream consequences

### 2.2 ReAct-Format DPO

#### FireAct (Chen et al., 2023)
- **arXiv**: 2305.16291
- **Method**: Fine-tunes models on diverse agent trajectories (ReAct, CoT, ReflectAct). Doesn't use DPO directly but provides the trajectory data format.
- **Relevance**: The trajectory format from FireAct can be directly used to construct DPO pairs by comparing successful vs. failed trajectories.

#### Agent-FLAN Negative Sampling (Chen et al., 2024)
- **arXiv**: 2403.12881
- **Method**: Explicitly constructs negative examples (wrong tool calls, hallucinated actions, format violations) as training data.
- **DPO Extension**: These negative examples serve as excellent "rejected" samples for DPO:
  - Chosen: Correct ReAct format with valid action
  - Rejected: Hallucinated action, wrong format, or invalid tool call

#### AgentTuning Trajectory Quality (Zeng et al., 2023)
- **arXiv**: 2310.12823
- **Method**: Filters GPT-4 trajectories by task completion success
- **DPO Extension**: Successful trajectories -> chosen, failed trajectories -> rejected

### 2.3 Text-to-SQL DPO

#### Preference-Guided SQL Generation (synthesized from literature)
- **Method**: For DB tasks, DPO pairs have a natural construction:
  - Chosen: SQL that executes correctly and returns the right result
  - Rejected: SQL that either fails to execute, returns wrong results, or has syntax errors
- **Advantage**: Completely automated -- no human annotation needed. SQL execution is deterministic.

#### DAIL-SQL with Self-Consistency (Gao et al., 2023)
- **arXiv**: 2308.15363
- **Method**: Generates multiple SQL candidates and uses execution consistency for selection
- **DPO Extension**: The consistent-execution SQL is chosen, the inconsistent ones are rejected

#### DTS-SQL: Decomposed Text-to-SQL with Small Language Models (Pourreza & Rafiei, 2024)
- **arXiv**: 2402.01117
- **Method**: Decomposes text-to-SQL into schema linking + SQL generation + self-correction
- **Relevance**: The self-correction step naturally produces (initial_sql, corrected_sql) pairs usable for DPO

### 2.4 Embodied Agent DPO (ALFWorld-Adjacent)

#### Embodied Planner-R1 (2025)
- **arXiv**: 2506.23127
- **Method**: Uses pure RL (GRPO) for ALFWorld achieving 97.78% success rate
- **Insight**: RL is preferred over DPO for embodied tasks, but DPO can serve as a warm-start

#### LEMA: Learning from Mistakes (An et al., 2024)
- **arXiv**: 2310.20689
- **Method**: Collects mistake trajectories and correct trajectories, fine-tunes on corrections
- **DPO Extension**: (correct_trajectory, mistake_trajectory) pairs

#### Trial and Error (Song et al., 2024)
- **Method**: Agent explores, collects success/failure trajectories, then learns from the contrast
- **Relevance**: High -- directly produces DPO-compatible trajectory pairs

---

## 3. Available Data Sources

### 3.1 Text-to-SQL Datasets (DB Bench Direction)

| Dataset | Size | Multi-Turn | SQL Complexity | License | DPO Suitability |
|---------|------|------------|----------------|---------|-----------------|
| **Spider** | 10,181 Q | No (convertible) | Medium-Hard | CC-BY-SA 4.0 | High -- gold SQL for execution-based pairing |
| **SParC** | 4,298 dialogues | Yes | Medium | CC-BY-SA 4.0 | Very High -- native multi-turn |
| **CoSQL** | 3,007 dialogues | Yes | Medium-Hard | CC-BY-SA 4.0 | Very High -- conversational SQL |
| **BIRD** | 12,751 Q | No (convertible) | Hard (real-world) | CC-BY-SA 4.0 | High -- challenging SQL for diverse pairs |
| **WikiSQL** | 80,654 Q | No | Simple | BSD-3 | Low -- too simple for meaningful DPO |
| **gretelai/synthetic_text_to_sql** | 100K+ | No | Variable | Apache 2.0 | Medium -- quality varies |
| **CHASE** | 17,940 Q | No | Medium | - | Medium -- Chinese SQL |
| **KaggleDBQA** | 400 Q | No | Real-world | CC-BY-SA 4.0 | Low (small) but real-world quality |

**Recommendation for DPO Pairs (DB)**:
1. **Primary**: Spider + SParC + CoSQL (gold SQL available, multi-turn, well-studied)
2. **Supplementary**: BIRD (more challenging SQL, better for creating discriminative pairs)
3. **Volume**: gretelai/synthetic_text_to_sql (if more volume needed)

### 3.2 Environment Interaction Datasets (ALFWorld Direction)

| Dataset | Size | Environment Type | ALFWorld Similarity | DPO Suitability |
|---------|------|------------------|---------------------|-----------------|
| **agent-eto/eto-sft-trajectory** | ~50K | Mixed (WebShop, SciWorld, ALFWorld) | High (must exclude ALFWorld!) | High (WebShop/SciWorld portions) |
| **AgentBank** (arXiv:2410.07706) | 50K+ | Mixed (ALFWorld, WebShop, etc.) | Contains ALFWorld -- must filter | Medium (after filtering) |
| **WebShop** (Yao et al., 2022) | 12K+ | Web shopping | Medium | High -- ReAct format, goal-directed |
| **ScienceWorld** | Variable | Science experiments | Medium (text-based, goal-directed) | High |
| **Jericho** (Hausknecht et al.) | Variable | Text adventure games | Medium-High | Medium -- requires environment access |
| **BabyAI** | Variable | Grid navigation | Low-Medium | Low -- too different from ALFWorld |
| **ToolBench** (Qin et al., 2024) | 200K+ | Tool-use APIs | Low | Low for ALFWorld, Medium for general agent |
| **InterCode** (Yang et al., 2024) | ~2K | Code execution | Low | Low |

**Constraint**: ALFWorld/TextWorld data strictly excluded from training.

**Recommendation for DPO Pairs (Env)**:
1. **Primary**: WebShop trajectories (ReAct format, goal-directed, closest to ALFWorld without being ALFWorld)
2. **Primary**: ScienceWorld trajectories (multi-step reasoning in text environments)
3. **Supplementary**: Synthetic ReAct trajectories (household/navigation templates, already implemented in `01_build_training_set_v1.py`)

### 3.3 Agent Trajectory Datasets

| Dataset | Source | Format | Filtering Needed | DPO Pair Strategy |
|---------|--------|--------|-------------------|-------------------|
| **agent-eto/eto-sft-trajectory** | ETO project | ChatML | Yes (exclude ALFWorld) | Generate pairs from success/failure split |
| **AgentInstruct** (Microsoft, 2024) | Synthetic | ChatML | No | Pre-labeled quality tiers |
| **Agent Lumos** training data | Lumos project | Structured | No | Planning/Grounding separation |
| **glaiveai/glaive-function-calling-v2** | Synthetic | Function calling | No | Tool-use DPO pairs |
| **Gorilla API trajectories** | Berkeley | API calls | No | Correct/incorrect API call pairs |

---

## 4. Practical Pair Generation Strategies

### 4.1 Strategy A: Self-Play Best-of-N (Current Implementation)

This is what `03_generate_dpo_pairs_v1.py` already implements.

**Process**:
1. Load SFT model (Qwen2.5-7B + LoRA adapter)
2. For each prompt: generate K=8 completions at temperature=0.8
3. Score each with task-specific reward:
   - DB: SQL execution match (gold available) -> {1.0, Jaccard*0.5, -0.25, -0.5}
   - Env: ReAct format (0.4) + action validity (0.3) + reasoning quality (0.3)
4. Select highest/lowest scoring pair (min_diff >= 0.1)

**Strengths**:
- Fully automated, no external API calls
- On-policy: pairs come from the model being trained
- DB reward is strong (deterministic SQL execution)

**Weaknesses**:
- Env reward is weak (heuristic-only, no actual environment feedback)
- K=8 may not produce sufficient diversity on small models
- Computationally expensive: 8 generations per prompt

**Expected Yield**: ~60-70% of prompts produce valid pairs (some will have all-similar scores)

### 4.2 Strategy B: Teacher-Student Pairs

**Process**:
1. For each prompt: generate 1 response from Qwen2.5-72B (or GPT-OSS-120B) -> chosen
2. Generate 1-4 responses from Qwen2.5-7B (SFT model) -> select worst as rejected
3. Verify: chosen_score > rejected_score with sufficient margin

**Implementation with Whitelist Models**:
```python
# Teacher model options (from whitelist):
TEACHER_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",  # Best quality for DB tasks
    "openai/gpt-oss-120b",          # Largest available
    "Qwen/Qwen3-32B",               # Good balance of quality/speed
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",  # Especially good for SQL
]
```

**Strengths**:
- Consistently high-quality chosen responses
- Large quality gap between chosen and rejected
- Works well even when the student model is weak

**Weaknesses**:
- API costs / inference time for teacher model
- Off-policy: chosen comes from a different distribution
- Risk of the student learning teacher-specific patterns

### 4.3 Strategy C: Hybrid (Recommended)

**Process**:
1. **DB Pairs (500-800)**:
   - Phase 1: Self-play K=8 from SFT model on Spider/SParC/CoSQL prompts
   - SQL execution scoring (deterministic, high quality)
   - Expected yield: ~400-500 pairs
   - Phase 2 (if needed): Teacher (Qwen2.5-72B) generates correct SQL -> chosen, student errors -> rejected
   - Additional ~100-200 pairs

2. **Env Pairs (300-500)**:
   - Phase 1: Self-play K=8 from SFT model on WebShop/ScienceWorld/synthetic prompts
   - Heuristic ReAct scoring
   - Expected yield: ~200-300 pairs
   - Phase 2: Teacher (Qwen2.5-72B) generates high-quality ReAct trajectories -> chosen
   - Student generates suboptimal trajectories -> rejected
   - LLM-as-Judge (Qwen3-32B) validates pair quality
   - Additional ~100-200 pairs

3. **General Pairs (optional, 100-200)**:
   - From UltraFeedback or similar pre-existing preference datasets
   - Maintains general capability during DPO

**Total Target**: 800-1,300 pairs (600-800 DB + 300-500 Env)

### 4.4 Strategy D: Iterative Online DPO

Based on the iterative approach in the RL plan v1 (Option C).

**Process**:
```
Round 1:
  SFT model -> generate K=8 -> score -> 600-800 pairs -> DPO train (1 epoch)

Round 2:
  DPO-round1 model -> generate K=8 -> score -> 400-600 pairs -> DPO train (1 epoch)

Round 3 (optional):
  DPO-round2 model -> generate K=8 -> score -> 300-500 pairs -> DPO train (1 epoch)
```

**Key Insight from Literature**: Each round should generate new pairs from the *latest* policy. Reusing old pairs causes distribution shift and degrades performance.

---

## 5. Detailed Pair Generation Pipeline Recommendation

### Phase 1: Prompt Dataset Construction (Prerequisite)

**Already exists**: `02_build_rl_prompts_v1.py` generates prompts. These same prompts can be reused for DPO pair generation.

**DB Prompts (500-800)**:
- Spider train: 200-300 (group by DB, multi-turn conversion)
- SParC train: 150-200 (native multi-turn)
- CoSQL train: 100-150 (conversational SQL)
- BIRD train: 50-100 (harder real-world SQL)
- Each prompt includes: schema, question, gold_sql, db_path (SQLite)

**Env Prompts (300-500)**:
- WebShop tasks: 100-150
- ScienceWorld tasks: 100-150
- Synthetic ReAct templates: 100-200
- Each prompt includes: task description, system prompt with ReAct format

### Phase 2: Pair Generation

#### Option 1: Pure Self-Play (Simplest)

```
SFT Model -> K=8 completions per prompt -> Reward scoring -> Best/Worst pair
```

**Estimated Time** (A100 40GB):
- DB: 800 prompts x 8 completions x ~1024 tokens = ~6.5M tokens
  - At ~200 tokens/sec (4-bit, batch=4): ~9 hours
- Env: 500 prompts x 8 completions x ~1024 tokens = ~4M tokens
  - ~5.5 hours
- **Total**: ~15 hours (can be split across Colab sessions)

**Expected Yield**:
- DB: ~500-600 valid pairs (70-75% yield, SQL execution provides clear signal)
- Env: ~250-350 valid pairs (50-70% yield, heuristic reward is noisier)
- **Total**: ~750-950 pairs

#### Option 2: Teacher-Student Hybrid (Higher Quality)

```
Step 1: Self-play (same as Option 1 but K=4 to save time)
Step 2: For prompts with low self-play yield, use teacher model
```

**Teacher Model Inference** (Qwen2.5-72B-Instruct via API or Qwen3-32B via GGUF):
- 200-300 prompts x 1 completion x ~1024 tokens = ~250K-300K tokens
- Time depends on API/local inference setup

**Expected Yield**: ~900-1,200 pairs (higher quality, especially for Env tasks)

### Phase 3: Pair Quality Assurance

**Filtering Criteria**:
1. `chosen_score - rejected_score >= 0.1` (minimum margin)
2. `len(chosen) > 10 characters` (not degenerate)
3. `len(rejected) > 10 characters` (not degenerate)
4. For DB: chosen SQL must execute without error
5. For Env: chosen must contain valid ReAct format (Thought + Action + Action Input)
6. No contamination (ALFWorld/TextWorld keywords)

**Optional LLM-as-Judge Validation**:
- Sample 100-200 pairs
- Have Qwen3-32B or Qwen2.5-72B evaluate: "Which response is better and why?"
- Check agreement with reward-based selection
- If agreement < 80%, review and adjust reward functions

### Phase 4: DPO Training

Already implemented in `02_dpo_qwen25_7b.py`. Key parameters:
- beta=0.1 (standard)
- loss_type="sigmoid" (standard DPO)
- 1 epoch per round
- LoRA rank 32 (lower than SFT's 64)

---

## 6. Comparative Analysis of Approaches

| Criterion | Self-Play (Option 1) | Teacher-Student (Option 2) | Hybrid (Option 3) | Iterative (Option 4) |
|-----------|---------------------|---------------------------|-------------------|---------------------|
| **Implementation Complexity** | Low | Medium | Medium-High | High |
| **Pair Quality (DB)** | High (SQL execution) | Very High | Very High | Very High |
| **Pair Quality (Env)** | Medium (heuristic) | High | High | Medium-High |
| **API Cost** | None | Medium | Low-Medium | None |
| **Computation Time** | ~15h | ~8h + API | ~12h + API | ~45h (3 rounds) |
| **Expected Pair Count** | 750-950 | 900-1,200 | 900-1,200 | 1,500-2,500 (cumulative) |
| **Research Support** | Strong (RAFT) | Strong (Zephyr) | Strong | Strong (Iterative DPO) |
| **Risk of Distribution Shift** | Low | Medium | Low-Medium | Low (each round is on-policy) |

---

## 7. Specific Recommendations

### Primary Recommendation: Hybrid Approach (Option 3)

**Rationale**: Balances quality, feasibility, and cost. Uses the strong SQL execution signal for DB tasks (self-play is sufficient) and supplements Env tasks with teacher model quality.

### Concrete Data Specification

```
DPO Preference Dataset v1:
  Total Pairs Target: 900-1,200

  DB Pairs: 600-800
    Sources:
      - Spider (gold SQL, 200 DB schemas): 200-300 pairs
      - SParC (multi-turn, 4,298 dialogues): 150-200 pairs
      - CoSQL (conversational, 3,007 dialogues): 100-150 pairs
      - BIRD (real-world, challenging SQL): 50-100 pairs
      - gretelai/synthetic_text_to_sql: 50-100 pairs (volume filler)
    Pair Construction:
      Primary: Self-play K=8, SQL execution scoring
      Backup: Qwen2.5-72B teacher for low-yield prompts

  Env Pairs: 300-500
    Sources:
      - WebShop trajectories: 100-150 pairs
      - ScienceWorld trajectories: 100-150 pairs
      - Synthetic ReAct templates: 50-100 pairs
      - Additional synthetic household/navigation: 50-100 pairs
    Pair Construction:
      Primary: Self-play K=8, heuristic ReAct scoring
      Quality boost: Qwen2.5-72B teacher for chosen, validated by LLM-as-Judge

  General Pairs (optional): 100-200
    Sources:
      - UltraFeedback (pre-existing preference data): 100-200 pairs
    Purpose: Maintain general capability
```

### Implementation Priority

1. **Phase 1** (Week 1): Generate DB pairs via self-play. This is the highest-value, lowest-risk step because SQL execution provides deterministic rewards.

2. **Phase 2** (Week 1-2): Generate Env pairs via self-play + teacher supplement. Env rewards are noisier, so teacher model quality helps.

3. **Phase 3** (Week 2): Run DPO training round 1. Evaluate on held-out prompts.

4. **Phase 4** (Week 2-3, if time permits): Iterative round 2 with updated model.

---

## 8. Research Gaps and Opportunities

1. **Step-Level DPO for Agent Tasks**: Most DPO work operates at the full-response level. Applying Step-DPO (Lai et al., 2024) to agent trajectories -- comparing individual (Thought, Action) turns rather than full trajectories -- could significantly improve fine-grained agent behavior. This is underexplored.

2. **Environment Simulation for Reward**: The main weakness of the current Env reward is its heuristic nature. A lightweight text-based environment simulator (not ALFWorld, but similar mechanics) could provide more meaningful rewards. WebShop already has this; extending it to household tasks would be valuable.

3. **Cross-Task Transfer in DPO**: Whether DPO pairs from DB tasks help with Env tasks (and vice versa) is unclear. The mixing ratio between DB and Env pairs may matter significantly.

4. **KTO as Alternative**: Kahneman-Tversky Optimization (KTO) only requires binary labels (good/bad) instead of explicit pairs. This could be particularly useful for Env tasks where constructing meaningful pairs is harder.

5. **DPO + GRPO Combination**: Using DPO as a pre-training step before GRPO could combine the stability of DPO with the online exploration of GRPO. This pipeline (SFT -> DPO -> GRPO) is relatively unexplored for agent tasks.

---

## 9. Key References

### DPO Theory and Extensions
- Rafailov et al. (2023) "DPO: Your Language Model is Secretly a Reward Model" - arXiv:2305.18290 (NeurIPS 2023)
- Ethayarajh et al. (2024) "KTO: Model Alignment as Prospect Theoretic Optimization" - arXiv:2402.01306
- Azar et al. (2024) "IPO: A General Theoretical Paradigm" - arXiv:2310.12036
- Lai et al. (2024) "Step-DPO" - arXiv:2406.18629

### Preference Data Construction
- Dong et al. (2023) "RAFT: Reward rAnked FineTuning" - arXiv:2304.06767
- Chen et al. (2024) "SPIN: Self-Play Fine-Tuning" - arXiv:2401.01335
- Yuan et al. (2024) "Self-Rewarding Language Models" - arXiv:2401.10020
- Tunstall et al. (2023) "Zephyr: Direct Distillation of LM Alignment" - arXiv:2310.16944
- Zheng et al. (2023) "Judging LLM-as-a-Judge" - arXiv:2306.05685

### Agent-Specific
- Zeng et al. (2023) "AgentTuning" - arXiv:2310.12823
- Chen et al. (2024) "Agent-FLAN" - arXiv:2403.12881
- Chen et al. (2023) "FireAct" - arXiv:2305.16291
- Song et al. (2024) "Trial and Error: Agent Self-Improvement"

### Text-to-SQL
- Gao et al. (2023) "DAIL-SQL" - arXiv:2308.15363
- Pourreza & Rafiei (2024) "DTS-SQL" - arXiv:2402.01117
- Li et al. (2024) "CodeS" - SIGMOD 2024

### Iterative/Online DPO
- Xu et al. (2024) "Online DPO" - various implementations in trl
- Guo et al. (2024) "Direct Language Model Alignment from Online AI Feedback" - arXiv:2402.04792
