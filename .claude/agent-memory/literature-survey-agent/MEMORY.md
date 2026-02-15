# Literature Survey Agent - Memory

## Project Context
- Goal: Improve AgentBench scores (DB Bench, ALFWorld) using small models (Qwen2.5-7B, Qwen3-4B)
- Environment: Google Colab, LoRA/QLoRA training
- Notebook format: marimo (.py) -> ipynb conversion

## Key Findings (2026-02-14 Survey)
- AgentTuning (THUDM) is the most directly relevant method: GPT-4 trajectory SFT + general data mixing
- Agent-FLAN's negative sampling reduces hallucination significantly
- SFT + DPO/GRPO two-stage training outperforms SFT alone for agent tasks
- Qwen3's `<think>` mode naturally maps to ReAct's Thought step
- Data quality > quantity: 1-2K high-quality trajectories can achieve major improvements
- Mixing ratio (agent:general = 1:3-5) is critical to maintain general capabilities

## Detailed Survey Files
- [Full Survey](survey_agentbench_small_models.md) - Comprehensive literature review
- [DPO Preference Data Survey](survey_dpo_preference_data.md) - DPO pair construction methods for agent tasks

## Key Papers Registry
- AgentBench: arXiv:2308.03688 (ICLR 2024)
- AgentTuning: arXiv:2310.12823 (most relevant for our approach)
- Agent-FLAN: arXiv:2403.12881 (negative sampling insight)
- FireAct: arXiv:2305.16291 (multi-style trajectory mixing)
- ReAct: arXiv:2210.03629 (base format for agent data)
- Reflexion: arXiv:2303.11366 (self-reflection patterns)
- CodeS: SIGMOD 2024 (small model text-to-SQL, highly relevant for DB Bench)
- Agent Lumos: arXiv:2311.09593 (planning/grounding separation)

## DPO Key Findings (2026-02-15 Survey)
- Best-of-N Rejection Sampling (K=8) is the most practical DPO pair construction method
- SQL execution-based rewards give the strongest signal for DB DPO pairs (deterministic, unambiguous)
- Teacher-Student pairing (Qwen2.5-72B chosen + Qwen2.5-7B rejected) is effective when self-play yield is low
- Iterative DPO (2-3 rounds, regenerating pairs each round) significantly outperforms single-round
- Step-DPO (per-turn DPO) is promising for agent tasks but underexplored
- KTO (binary labels only) is a viable alternative when pairing is difficult (e.g., Env tasks)
- Target: 900-1,200 pairs total (600-800 DB + 300-500 Env)

## Recommended Training Pipeline
1. Phase 1: Trajectory collection via GPT-4/Claude API (DB Bench: 500-1K, ALFWorld: 300-500)
2. Phase 2: SFT with LoRA/QLoRA + general data mixing
3. Phase 3: DPO/GRPO with self-generated success/failure pairs

## Tool Access Notes
- WebSearch and WebFetch are denied in this environment
- All survey content is based on training data (cutoff: May 2025)
- Recommend user verify latest papers on arXiv/Semantic Scholar
