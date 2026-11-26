# GenInheritAI-LLM Implementation

Colab-ready training and data preparation pipeline for **GenInherit-Base**, a dual-stream decoder-only transformer for multi-generational genomic inheritance reasoning.

## Structure
- `Phase-1_Data_Preparation/` — Public genomic dataset download (ClinVar, HPO, MANE, ClinGen), synthetic pedigree cohort generation, 4,096-token BioDSL tokenizer, PSTS-v1 serialization with Wright-Malecot kinship Fourier channels, and binary `.pt` sharding.
- `Phase-2_Model_Training/` — Dual-stream GQA + RoPE transformer (`GenInheritBase`), float32-stable composite loss, resumable fp16 training loop, progressive depth growth (`grow.py`), and Colab notebooks/configs (30M -> 55M -> 82M -> 248M).
