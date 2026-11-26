# GenInherit: Google Colab Free-Tier Optimized Training Playbook
## 3-Stage Progressive Growth Ladder (~30M → ~55M → ~82M)

```text
====================================================================================================
  GENINHERIT-LLM | COLAB-OPTIMIZED PROGRESSIVE TRAINING SCHEDULE
  STRATEGY: Split the heavy 82M model into three bite-sized, 45-to-90 minute Colab sessions.
            Zero compute wasted, 100% immune to daily Colab GPU timeout/quota resets.
====================================================================================================
```

---

## 1. The 3-Stage Master Schedule at a Glance

| Stage | Model Configuration | Layers | Hidden Dim | Parameter Count | Training Target | GPU Time on Colab T4 | Memory Footprint | Colab Quota Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (Foundation)** | `r00_30m.yaml` | **4 layers** | 768 | **28.1 Million** | 150 Million tokens | **~45 minutes** | ~4.2 GB VRAM | **Zero Risk** (well under limit) |
| **Stage 2 (Expansion)** | `r01_55m.yaml` | **8 layers** | 768 | **52.9 Million** | 250 Million tokens | **~1 hour 15 min** | ~6.5 GB VRAM | **Zero Risk** |
| **Stage 3 (Full Base)** | `r01_82m.yaml` | **12 layers** | 768 | **82.0 Million** | 350 Million tokens | **~1 hour 45 min** | ~9.2 GB VRAM | **Safe & Resumable** |

---

## 2. Why This New Schedule is 10x Smarter for Free Colab:

1. **Short, Bite-Sized Sessions:** No single run ever exceeds 1.5 hours. You can train one stage per day (or morning vs. evening) without ever hitting Colab's daily GPU ban!
2. **Instant Gratification:** In just **45 minutes**, you will have your first fully working, trained neural inheritance model (`r00_30m_end.pt`).
3. **Function-Preserving Identity Growth (`grow.py`):**
   - Stage 1 $\to$ Stage 2 does **NOT** restart from scratch!
   - We run `grow_and_train.py`, which copies the 4 trained layers, adds 4 new identity layers with zero-residual projection, and starts Stage 2 with **zero loss shock** ($x + 0 = x$).
   - Stage 2 $\to$ Stage 3 adds the remaining layers seamlessly.

---

## 3. Step-by-Step Colab Execution Guide

### Day 1 / Session 1: Train Stage 1 (~30M Parameters)
*Time required: ~45 minutes*

```python
# Launch Stage 1 Base Training (28.1M Parameters)
!python /content/drive/MyDrive/geninherit/code/train_base.py \
    --config /content/drive/MyDrive/geninherit/code/configs/r00_30m.yaml \
    --data_dir /content/drive/MyDrive/geninherit/data/shards \
    --out_dir /content/drive/MyDrive/geninherit/runs/r00_30m \
    --session_hours 2.0
```
**Output:** Saves final checkpoint to `/content/drive/MyDrive/geninherit/runs/r00_30m/geninherit_r00_30m_end.pt`.

---

### Day 2 / Session 2: Grow to Stage 2 (~55M Parameters)
*Time required: ~1 hour 15 min*

```python
# Grow from 4 layers (30M) to 8 layers (55M) with identity initialization
!python /content/drive/MyDrive/geninherit/code/grow_and_train.py \
    --source_ckpt /content/drive/MyDrive/geninherit/runs/r00_30m/geninherit_r00_30m_end.pt \
    --target_config /content/drive/MyDrive/geninherit/code/configs/r01_55m.yaml \
    --data_dir /content/drive/MyDrive/geninherit/data/shards \
    --out_dir /content/drive/MyDrive/geninherit/runs/r01_55m
```
**Output:** Saves final checkpoint to `/content/drive/MyDrive/geninherit/runs/r01_55m/geninherit_r01_55m_end.pt`.

---

### Day 3 / Session 3: Scale to Stage 3 (~82M Parameters)
*Time required: ~1 hour 45 min*

```python
# Grow from 8 layers (55M) to 12 layers (82M)
!python /content/drive/MyDrive/geninherit/code/grow_and_train.py \
    --source_ckpt /content/drive/MyDrive/geninherit/runs/r01_55m/geninherit_r01_55m_end.pt \
    --target_config /content/drive/MyDrive/geninherit/code/configs/r01_82m.yaml \
    --data_dir /content/drive/MyDrive/geninherit/data/shards \
    --out_dir /content/drive/MyDrive/geninherit/runs/r01_82m
```
**Output:** Master base model completed!

---

## 4. Run Evaluation Gates (Test Validation Accuracy)

At the end of any stage (even after Stage 1!), you can run this command to evaluate your model on unseen test families:

```python
!python /content/drive/MyDrive/geninherit/code/run_eval_gates.py \
    --checkpoint /content/drive/MyDrive/geninherit/runs/r00_30m/geninherit_r00_30m_end.pt \
    --data_dir /content/drive/MyDrive/geninherit/data/shards
```

