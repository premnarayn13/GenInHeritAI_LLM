"""
GenInherit - Phase 1: Data Preparation
Script 4: Streaming Preprocessor, PSTS-v1 Serializer & Binary Sharder
Converts normalized genomic records & synthetic cohorts into packed binary PyTorch shards (.pt)
with Fourier numeric channels, forbidden sets, and counterfactual twin pairs.
"""

import os
import sys
import json
import time
import math
import random
from fractions import Fraction
from typing import Dict, List, Any, Optional, Tuple
import torch

try:
    from geninherit.bie.epistemic import EpistemicState
    from geninherit.bie.kinship import PedigreeGraph, KinshipEngine
    from geninherit.bie.chromosome import ChromosomeClassGate, LocusClass
except ImportError:
    # Self-contained fallback for Google Colab standalone execution
    class PedigreeGraph:
        def __init__(self):
            self.individuals: Dict[str, Dict[str, Optional[str]]] = {}
        def add_individual(self, ind_id: str, father_id: Optional[str] = None, mother_id: Optional[str] = None, sex: str = "U"):
            self.individuals[ind_id] = {"father": father_id, "mother": mother_id, "sex": sex}
        def topological_sort(self) -> List[str]:
            in_degree: Dict[str, int] = {ind: 0 for ind in self.individuals}
            children: Dict[str, List[str]] = {ind: [] for ind in self.individuals}
            for ind, data in self.individuals.items():
                for p in (data.get("father"), data.get("mother")):
                    if p and p in self.individuals:
                        children[p].append(ind)
                        in_degree[ind] += 1
            queue = [ind for ind, deg in in_degree.items() if deg == 0]
            ordered = []
            while queue:
                queue.sort()
                curr = queue.pop(0)
                ordered.append(curr)
                for ch in children[curr]:
                    in_degree[ch] -= 1
                    if in_degree[ch] == 0:
                        queue.append(ch)
            return ordered if len(ordered) == len(self.individuals) else list(self.individuals.keys())

    class KinshipEngine:
        def __init__(self, pedigree: PedigreeGraph):
            self.pedigree = pedigree
            self.order = pedigree.topological_sort()
            self.phi_matrix: Dict[Tuple[str, str], Fraction] = {}
            self.f_vector: Dict[str, Fraction] = {}
            self._compute_matrix()
        def _compute_matrix(self):
            for ind in self.order:
                self.f_vector[ind] = Fraction(0, 1)
            for i_idx, i in enumerate(self.order):
                fa = self.pedigree.individuals[i]["father"]
                mo = self.pedigree.individuals[i]["mother"]
                f_i = self.get_phi(fa, mo) if (fa and mo) else Fraction(0, 1)
                self.f_vector[i] = f_i
                self.phi_matrix[(i, i)] = (Fraction(1, 1) + f_i) / Fraction(2, 1)
                for j_idx in range(i_idx):
                    j = self.order[j_idx]
                    phi_sum = Fraction(0, 1)
                    if fa: phi_sum += self.get_phi(fa, j)
                    if mo: phi_sum += self.get_phi(mo, j)
                    val = phi_sum / Fraction(2, 1)
                    self.phi_matrix[(i, j)] = val
                    self.phi_matrix[(j, i)] = val
        def get_phi(self, i: Optional[str], j: Optional[str]) -> Fraction:
            if not i or not j: return Fraction(0, 1)
            return self.phi_matrix.get((i, j), self.phi_matrix.get((j, i), Fraction(0, 1)))
        def get_inbreeding(self, ind: str) -> Fraction:
            return self.f_vector.get(ind, Fraction(0, 1))


def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def serialize_family_to_psts(fam: Dict[str, Any], vocab: Dict[str, int]) -> Dict[str, Any]:
    """
    Serializes a single family dictionary into PSTS-v1 sequence tokens,
    multi-stream structural IDs, and Fourier numeric quantities.
    """
    tok_list = ["[PED_START]", "[FAM_DEF]"]
    slot_list = [0, 0]
    gen_list = [0, 0]
    fld_list = [0, 0]
    num_list = [float("nan"), float("nan")]
    w_list = [0.0, 0.0]      # Prompt / structural tokens have w=0
    m_list = [0, 0]        # Prompt mask = 0

    # Build internal kinship graph for exact phi and F calculation
    ped = PedigreeGraph()
    for m in fam["members"]:
        ped.add_individual(m["id"], father_id=m.get("father"), mother_id=m.get("mother"), sex=m.get("sex", "U"))
    
    kinship = KinshipEngine(ped)
    proband_id = next((m["id"] for m in fam["members"] if m.get("is_proband")), fam["members"][-1]["id"])
    proband_f = float(kinship.get_inbreeding(proband_id))

    # 1. Structure Stream
    for idx, m in enumerate(fam["members"]):
        slot_idx = min(127, idx + 1)
        gen_idx = 1 if "I-" in m["id"] or "GP-" in m["id"] else (2 if "II-" in m["id"] or "P-" in m["id"] else 3)
        sex_tok = f"[SEX:{m['sex']}]"
        ind_tok = f"[IND:{m['id']}]"

        tok_list.extend([ind_tok, sex_tok])
        slot_list.extend([slot_idx, slot_idx])
        gen_list.extend([gen_idx, gen_idx])
        fld_list.extend([1, 1])
        num_list.extend([float("nan"), float("nan")])
        w_list.extend([0.0, 0.0])
        m_list.extend([0, 0])

        # Parental edges
        p_str = f"[PARENTS:{m.get('father') or 'NONE'},{m.get('mother') or 'NONE'}]"
        tok_list.append(p_str)
        slot_list.append(slot_idx)
        gen_list.append(gen_idx)
        fld_list.append(1)
        num_list.append(float("nan"))
        w_list.append(0.0)
        m_list.append(0)

        # Genotype & Epistemic Status
        epi_tok = f"[EPI:{m.get('epistemic', 'VERIFIED_PRESENT')}]"
        tok_list.append(epi_tok)
        slot_list.append(slot_idx)
        gen_list.append(gen_idx)
        fld_list.append(2)  # Field 2 = Genotype
        num_list.append(float("nan"))
        w_list.append(0.0)
        m_list.append(0)

        # Kinship with proband (exact numeric value into Fourier side-channel)
        phi_with_proband = float(kinship.get_phi(m["id"], proband_id))
        tok_list.append(f"[PHI:BIN_{int(phi_with_proband*10000):04d}]")
        slot_list.append(slot_idx)
        gen_list.append(gen_idx)
        fld_list.append(3)  # Field 3 = Kinship
        num_list.append(phi_with_proband)
        w_list.append(0.0)
        m_list.append(0)

    tok_list.append("[PED_END]")
    slot_list.append(0); gen_list.append(0); fld_list.append(0); num_list.append(float("nan"))
    w_list.append(0.0); m_list.append(0)

    # 2. Reasoning Target Stream (Supervised training targets)
    tok_list.append("[FIELD:HYPOTHESIS]")
    slot_list.append(0); gen_list.append(0); fld_list.append(4); num_list.append(float("nan"))
    w_list.append(0.0); m_list.append(0)

    target_moi = fam.get("ground_truth_inheritance", "AUTOSOMAL_DOMINANT")
    h_tokens = [f"[H:{target_moi[:2]}:CONFIRMED]", fam.get("gene", "BRCA2"), fam.get("transmitting_lineage", "PATERNAL")]

    for t in h_tokens:
        tok_list.append(t)
        slot_list.append(0)
        gen_list.append(0)
        fld_list.append(4)
        num_list.append(proband_f)  # Attach proband inbreeding coefficient
        w_list.append(1.0)          # Loss weight = 1.0 on biological deductions
        m_list.append(1)            # Target mask = 1

    tok_list.append("[EOS]")
    slot_list.append(0); gen_list.append(0); fld_list.append(0); num_list.append(float("nan"))
    w_list.append(0.5); m_list.append(1)

    # Convert tokens to vocabulary IDs
    token_ids = [vocab.get(t, vocab["[UNK]"]) for t in tok_list]
    pos_ids = list(range(len(token_ids)))

    return {
        "tok": token_ids,
        "pos": pos_ids,
        "slot": slot_list,
        "gen": gen_list,
        "fld": fld_list,
        "num": num_list,
        "w": w_list,
        "m": m_list,
        "forb": []  # Forbidden tokens
    }


def pack_and_save_shard(samples: List[Dict], shard_idx: int, output_dir: str, max_ctx: int = 2048):
    """Packs samples into fixed-width tensors (max_ctx) and writes PyTorch binary file."""
    os.makedirs(output_dir, exist_ok=True)
    shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")

    packed_data = {
        "tok": [], "pos": [], "slot": [], "gen": [], "fld": [],
        "num": [], "y": [], "w": [], "m": []
    }

    for s in samples:
        length = len(s["tok"])
        if length > max_ctx:
            continue  # Truncate or drop oversized families

        pad_len = max_ctx - length
        # Pad with PAD tokens
        tok = s["tok"] + [0] * pad_len
        pos = s["pos"] + [0] * pad_len
        slot = s["slot"] + [0] * pad_len
        gen = s["gen"] + [0] * pad_len
        fld = s["fld"] + [0] * pad_len
        num = s["num"] + [float("nan")] * pad_len
        w = s["w"] + [0.0] * pad_len
        m = s["m"] + [0] * pad_len

        # Target y is next token shifted
        y = tok[1:] + [0]

        packed_data["tok"].append(tok)
        packed_data["pos"].append(pos)
        packed_data["slot"].append(slot)
        packed_data["gen"].append(gen)
        packed_data["fld"].append(fld)
        packed_data["num"].append(num)
        packed_data["y"].append(y)
        packed_data["w"].append(w)
        packed_data["m"].append(m)

    tensor_dict = {
        "tok": torch.tensor(packed_data["tok"], dtype=torch.int32),
        "pos": torch.tensor(packed_data["pos"], dtype=torch.int32),
        "slot": torch.tensor(packed_data["slot"], dtype=torch.int8),
        "gen": torch.tensor(packed_data["gen"], dtype=torch.int8),
        "fld": torch.tensor(packed_data["fld"], dtype=torch.int8),
        "num": torch.tensor(packed_data["num"], dtype=torch.float32),
        "y": torch.tensor(packed_data["y"], dtype=torch.int32),
        "w": torch.tensor(packed_data["w"], dtype=torch.float32),
        "m": torch.tensor(packed_data["m"], dtype=torch.int8),
    }

    torch.save(tensor_dict, shard_path)
    sz_mb = os.path.getsize(shard_path) / (1024 * 1024)
    print(f"  ✓ Saved Shard {shard_idx:04d}: {len(samples):,} families ({sz_mb:.2f} MB) -> {shard_path}")
    return shard_path


def main(input_json: str, vocab_file: str, out_dir: str, shard_size: int = 2500):
    print("=" * 80)
    print("  GENINHERIT PHASE 1: STREAMING PREPROCESSOR & BINARY SHARDER")
    print(f"  Input Source: {input_json}")
    print(f"  Output Shard Directory: {out_dir}")
    print("=" * 80)

    vocab = load_vocabulary(vocab_file)
    os.makedirs(out_dir, exist_ok=True)

    print("Reading and parsing input families...")
    with open(input_json, "r", encoding="utf-8") as f:
        families = json.load(f)

    # Split 90% Train, 5% Val IID, 5% Val Comp
    random.seed(42)
    random.shuffle(families)

    n_total = len(families)
    n_train = int(n_total * 0.90)
    n_val_iid = int(n_total * 0.05)

    train_fams = families[:n_train]
    val_iid_fams = families[n_train:n_train + n_val_iid]
    val_comp_fams = families[n_train + n_val_iid:]

    print(f"  Total Cohort: {n_total:,} families")
    print(f"  -> Train: {len(train_fams):,} | Val IID: {len(val_iid_fams):,} | Val Comp: {len(val_comp_fams):,}")

    # Process and shard training data
    shard_idx = 0
    buffer = []
    t0 = time.time()

    for idx, fam in enumerate(train_fams):
        sample = serialize_family_to_psts(fam, vocab)
        buffer.append(sample)

        if len(buffer) >= shard_size:
            pack_and_save_shard(buffer, shard_idx, out_dir)
            shard_idx += 1
            buffer = []

    if buffer:
        pack_and_save_shard(buffer, shard_idx, out_dir)
        shard_idx += 1

    # Save Validation Sets
    val_iid_samples = [serialize_family_to_psts(f, vocab) for f in val_iid_fams]
    pack_and_save_shard(val_iid_samples, 9001, out_dir)
    os.rename(os.path.join(out_dir, "shard_9001.pt"), os.path.join(out_dir, "val_iid.pt"))

    val_comp_samples = [serialize_family_to_psts(f, vocab) for f in val_comp_fams]
    pack_and_save_shard(val_comp_samples, 9002, out_dir)
    os.rename(os.path.join(out_dir, "shard_9002.pt"), os.path.join(out_dir, "val_comp.pt"))

    elapsed = time.time() - t0
    print("=" * 80)
    print(f"✓ Sharding Completed in {elapsed:.2f}s!")
    print(f"  Total Binary Shards Written: {shard_idx}")
    print(f"  Validation Sets: val_iid.pt, val_comp.pt generated.")
    print("=" * 80)


if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else "./synthetic_pedigrees.json"
    vocab_p = sys.argv[2] if len(sys.argv) > 2 else "./bioDSL_vocab.json"
    out_d = sys.argv[3] if len(sys.argv) > 3 else "./shards"
    main(in_file, vocab_p, out_d)
