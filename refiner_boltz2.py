#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Boltz binder refinement + LigandMPNN sequence optimization

This is a refactored, modularized single-file script that:
- Preserves the original behavior and defaults
- Exposes configuration via command-line arguments
"""

# @title 🧩 Setup and Core Imports
import os
from pathlib import Path
import warnings
import contextlib
import io
import copy
from matplotlib.pylab import f
import yaml
import argparse

import numpy as np

import pandas as pd
import torch
import random
from collections import defaultdict

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
    category=UserWarning,
)

import py2Dmol
from LigandMPNN.wrapper import LigandMPNNWrapper

# --- New/Refactored Imports ---
from boltz_ph.constants import CHAIN_TO_NUMBER
from utils.metrics import get_CA_and_sequence  # Used implicitly in design.py
from utils.convert import calculate_holo_apo_rmsd, convert_cif_files_to_pdb
# -----------------------------

from boltz_ph.model_utils import (
    binder_binds_contacts,
    extract_sequence_from_structure,
    clean_memory,
    design_sequence,
    get_boltz_model,
    get_cif,
    load_canonicals,
    plot_from_pdb,
    # plot_run_metrics,
    process_msa,
    run_prediction,
    sample_seq,
    save_pdb,
    shallow_copy_tensor_dict,
    smart_split,
)

print("✅ Core functionality imported.")


# -------------------------------------------------------------------------
# Helpers: CLI parsing and args container
# -------------------------------------------------------------------------

class Args:
    """Simple args container (for dot-notation)."""
    def __init__(self, **entries):
        self.__dict__.update(entries)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_cli_args():
    """Parse command-line arguments and return a populated Args instance.

    Defaults are chosen to match the original script.
    """
    parser = argparse.ArgumentParser(
        description="Boltz binder refinement + LigandMPNN optimization"
    )

    # --- General Setup ---
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--grad_enabled", type=str2bool, default=False)
    parser.add_argument("--name", type=str, default="l181_s575430_mpnn2_refined_2")
    parser.add_argument("--mode", type=str, default="binder",
                        choices=["unconditional", "binder"])
    parser.add_argument("--num_designs", type=int, default=3)
    parser.add_argument("--num_cycles", type=int, default=5)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./ph_refining_outputs",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=os.getcwd(),
        help="Working directory (default: current directory)",
    )

    # --- New Refiner/Cycling Input ---
    parser.add_argument(
        "--initial_design_path",
        type=str,
        default="/home/tadas/code/adaptyv_nipah_competition/handpicked_BC_designs_his/nipah_binder_l101_s30533_mpnn4_model4.pdb",
        help="Initial CIF or PDB file to start the refinement from",
    )
    # --- New: sequence-only mode ---
    parser.add_argument(
        "--sequence_only",
        type=str2bool,
        default=False,
        help="If true, ignore initial_design_path and use binder_seq/protein_seqs directly."
    )
    parser.add_argument(
        "--binder_seq",
        type=str,
        default=None,
        help="Binder sequence (required if --sequence_only True)"
    )
    parser.add_argument(
        "--target_seq",
        type=str,
        default=None,
        help="Target protein sequence (required if --sequence_only True)"
    )
    parser.add_argument(
        "--binder_dir",
        type=str,
        default=None,
        help="Directory containing FASTA files; run pipeline once per binder."
    )


    parser.add_argument("--binder_chain", type=str, default="B")

    # --- Target Protein(s) ---
    parser.add_argument("--protein_ids", type=str, default="A")
    # Default behavior: derive protein_seqs from structure if not given:
    parser.add_argument(
        "--protein_seqs",
        type=str,
        default=None,
        help="If not provided, extracted from initial_design_path and protein_ids",
    )
    parser.add_argument(
        "--protein_msas",
        type=str,
        default="",
        help='"" means generate, "empty" is single sequence',
    )
    parser.add_argument("--cyclics", type=str, default="")

    # --- Non-Protein Components (Ligand/Nucleic Acid) ---
    parser.add_argument("--ligand_id", type=str, default="")
    parser.add_argument("--ligand_smiles", type=str, default="")
    parser.add_argument("--ligand_ccd", type=str, default="")
    parser.add_argument("--nucleic_type", type=str, default="")
    parser.add_argument("--nucleic_id", type=str, default="")
    parser.add_argument("--nucleic_seq", type=str, default="")

    # --- Templates and Constraints ---
    parser.add_argument("--template_path", type=str, default="")
    parser.add_argument("--template_chain_id", type=str, default="")
    parser.add_argument("--add_constraints", type=str2bool, default=False)
    parser.add_argument(
        "--contact_residues",
        type=str,
        default="",
        help='e.g. "1,2,5,10" on target chain',
    )
    parser.add_argument("--constraint_target_chain", type=str, default="B")
    parser.add_argument("--contact_cutoff", type=float, default=10.0)
    parser.add_argument("--max_contact_filter_retries", type=int, default=6)
    parser.add_argument("--no_contact_filter", type=str2bool, default=False)

    # --- Model & Diffusion Parameters ---
    parser.add_argument("--no_potentials", type=str2bool, default=True)
    parser.add_argument("--diffuse_steps", type=int, default=200)
    parser.add_argument("--recycling_steps", type=int, default=6)
    parser.add_argument(
        "--boltz_model_version",
        type=str,
        default="boltz2",
        choices=["boltz1", "boltz2"],
    )
    parser.add_argument(
        "--boltz_model_path",
        type=str,
        default=os.path.expanduser("~/.boltz/boltz2_conf.ckpt"),
    )
    parser.add_argument(
        "--ccd_path",
        type=str,
        default=str(Path(os.path.expanduser("~/.boltz/mols"))),
    )
    parser.add_argument("--logmd", type=str2bool, default=False)

    # --- Design & Optimization ---
    parser.add_argument("--randomly_kill_helix_feature", type=str2bool, default=False)
    parser.add_argument("--negative_helix_constant", type=float, default=0.0)
    parser.add_argument("--alanine_bias", type=str2bool, default=True)
    parser.add_argument("--temperature_start", type=float, default=0.05)
    parser.add_argument("--temperature_end", type=float, default=0.001)
    parser.add_argument("--alanine_bias_start", type=float, default=-0.5)
    parser.add_argument("--alanine_bias_end", type=float, default=-0.2)
    parser.add_argument("--omit_AA", type=str, default="C")
    parser.add_argument("--exclude_P", type=str2bool, default=False)
    parser.add_argument("--frac_X", type=float, default=0.5)
    parser.add_argument("--high_iptm_threshold", type=float, default=0.8)

    # --- Optional: Validation Parameters (External Dependencies) ---
    parser.add_argument(
        "--alphafold_dir",
        type=str,
        default=os.path.expanduser("~/alphafold3"),
    )
    parser.add_argument("--af3_docker_name", type=str, default="alphafold3_yc")
    parser.add_argument(
        "--af3_database_settings",
        type=str,
        default=os.path.expanduser("~/alphafold3/alphafold3_data_save"),
    )
    parser.add_argument(
        "--hmmer_path",
        type=str,
        default=os.path.expanduser("~/.conda/envs/alphafold3_venv"),
    )
    parser.add_argument("--use_msa_for_af3", type=str2bool, default=False)
    parser.add_argument("--plot", type=str2bool, default=True)
    parser.add_argument("--viewer", type=str2bool, default=True)

    ns = parser.parse_args()

    # For ccd_path, normalize to Path
    ns.ccd_path = Path(os.path.expanduser(str(ns.ccd_path)))

    # Derive protein_seqs from structure if not supplied (match original behavior)
    # Handle sequence-only mode
    # sequence-only mode logic
    if ns.sequence_only:
        # target is always required
        if ns.target_seq is None:
            raise ValueError("--sequence_only requires --target_seq")

        # binder_seq is ONLY required if binder_dir is NOT provided
        if ns.binder_dir is None and ns.binder_seq is None:
            raise ValueError("--sequence_only requires --binder_seq OR --binder_dir")

        ns.protein_ids = "A"
        ns.protein_seqs = ns.target_seq
    else:
        # normal PDB-based extraction
        if ns.protein_seqs is None:
            ns.protein_seqs = extract_sequence_from_structure(
                ns.initial_design_path, ns.protein_ids
            )


    return Args(**vars(ns))

import matplotlib.pyplot as plt
# -------------------------------------------------------------------------
# Core functional blocks
# -------------------------------------------------------------------------
def plot_run_metrics(
    run_save_dir: str, name: str, run_id: int, num_cycles: int, run_metrics: dict
):
    """Plots per-run metrics (iPTM, pLDDT, Alanine Count) over design cycles."""
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    colors = ["#9B59B6", "#E94560", "#FF7F11", "#2ECC71", "#3498DB"]
    
    # Helper to retrieve data and format
    def get_metric_data(key_suffix, label, ymin, ymax, fmt):
        values = [run_metrics.get(f"cycle_{i}_{key_suffix}", float("nan")) for i in range(num_cycles + 1)]
        return (label, values, ymin, ymax, fmt)

    metrics_list = [
        get_metric_data("iptm", "iPTM", 0, 1, "{:.3f}"),
        get_metric_data("plddt", "pLDDT", 0, 1, "{:.1f}"), # Corrected pLDDT max to 100
        get_metric_data("iplddt", "iPLDDT", 0, 1, "{:.1f}"), # Corrected iPLDDT max to 100
        get_metric_data(
            "alanine", 
            "Alanine Count", 
            0, 
            max([run_metrics.get(f"cycle_{i}_alanine", 0) for i in range(num_cycles + 1)]) + 2, 
            "{}",
        ),
        get_metric_data("ipsae_min", "ipSAE_min", 0, 1, "{:.3f}"),
    ]
    
    design_cycles = list(range(num_cycles + 1))
    
    for ax, (label, values, ymin, ymax, fmt), color in zip(axs, metrics_list, colors):
        valid_indices = [i for i, y in enumerate(values) if not pd.isnull(y)]
        valid_cycles = [design_cycles[i] for i in valid_indices]
        valid_values = [values[i] for i in valid_indices]

        ax.plot(
            valid_cycles,
            valid_values,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=2,
        )
        ax.set(
            xlabel="Design Iteration",
            ylabel=label,
            title=f"{label} (Run {run_id})",
            xticks=design_cycles,
            ylim=(ymin, ymax),
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        
        # Annotate points
        for x, y in zip(design_cycles, values):
            if not pd.isnull(y):
                 ax.annotate(
                    fmt.format(y),
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9,
                )
                
    plt.tight_layout()
    plot_filename = f"{name}_run_{run_id}_design_cycle_results.png"
    plt.savefig(f"{run_save_dir}/{plot_filename}", dpi=300)
    plt.show(block=False)

def initialize_models_and_paths(args):
    """Initialize device, models, and main directories."""
    # Device
    device = (
        f"cuda:{args.gpu_id}"
        if torch.cuda.is_available() and args.gpu_id >= 0
        else "cpu"
    )
    print(f"Using device: {device}")

    # Prediction args for Boltz
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.diffuse_steps,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": False,
        "max_parallel_samples": 1,
    }

    # CCD library and Boltz model
    ccd_lib = load_canonicals(os.path.expanduser(str(args.ccd_path)))
    boltz_model = get_boltz_model(
        checkpoint=args.boltz_model_path,
        predict_args=predict_args,
        device=device,
        model_version=args.boltz_model_version,
        no_potentials=args.no_potentials,
        grad_enabled=args.grad_enabled,
    )

    # LigandMPNN wrapper
    designer = LigandMPNNWrapper(os.path.join(args.work_dir, "./LigandMPNN/run.py"))

    # Directories
    protein_hunter_save_dir = os.path.join(args.save_dir, "0_protein_hunter_design")
    os.makedirs(protein_hunter_save_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    return device, ccd_lib, boltz_model, designer, protein_hunter_save_dir


def build_sequences_and_data(args, protein_hunter_save_dir, binder_seq):
    """Build the 'data' dictionary and determine model_type/pocket_conditioning."""
    sequences = []

    # Process multi-chain/MSA inputs
    protein_ids_list = smart_split(args.protein_ids)
    protein_seqs_list = smart_split(args.protein_seqs)
    protein_msas_list = (
        smart_split(args.protein_msas)
        if args.protein_msas
        else [""] * len(protein_ids_list)
    )
    cyclics_list = (
        smart_split(args.cyclics) if args.cyclics else ["False"] * len(protein_ids_list)
    )

    max_len = max(
        len(protein_ids_list),
        len(protein_seqs_list),
        len(protein_msas_list),
        len(cyclics_list),
    )
    for l in [protein_ids_list, protein_seqs_list, protein_msas_list, cyclics_list]:
        while len(l) < max_len:
            l.append("")

    seq_to_indices = defaultdict(list)
    for idx, seq in enumerate(protein_seqs_list):
        if seq:
            seq_to_indices[seq].append(idx)
    seq_to_final_msa = {}

    # Suppress MSA generation output during this phase
    with contextlib.redirect_stdout(io.StringIO()):
        for seq, idx_list in seq_to_indices.items():
            chosen_msa = next(
                (
                    protein_msas_list[i]
                    for i in idx_list
                    if protein_msas_list[i] and protein_msas_list[i] != "empty"
                ),
                None,
            )
            if chosen_msa is None:
                chosen_msa = ""

            if chosen_msa == "":
                pid = (
                    protein_ids_list[idx_list[0]]
                    if protein_ids_list[idx_list[0]]
                    else f"CHAIN_{idx_list[0]}"
                )
                msa_value = process_msa(pid, seq, Path(protein_hunter_save_dir))
                seq_to_final_msa[seq] = str(msa_value)
            elif chosen_msa == "empty":
                seq_to_final_msa[seq] = "empty"
            else:
                seq_to_final_msa[seq] = chosen_msa

    # Build sequences list and add X-binder
    for pid, seq, cyc in zip(protein_ids_list, protein_seqs_list, cyclics_list):
        if not pid or not seq:
            continue
        final_msa = seq_to_final_msa.get(seq, "empty")
        cyc_val = cyc.lower() in ["true", "1", "yes"]
        sequences.append(
            {
                "protein": {
                    "id": [pid],
                    "sequence": seq,
                    "msa": final_msa,
                    "cyclic": cyc_val,
                }
            }
        )

    # Binder
    sequences.append(
        {
            "protein": {
                "id": [args.binder_chain],
                "sequence": binder_seq,
                "msa": "empty",
                "cyclic": False,
            }
        }
    )

    # Ligand / nucleic
    if args.ligand_smiles:
        sequences.append(
            {"ligand": {"id": [args.ligand_id], "smiles": args.ligand_smiles}}
        )
    elif args.ligand_ccd:
        sequences.append(
            {"ligand": {"id": [args.ligand_id], "ccd": args.ligand_ccd}}
        )
    if args.nucleic_seq:
        sequences.append(
            {
                args.nucleic_type: {
                    "id": [args.nucleic_id],
                    "sequence": args.nucleic_seq,
                }
            }
        )

    # Templates and constraints
    templates = []
    if args.template_path:
        template_path_list = smart_split(args.template_path)
        template_chain_id_list = (
            smart_split(args.template_chain_id) if args.template_chain_id else []
        )
        template_files = [get_cif(tp) for tp in template_path_list]
        for i, template_file in enumerate(template_files):
            t_block = (
                {"cif": template_file}
                if template_file.endswith(".cif")
                else {"pdb": template_file}
            )
            if template_chain_id_list and i < len(template_chain_id_list):
                t_block["chain_id"] = template_chain_id_list[i]
            templates.append(t_block)

    data = {"sequences": sequences}
    if templates:
        data["templates"] = templates

    pocket_conditioning = args.add_constraints

    if args.add_constraints:
        residues = args.contact_residues.split(",")
        contacts = [
            [args.constraint_target_chain, int(res)]
            for res in residues
            if res.strip() != ""
        ]
        constraints = [{"pocket": {"binder": args.binder_chain, "contacts": contacts}}]
        data["constraints"] = constraints

    data["sequences"] = sorted(
        data["sequences"], key=lambda entry: list(entry.values())[0]["id"][0]
    )

    any_ligand_or_nucleic = args.ligand_smiles or args.ligand_ccd or args.nucleic_seq
    model_type = "ligand_mpnn" if any_ligand_or_nucleic else "soluble_mpnn"

    print("✅ Models ready and base data configured.")
    print("Mode:", args.mode)
    print("Data dictionary (base):\n", data)

    return data, model_type, pocket_conditioning


# -------------------------------------------------------------------------
# Design loop utilities
# -------------------------------------------------------------------------
def compute_ipae_from_pae(pae, binder_len, target_len):
    """
    Compute inter-chain ipAE (binder–target) from global PAE matrix,
    respecting chain alphabetical ordering (PH sorting).
    """
    pae = pae.detach().cpu().numpy()[0]  # (L, L)
    L_A = target_len
    L_B = binder_len

    # Inter-chain blocks:
    block_AB = pae[0:L_A, L_A:L_A + L_B]     # target → binder
    block_BA = pae[L_A:L_A + L_B, 0:L_A]     # binder → target

    inter = np.concatenate([block_AB.flatten(), block_BA.flatten()])
    return float(inter.mean()) if inter.size > 0 else 0.0


def _d0(L):
    """Yang–Skolnick d0 function (protein–protein)."""
    L = float(max(L, 27))
    return max(1.0, 1.24 * (L - 15)**(1/3) - 1.8)


def calculate_ipsae_min(structure, coords, pae, target_len, binder_len,
                        pae_cutoff=15.0, dist_cutoff=15.0):
    """
    Compute ipSAE_min(A,B) between two chains:
      A = target residues [0:target_len]
      B = binder residues [target_len:target_len+binder_len]

    Uses CA coordinates extracted from structure + coords.
    """

    # 1) CA coordinates (L_res, 3)
    ca = extract_ca_coords(structure, coords)

    L = target_len + binder_len
    assert ca.shape[0] == L, f"Residue count mismatch: expected {L}, got {ca.shape[0]}"

    # 2) PAE matrix (L, L)
    pae = pae.detach().cpu().numpy()[0]

    # 3) Distance matrix
    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    # 4) Chain index arrays
    A = np.arange(0, target_len)
    B = np.arange(target_len, target_len + binder_len)

    # 5) d0 for PTM transform
    Lclamp = max(L, 27)
    d0 = max(1.0, 1.24 * (Lclamp - 15)**(1/3) - 1.8)

    def _asym(source, target):
        best = 0.0
        for i in source:
            mask = (pae[i, target] < pae_cutoff) & (dist[i, target] < dist_cutoff)
            if not np.any(mask):
                continue
            vals = pae[i, target][mask]
            ptm = 1.0 / (1.0 + (vals / d0)**2)
            best = max(best, float(ptm.mean()))
        return best

    # 6) bidirectional asymmetric scores
    a_to_b = _asym(A, B)
    b_to_a = _asym(B, A)

    return float(min(a_to_b, b_to_a))





def compute_iptm(pair_chains, ref_chain_idx):
    if len(pair_chains) > 1:
        vals = [
            (
                pair_chains[ref_chain_idx][i].detach().cpu().numpy()
                + pair_chains[i][ref_chain_idx].detach().cpu().numpy()
            )
            / 2.0
            for i in range(len(pair_chains))
            if i != ref_chain_idx
        ]
        return float(np.mean(vals) if vals else 0.0)
    else:
        return 0.0


def get_float(item, key, default_val):
    return float(
        item.get(key, torch.tensor([default_val])).detach().cpu().numpy()[0]
    )


def run_design_and_optimization(
    args, data, model_type, pocket_conditioning, boltz_model, ccd_lib, device, designer
):
    """Core multi-design, multi-cycle loop. Returns all_run_metrics."""
    all_run_metrics = []

    if args.sequence_only:
        binder_seq = args.binder_seq
    else:
        binder_seq = (
            args.binder_seq if args.sequence_only
            else extract_sequence_from_structure(args.initial_design_path, args.binder_chain)
        )

    binder_length = len(binder_seq)
    binder_chain_idx = CHAIN_TO_NUMBER[args.binder_chain]

    for design_id in range(args.num_designs):
        viewer = args.viewer  # may become a py2Dmol object later

        if viewer:
            viewer = py2Dmol.view((600, 400), color="plddt")
            viewer.show()

        run_id = str(design_id)
        run_save_dir = os.path.join(args.this_binder_run_dir, f"run_{run_id}")
        os.makedirs(run_save_dir, exist_ok=True)
        data_cp = copy.deepcopy(data)
        print("\n" + "=" * 54)
        print(f"=== Starting Design Run {run_id}/{args.num_designs - 1} ===")
        print("=" * 54)

        best_iptm, best_seq, best_structure, best_output, best_pdb_filename = (
            float("-inf"),
            None,
            None,
            None,
            None,
        )
        best_ipsae_min = -1.0
        best_ipae = -1.0
        best_cycle_idx, best_alanine_percentage = -1, None
        run_metrics = {"run_id": run_id}
        print(f"Binder initial sequence length: {binder_length}")

        # --- Cycle 0 prediction ---
        with contextlib.redirect_stdout(io.StringIO()):
            output, structure = run_prediction(
                data_cp,
                args.binder_chain,
                randomly_kill_helix_feature=args.randomly_kill_helix_feature,
                negative_helix_constant=args.negative_helix_constant,
                boltz_model=boltz_model,
                ccd_lib=ccd_lib,
                ccd_path=args.ccd_path,
                logmd=args.logmd,
                device=device,
                boltz_model_version=args.boltz_model_version,
                pocket_conditioning=pocket_conditioning,
            )
        pdb_filename = (
            f"{run_save_dir}/{args.this_binder_name}_run_{run_id}_predicted_cycle_0.pdb"
        )
        save_pdb(
            structure,
            output["coords"],
            output["plddt"].detach().cpu().numpy()[0],
            pdb_filename,
        )

        cycle_0_iptm = compute_iptm(output["pair_chains_iptm"], binder_chain_idx)
        cycle_0_ipae = compute_ipae_from_pae(output["pae"], binder_length, len(args.protein_seqs))


        cycle_0_ipsae_min = calculate_ipsae_min(
            structure,
            output["coords"],
            output["pae"],
            target_len=len(args.protein_seqs),
            binder_len=binder_length,
        )
        print(f"cycle_0 ipAE: {cycle_0_ipae:.2f}")
        print(f"cycle_0 ipSAE_min: {cycle_0_ipsae_min:.2f}")


        run_metrics.update(
            {
                "cycle_0_iptm": cycle_0_iptm,
                "cycle_0_plddt": get_float(output, "complex_plddt", 0.0),
                "cycle_0_iplddt": get_float(output, "complex_iplddt", 0.0),
                "cycle_0_alanine": binder_seq.count("A") if binder_length else 0.0,
                "cycle_0_seq": binder_seq,
                "cycle_0_ipae": cycle_0_ipae,
                "cycle_0_ipsae_min": cycle_0_ipsae_min,
            }
        )

        # --- Cycles 1...N with sequence/structure optimization
        for cycle in range(args.num_cycles):
            print(f"\n--- Run {run_id}, Cycle {cycle + 1} ---")
            cycle_norm = (
                (cycle / (args.num_cycles - 1)) if args.num_cycles > 1 else 0.0
            )
            alpha = args.alanine_bias_start - cycle_norm * (
                args.alanine_bias_start - args.alanine_bias_end
            )
            temperature = args.temperature_start - cycle_norm * (
                args.temperature_start - args.temperature_end
            )
            design_kwargs = {
                "pdb_file": pdb_filename,
                "temperature": temperature,
                "chains_to_design": args.binder_chain,
                "omit_AA": f"{args.omit_AA},P"
                if cycle == 0
                else args.omit_AA,
            }
            if args.alanine_bias:
                design_kwargs["bias_AA"] = f"A:{alpha}"

            seq_str, logits = design_sequence(designer, model_type, **design_kwargs)
            seq = seq_str.split(":")[binder_chain_idx]
            alanine_count = seq.count("A")
            alanine_percentage = (
                alanine_count / binder_length if binder_length else 0.0
            )

            for seq_entry in data_cp["sequences"]:
                if (
                    "protein" in seq_entry
                    and args.binder_chain in seq_entry["protein"]["id"]
                ):
                    seq_entry["protein"]["sequence"] = seq
                    break

            with contextlib.redirect_stdout(io.StringIO()):
                output, structure = run_prediction(
                    data_cp,
                    args.binder_chain,
                    seq=seq,
                    randomly_kill_helix_feature=False,
                    negative_helix_constant=0.0,
                    boltz_model=boltz_model,
                    ccd_lib=ccd_lib,
                    ccd_path=args.ccd_path,
                    logmd=False,
                    device=device,
                )
            current_iptm = compute_iptm(
                output["pair_chains_iptm"], binder_chain_idx
            )
            curr_ipsae_min = calculate_ipsae_min(
                structure,
                output["coords"],
                output["pae"],
                target_len=len(args.protein_seqs),
                binder_len=binder_length,
            )
            curr_ipae = compute_ipae_from_pae(output["pae"], binder_length, len(args.protein_seqs))
            if alanine_percentage <= 0.20 and curr_ipsae_min > run_metrics.get("best_ipsae_min", -1):
                # update best ipSAE_min
                run_metrics["best_ipsae_min"] = curr_ipsae_min
                run_metrics["best_ipae"] = curr_ipae

                best_structure = copy.deepcopy(structure)
                best_output = shallow_copy_tensor_dict(output)
                best_pdb_filename = (
                    f"{run_save_dir}/{args.this_binder_name}_run_{run_id}_best_structure.pdb"
                )

                save_pdb(
                    best_structure,
                    best_output["coords"],
                    best_output["plddt"].detach().cpu().numpy()[0],
                    best_pdb_filename,
                )

                best_cycle_idx = cycle + 1
                best_seq = seq
                best_alanine_percentage = alanine_percentage

            curr_plddt = get_float(output, "complex_plddt", 0.0)
            curr_iplddt = get_float(output, "complex_iplddt", 0.0)
            run_metrics.update(
                {
                    f"cycle_{cycle + 1}_iptm": current_iptm,
                    f"cycle_{cycle + 1}_plddt": curr_plddt,
                    f"cycle_{cycle + 1}_iplddt": curr_iplddt,
                    f"cycle_{cycle + 1}_alanine": alanine_count,
                    f"cycle_{cycle + 1}_seq": seq,
                }
            )
            
            run_metrics[f"cycle_{cycle + 1}_ipae"] = curr_ipae

            run_metrics[f"cycle_{cycle + 1}_ipsae_min"] = curr_ipsae_min

            print(
                f"ipTM: {current_iptm:.2f}, "
                f"ipAE: {curr_ipae:.2f}, "
                f"ipSAE_min: {curr_ipsae_min:.2f}, "
                f"pLDDT: {curr_plddt:.2f}, "
                f"iPLDDT: {curr_iplddt:.2f}, "
                f"Ala%: {alanine_percentage * 100:.1f}"
            )

            pdb_filename = (
                f"{run_save_dir}/{args.this_binder_name}_run_{run_id}_predicted_cycle_{cycle + 1}.pdb"
            )
            save_pdb(
                structure,
                output["coords"],
                output["plddt"].detach().cpu().numpy()[0],
                pdb_filename,
            )
            if viewer:
                viewer.add_pdb(pdb_filename)
            del output
            del structure
            del logits
            clean_memory()

        # --- Finalize & plot metrics
        run_metrics.update(
            {
                "best_ipsae_min": run_metrics.get("best_ipsae_min", np.nan),
                "best_ipae": run_metrics.get("best_ipae", np.nan),
                "best_iptm": float(best_iptm if best_iptm != float("-inf") else np.nan),
                "best_cycle": best_cycle_idx,
                "best_seq": best_seq,
                "best_plddt": float(
                    best_output.get("complex_plddt", torch.tensor([np.nan]))
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
                if best_output
                else np.nan,
            }
        )
        all_run_metrics.append(run_metrics)
        if args.plot:
            plot_run_metrics(
                run_save_dir, args.this_binder_name, run_id, args.num_cycles, run_metrics
            )

    return all_run_metrics


def save_all_metrics(args, all_run_metrics):
    """Save all metrics to CSV (similar to original logic)."""
    summary_csv = os.path.join(args.save_dir, "summary_all_runs.csv")
    df = pd.DataFrame(all_run_metrics)

    # Reproduce the column logic as close as possible
    columns = ["run_id"] + [
        f"{metric}{cycle_suffix}"
        for cycle_suffix in [f"_{i}" for i in range(args.num_cycles + 1)]
        for metric in [
            "cycle_iptm",
            "cycle_plddt",
            "cycle_iplddt",
            "cycle_alanine",
            "cycle_seq",
        ]
    ]
    columns = [
        col.replace("cycle_", f"cycle_{i}_")
        if "cycle_" in col
        else col
        for i in range(args.num_cycles + 1)
        for col in columns
        if f"_{i}_" in col or col == "run_id"
    ]
    columns = sorted(set(columns), key=columns.index)  # keep order, remove dupes
    columns.extend(["best_iptm", "best_cycle", "best_plddt", "best_seq", "best_ipsae_min"])

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[[c for c in columns if c in df.columns]]
    df.to_csv(summary_csv, index=False)
    print(f"\n✅ All run/cycle metrics saved to {summary_csv}")

def load_fasta_sequences_from_dir(binder_dir):
    """Return list of (binder_name, sequence) for all .fa/.fasta files."""
    binders = []
    for fname in os.listdir(binder_dir):
        if not fname.lower().endswith((".fa", ".fasta", ".faa")):
            continue
        path = os.path.join(binder_dir, fname)
        seq = ""
        with open(path) as f:
            for line in f:
                if line.startswith(">"):
                    continue
                seq += line.strip()
        if seq:
            binder_name = os.path.splitext(fname)[0]
            binders.append((binder_name, seq))
    return binders

def extract_ca_coords(structure, coords):
    """
    Extract C-alpha coordinates (L_res, 3) using:
      - atom naming from structure.atoms
      - atom indexing from structure.residues
      - coordinates from output["coords"] (atom-major)
    """

    atom_coords = coords[0].detach().cpu().numpy()  # shape (N_atoms, 3)

    ca_list = []

    for chain in structure.chains:
        res_start = chain["res_idx"]
        res_end   = res_start + chain["res_num"]

        for res in structure.residues[res_start:res_end]:
            a0 = res["atom_idx"]
            a1 = a0 + res["atom_num"]
            residue_atoms = structure.atoms[a0:a1]

            # Find CA atom inside this residue
            found = False
            for local_i, atom in enumerate(residue_atoms):
                if atom["name"] == "CA":
                    global_i = a0 + local_i
                    ca_list.append(atom_coords[global_i])
                    found = True
                    break

            # fallback (rare): mean of atoms
            if not found:
                meanpos = atom_coords[a0:a1].mean(axis=0)
                ca_list.append(meanpos)

    return np.array(ca_list)  # shape (L_res, 3)




# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    # 1) Parse CLI args
    args = parse_cli_args()

    # 2) Initialize models, device, directories
    device, ccd_lib, boltz_model, designer, protein_hunter_save_dir = (
        initialize_models_and_paths(args)
    )

    # -------------- NEW FEATURE: binder_dir --------------
    if args.binder_dir is not None:
        binder_list = load_fasta_sequences_from_dir(args.binder_dir)
        if not binder_list:
            raise ValueError(f"No FASTA files found in binder_dir: {args.binder_dir}")
    else:
        # Normal case: single binder
        if args.sequence_only:
            binder_list = [("binder", args.binder_seq)]
        else:
            seq = extract_sequence_from_structure(args.initial_design_path,
                                                  args.binder_chain)
            binder_list = [("binder", seq)]
    # ------------------------------------------------------

    all_results = []

    # ----- LOOP OVER BINDERS -----
    for binder_name, binder_seq in binder_list:
        # update args for this binder
        args.binder_seq = binder_seq
        args.this_binder_name = binder_name
        args.this_binder_run_dir= os.path.join(args.save_dir, binder_name)


        expected_subdirs = [f"run_{i}" for i in range(args.num_designs)]
        if all(
            os.path.exists(os.path.join(args.this_binder_run_dir, sd))
            for sd in expected_subdirs
        ):
        # if os.path.exists(args.this_binder_run_dir):
        #     print(
        #         f"⚠ {args.this_binder_run_dir} already exists. Skipping..."
        #     )
            continue
        else:
            print(f"✅ Starting pipeline for binder {binder_name} Length {len(args.binder_seq)}. ")
        print(f"\n=============================")
        print(f"🔁 Running pipeline for binder: {binder_name}")
        print(f"=============================\n")

        os.makedirs(args.this_binder_run_dir, exist_ok=True)

        # Build data dictionary
        data, model_type, pocket_conditioning = build_sequences_and_data(
            args, protein_hunter_save_dir, binder_seq
        )

        # Run design and optimization
        metrics = run_design_and_optimization(
            args,
            data,
            model_type,
            pocket_conditioning,
            boltz_model,
            ccd_lib,
            device,
            designer,
        )

        # Store results with binder_name label
        for m in metrics:
            m["binder_name"] = binder_name
        all_results.extend(metrics)
        clean_memory()
    
    # Save combined results for ALL binders
    save_all_metrics(args, all_results)



if __name__ == "__main__":
    main()
