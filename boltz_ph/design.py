import argparse
from ast import Store
import os

from pipeline import ProteinHunter_Boltz

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_single_fasta(fpath):
    """Read a FASTA file that contains exactly one sequence.
    Returns (name, seq)."""
    name = None
    seq_lines = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    raise ValueError(f"FASTA {fpath} contains multiple headers!")
                name = line[1:].strip()
            else:
                seq_lines.append(line)

    if name is None:
        raise ValueError(f"FASTA {fpath} contains no header.")
    if not seq_lines:
        raise ValueError(f"FASTA {fpath} contains no sequence lines.")

    seq = "".join(seq_lines)
    return name, seq

# Keep parse_args() here for CLI functionality
def parse_args():
    parser = argparse.ArgumentParser(
        description="Boltz protein design with cycle optimization"
    )
    # --- Existing Arguments (omitted for brevity, keep all original args) ---
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--grad_enabled", action="store_true", default=False)
    parser.add_argument("--name", default="target_name_is_missing", type=str)
    parser.add_argument(
        "--mode", default="binder", choices=["binder", "unconditional"], type=str
    )
    parser.add_argument("--num_designs", default=50, type=int)
    parser.add_argument("--num_cycles", default=5, type=int)
    parser.add_argument("--cyclic", action="store_true", default=False, help="Enable cyclic peptide design.")
    parser.add_argument("--min_protein_length", default=100, type=int)
    parser.add_argument("--max_protein_length", default=150, type=int)
    parser.add_argument("--seq", default="", type=str)
    parser.add_argument("--refiner_mode", action="store_true", default=False)
    parser.add_argument(
        "--protein_seqs",
        default="",
        type=str,
    )
    parser.add_argument("--msa_mode", default="mmseqs", choices=["single", "mmseqs"], type=str)

    parser.add_argument("--ligand_smiles", default="", type=str)
    parser.add_argument("--ligand_ccd", default="", type=str)
    parser.add_argument(
        "--nucleic_type", default="dna", choices=["dna", "rna"], type=str
    )
    parser.add_argument("--nucleic_seq", default="", type=str)
    parser.add_argument(
        "--template_path", default="", type=str
    )  #pdb code or path(s) to .cif/.pdb, multiple allowed separated by colon or comma
    parser.add_argument(
        "--template_cif_chain_id", default="", type=str
    )  # for mmCIF files, the chain id to use for the template (for alignment)
    parser.add_argument("--diffuse_steps", default=200, type=int)
    parser.add_argument("--recycling_steps", default=3, type=int)
    parser.add_argument("--boltz_model_version", default="boltz2", type=str)
    parser.add_argument(
        "--boltz_model_path",
        default="~/.boltz/boltz2_conf.ckpt",
        type=str,
    )
    parser.add_argument("--ccd_path", default="~/.boltz/mols", type=str)
    parser.add_argument("--randomly_kill_helix_feature", action="store_true", default=False)
    parser.add_argument("--negative_helix_constant", default=0.2, type=float)
    parser.add_argument("--logmd", action="store_true", default=False)
    parser.add_argument("--save_dir", default="", type=str)
    parser.add_argument("--omit_AA", default="C", type=str)
    parser.add_argument("--exclude_P", action="store_true", default=False)
    parser.add_argument("--percent_X", default=90, type=int)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot cycles figs per run (requires matplotlib)",
    )

    parser.add_argument(
        "--contact_residues", 
        default="", 
        type=str,
        help="Specify which target chain residues must contact the binder (currently only supports protein contacts). For more than two chains, separate by |, e.g., '1,2,5,10 | 3,5,10'."
    )

    parser.add_argument(
        "--no_contact_filter",
        action="store_true",
        help="Do not filter or restart for unbound contact residues at cycle 0",
    )
    parser.add_argument("--max_contact_filter_retries", default=6, type=int)
    parser.add_argument("--contact_cutoff", default=15.0, type=float)

    parser.add_argument(
        "--alphafold_dir", default=os.path.expanduser("~/alphafold3"), type=str
    )
    parser.add_argument("--af3_docker_name", default="alphafold3_yc", type=str)
    parser.add_argument(
        "--af3_database_settings", default="~/alphafold3/alphafold3_data_save", type=str
    )
    parser.add_argument(
        "--hmmer_path",
        default="~/.conda/envs/alphafold3_venv",
        type=str,
    )
    parser.add_argument("--use_alphafold3_validation", action="store_true", default=False)
    parser.add_argument("--use_msa_for_af3", action="store_true")
    parser.add_argument("--work_dir", default="", type=str)

    # temp and bias params
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--alanine_bias_start", default=-0.5, type=float)
    parser.add_argument("--alanine_bias_end", default=-0.1, type=float)
    parser.add_argument("--alanine_bias", action="store_true")
    parser.add_argument("--high_iptm_threshold", default=0.8, type=float)
    parser.add_argument("--high_plddt_threshold", default=0.8, type=float)
    parser.add_argument("--high_ipsae_min_threshold", default=0.6, type=float)
    parser.add_argument("--use_exploratory", action="store_true")
    # --- End Existing Arguments ---

    # Multi-fasta mode
    parser.add_argument("--binder_dir", help="Directory containing .fasta files to process")

    return parser.parse_args()

def print_args(args):
    print("="*40)
    print("Design Configuration:")
    for k, v in vars(args).items():
        print(f"{k:30}: {v}")
    print("="*40)

def main():
    args = parse_args()
    # Pretty print each argument in a row for better visualization
    print_args(args)

    # Create output directory if needed
    os.makedirs(args.save_dir, exist_ok=True)

    if args.binder_dir:
        # Process each .fasta file
        fasta_files = [f for f in os.listdir(args.binder_dir) if f.endswith(".fasta")]
        if not fasta_files:
            raise ValueError(f"No .fasta files found in binder_dir: {args.binder_dir}")

        for fasta in fasta_files:
            fpath = os.path.join(args.binder_dir, fasta)
            binder_name, binder_seq = read_single_fasta(fpath)
            # Modify args for this binder
            args.name = binder_name
            args.seq = binder_seq
            args.save_dir = os.path.join(args.save_dir, binder_name)
            os.makedirs(args.save_dir, exist_ok=True)
            print(f"\nRunning design for binder: {binder_name}")
            protein_hunter = ProteinHunter_Boltz(args)
            protein_hunter.run_pipeline()
    else:
        # Single run mode
        if not args.name:
            raise ValueError("If --binder_dir is not provided, you must supply --name")
        protein_hunter = ProteinHunter_Boltz(args)
        protein_hunter.run_pipeline()

if __name__ == "__main__":
    main()
