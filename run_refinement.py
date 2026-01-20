#!/usr/bin/env python3
import os
import argparse
import subprocess

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


def run_single_design(args, binder_name, binder_seq):
    """Run the subprocess call for one binder."""
    save_dir = os.path.join(args.save_dir, binder_name)
    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        "python", args.boltz_ph_script_path,
        "--name", binder_name,
        "--seq", binder_seq,
        "--protein_seqs", args.protein_seqs,
        "--save_dir", save_dir,
        "--num_designs", str(args.num_designs),
        "--num_cycles", str(args.num_cycles),
        "--msa_mode", args.msa_mode,
        "--gpu_id", args.gpu_id,
        "--high_iptm_threshold", str(args.high_iptm_threshold),
        "--high_ipsae_min_threshold", str(args.high_ipsae_min_threshold),
        "--high_plddt_threshold", str(args.high_plddt_threshold),
        "--percent_X", str(args.percent_X),
        "--recycling_steps", str(args.recycling_steps),
        "--diffuse_steps", str(args.diffuse_steps),
        "--min_protein_length", str(args.min_protein_length),
        "--max_protein_length", str(args.max_protein_length),
    ]

    if args.use_msa_for_af3:
        cmd.append("--use_msa_for_af3")
    if args.plot:
        cmd.append("--plot")
    if args.use_exploratory:
        cmd.append("--use_exploratory")
    if args.exclude_P:
        cmd.append("--exclude_P")
    print(f"\nRunning design for binder: {binder_name}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recycling_steps", type=int, default=0)
    parser.add_argument("--diffuse_steps", type=int, default=0)
    parser.add_argument("--boltz_ph_script_path",default=os.path.abspath(os.path.join(os.path.dirname(__file__), "boltz_ph", "design.py")))
    parser.add_argument("--protein_seqs", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--num_designs", type=int, default=2)
    parser.add_argument("--num_cycles", type=int, default=7)
    parser.add_argument("--msa_mode", default="mmseqs")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--high_iptm_threshold", type=float, default=0.8)
    parser.add_argument("--high_ipsae_min_threshold", type=float, default=0.6)
    parser.add_argument("--high_plddt_threshold", type=float, default=0.8)
    parser.add_argument("--percent_X", type=int, default=10)
    parser.add_argument("--use_msa_for_af3", action="store_true")
    parser.add_argument("--use_genetic_algorithm", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--exclude_P", action="store_true")
    parser.add_argument("--use_exploratory", action="store_true")
    parser.add_argument("--min_protein_length", type=int, default=140)
    parser.add_argument("--max_protein_length", type=int, default=150)

    # Single binder mode
    parser.add_argument("--name")
    parser.add_argument("--seq")

    # Multi-fasta mode
    parser.add_argument("--binder_dir", help="Directory containing .fasta files to process")

    args = parser.parse_args()

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
            run_single_design(args, binder_name, binder_seq)

    else:
        # Single run mode
        if not args.name:
            raise ValueError(f"If --binder_dir is not provided, you must supply --name at least")

        run_single_design(args, args.name, args.seq)


if __name__ == "__main__":
    main()
