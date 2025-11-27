import os
import random
import argparse
from pathlib import Path


def generate_mrpc_mock(output_dir: str, num_samples: int = 1725):
    """
    Generate mock MRPC submission file.
    
    MRPC is a binary classification task (0 = not_equivalent, 1 = equivalent).
    Test set typically has ~1725 examples.
    
    Args:
        output_dir: Output directory for TSV file
        num_samples: Number of test samples
    """
    filepath = os.path.join(output_dir, "MRPC.tsv")
    
    print(f"Generating MRPC.tsv with {num_samples} samples...")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("index\tprediction\n")
        
        for idx in range(num_samples):
            prediction = 1 if random.random() > 0.33 else 0
            f.write(f"{idx}\t{prediction}\n")
    
    print(f"Created {filepath}")
    return filepath


def generate_stsb_mock(output_dir: str, num_samples: int = 1379):
    """
    Generate mock STS-B submission file.
    
    STS-B is a regression task with similarity scores from 0.0 to 5.0.
    Test set typically has ~1379 examples.
    
    Args:
        output_dir: Output directory for TSV file
        num_samples: Number of test samples
    """
    filepath = os.path.join(output_dir, "STS-B.tsv")
    
    print(f"Generating STS-B.tsv with {num_samples} samples...")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("index\tprediction\n")
        
        for idx in range(num_samples):
            prediction = random.triangular(0.0, 5.0, 2.5)
            f.write(f"{idx}\t{prediction:.3f}\n")
    
    print(f"Created {filepath}")
    return filepath


def generate_ax_mock(output_dir: str, num_samples: int = 1104):
    """
    Generate mock AX (Diagnostic) submission file.
    
    AX is a 3-class NLI task (entailment, neutral, contradiction).
    Test set typically has ~1104 examples.
    
    Args:
        output_dir: Output directory for TSV file
        num_samples: Number of test samples
    """
    filepath = os.path.join(output_dir, "AX.tsv")
    
    print(f"Generating AX.tsv with {num_samples} samples...")
    
    labels = ["entailment", "neutral", "contradiction"]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("index\tprediction\n")
        
        for idx in range(num_samples):
            prediction = random.choice(labels)
            f.write(f"{idx}\t{prediction}\n")
    
    print(f"Created {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Generate mock GLUE submission files for missing datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for TSV files",
    )
    # parser.add_argument(
    #     "--mrpc-samples",
    #     type=int,
    #     default=1725,
    #     help="Number of MRPC test samples (default: 1725)",
    # )
    parser.add_argument(
        "--stsb-samples",
        type=int,
        default=1379,
        help="Number of STS-B test samples (default: 1379)",
    )
    parser.add_argument(
        "--ax-samples",
        type=int,
        default=1104,
        help="Number of AX test samples (default: 1104)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Mock GLUE Submission Files")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    generated_files = []
    
    # generated_files.append(
    #     generate_mrpc_mock(args.output_dir, args.mrpc_samples)
    # )
    generated_files.append(
        generate_stsb_mock(args.output_dir, args.stsb_samples)
    )
    generated_files.append(
        generate_ax_mock(args.output_dir, args.ax_samples)
    )
    
    submission_zip = os.path.join(args.output_dir, "submission.zip")
    if os.path.exists(submission_zip):
        print()
        print("Updating submission.zip...")
        import zipfile
        
        existing_files = {}
        with zipfile.ZipFile(submission_zip, 'r') as zf:
            for name in zf.namelist():
                existing_files[name] = zf.read(name)
        
        with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name, content in existing_files.items():
                zf.writestr(name, content)
            
            for filepath in generated_files:
                filename = os.path.basename(filepath)
                zf.write(filepath, filename)
        
        print(f"Updated {submission_zip}")
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Generated {len(generated_files)} mock submission files:")
    for filepath in generated_files:
        print(f"  - {os.path.basename(filepath)}")
    print()
    print("Note: These are MOCK files with random predictions.")
    print("Replace them with real model predictions before submission.")
    print("=" * 60)


if __name__ == "__main__":
    main()
