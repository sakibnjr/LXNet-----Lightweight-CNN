import sys
from pathlib import Path

def count_files_minimal(root_dir):
    root_path = Path(root_dir)
    
    if not root_path.exists() or not root_path.is_dir():
        print(f"Error: '{root_dir}' is not a valid directory.")
        return

    # Header
    print(f"{'Folder Name':<40} | {'Total Files':<10}")
    print("-" * 55)

    # 1. Total files in the root directory (recursive)
    total_files = len([f for f in root_path.rglob('*') if f.is_file()])
    print(f"{root_path.name + ' (Total)':<40} | {total_files:<10}")

    # 2. Iterate through immediate subdirectories
    subfolders = sorted([p for p in root_path.iterdir() if p.is_dir()])
    
    for subfolder in subfolders:
        # Recursive count for this specific subfolder
        count = len([f for f in subfolder.rglob('*') if f.is_file()])
        print(f"{'  └─ ' + subfolder.name:<40} | {count:<10}")

if __name__ == "__main__":
    # Default to "./CLAHE_AUGMENTED_BALANCED" if no argument provided
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./RESIZED_AUGMENTED_BALANCED"
    count_files_minimal(target_dir)