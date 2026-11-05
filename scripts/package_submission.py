"""Package the project into the required submission format:

CS6106_TeamXX/
├── report/
│   └── CS6106_TeamXX_Report.pdf  (placeholder if not provided)
├── slides/
│   └── CS6106_TeamXX_Slides.pdf  (final version as presented)
└── code/
    ├── README.md                 (setup steps, how to run, GitHub link)
    ├── requirements.txt
    └── ...                       (clean source code)

Usage:
    python scripts/package_submission.py [TEAM_NUMBER]
    e.g. python scripts/package_submission.py 01
    (defaults to 'XX' if omitted)
"""
import argparse
import os
import shutil
import zipfile

IGNORED_EXTENSIONS = {".pyc", ".npz", ".zip", ".dat", ".log", ".aux", ".bcf", ".nav", ".out", ".snm", ".toc", ".blg"}
IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".git", "results/cache", "cache", "pmlb.egg-info"}


def is_ignored(root, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in IGNORED_EXTENSIONS:
        return True
    return False


def copy_tree_clean(src, dst):
    """Copy directory tree excluding caches, large dataset files, and temporary artifacts."""
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        # Filter out ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and "cache" not in d]

        rel_dir = os.path.relpath(root, src)
        target_dir = os.path.join(dst, rel_dir) if rel_dir != "." else dst
        os.makedirs(target_dir, exist_ok=True)

        for f in files:
            if is_ignored(root, f):
                continue
            # Also exclude raw MovieLens rating csv files from the submission code bundle
            if "data" in root and f.endswith(".csv") and ("ratings" in f or "movies" in f or "users" in f):
                continue
            src_file = os.path.join(root, f)
            dst_file = os.path.join(target_dir, f)
            shutil.copy2(src_file, dst_file)


def package(team_id="XX"):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sub_name = f"CS6106_Team{team_id}"
    out_dir = os.path.join(base_dir, sub_name)

    print(f"Packaging submission for {sub_name}...")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. report/
    report_dir = os.path.join(out_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"{sub_name}_Report.pdf")
    placeholder_note = os.path.join(report_dir, "README_REPORT.txt")
    with open(placeholder_note, "w", encoding="utf-8") as f:
        f.write(f"Please place your final report PDF here named '{sub_name}_Report.pdf' before final submission.\n")

    # If a report already exists in the workspace, copy it
    existing_reports = [
        os.path.join(base_dir, "report", f"{sub_name}_Report.pdf"),
        os.path.join(base_dir, "report", "CS6106_TeamXX_Report.pdf"),
        os.path.join(base_dir, "report.pdf"),
        os.path.join(base_dir, f"{sub_name}_Report.pdf"),
    ]
    for rep in existing_reports:
        if os.path.exists(rep):
            shutil.copy2(rep, report_file)
            print(f"--> Included existing report: {rep}")
            os.remove(placeholder_note)
            break

    # 2. slides/
    slides_dir = os.path.join(out_dir, "slides")
    os.makedirs(slides_dir, exist_ok=True)
    slides_dst = os.path.join(slides_dir, f"{sub_name}_Slides.pdf")
    slides_src = os.path.join(base_dir, "presentation", "main.pdf")
    if os.path.exists(slides_src):
        shutil.copy2(slides_src, slides_dst)
        print(f"--> Copied slides from presentation/main.pdf to slides/{sub_name}_Slides.pdf")
    else:
        print("Warning: presentation/main.pdf not found!")

    # 3. code/
    code_dir = os.path.join(out_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    # Copy root files
    root_files = ["README.md", "requirements.txt", "pyproject.toml", ".gitignore"]
    for rf in root_files:
        src_path = os.path.join(base_dir, rf)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(code_dir, rf))

    # Copy directories
    subdirs = ["src", "experiments", "scripts", "tests", "results", "plots"]
    for sd in subdirs:
        src_path = os.path.join(base_dir, sd)
        if os.path.exists(src_path):
            copy_tree_clean(src_path, os.path.join(code_dir, sd))

    # Create empty data directory with README / download instructions
    data_dir = os.path.join(code_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Datasets are excluded from this submission package per course guidelines.\n")
        f.write("To download MovieLens-1M, run: python scripts/download_movielens.py\n")

    # 4. Create zip archive
    zip_path = os.path.join(base_dir, f"{sub_name}.zip")
    print(f"Creating zip archive: {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, base_dir)
                zf.write(abs_path, rel_path)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"--> Successfully created {sub_name}.zip ({zip_size_mb:.2f} MB)")
    print(f"Submission folder ready at: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("team_id", nargs="?", default="XX", help="Team number, e.g. 01, 15 (default: XX)")
    args = parser.parse_args()
    package(args.team_id)
