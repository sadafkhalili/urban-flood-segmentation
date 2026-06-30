from pathlib import Path

from src.config import RAW_DATA_DIR


DATASET_DIR = RAW_DATA_DIR / "urban_sar_floods"

CLASSES = {
    "01_NF": "Non-Flood",
    "02_FO": "Flood Open Area",
    "03_FU": "Flood Urban",
}


def main():
    if not DATASET_DIR.exists():
        print("Dataset folder not found.")
        print(f"Expected path: {DATASET_DIR}")
        return

    print("Dataset found.")
    print(f"Path: {DATASET_DIR}\n")

    total_sar = 0
    total_gt = 0

    for folder, class_name in CLASSES.items():
        sar_dir = DATASET_DIR / folder / "SAR"
        gt_dir = DATASET_DIR / folder / "GT"

        sar_files = list(sar_dir.glob("*.tif"))
        gt_files = list(gt_dir.glob("*.tif"))

        print(f"{folder} - {class_name}")
        print(f"  SAR images: {len(sar_files)}")
        print(f"  GT masks  : {len(gt_files)}")

        total_sar += len(sar_files)
        total_gt += len(gt_files)

    print("\nSummary")
    print(f"Total SAR images: {total_sar}")
    print(f"Total GT masks  : {total_gt}")

    if total_sar == total_gt:
        print("\nDataset check completed successfully.")
    else:
        print("\nWarning: Number of SAR images and GT masks is different.")


if __name__ == "__main__":
    main()
