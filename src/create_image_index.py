from pathlib import Path
import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


DATASET_DIR = RAW_DATA_DIR / "urban_sar_floods"

CLASSES = {
    "01_NF": "Non-Flood",
    "02_FO": "Flood Open Area",
    "03_FU": "Flood Urban",
}


def main():
    if not DATASET_DIR.exists():
        print("Dataset folder not found.")
        return

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for folder, class_name in CLASSES.items():
        sar_dir = DATASET_DIR / folder / "SAR"
        gt_dir = DATASET_DIR / folder / "GT"

        for sar_path in sorted(sar_dir.glob("*.tif")):
            gt_name = sar_path.name.replace("_SAR.tif", "_GT.tif")
            gt_path = gt_dir / gt_name

            if gt_path.exists():
                rows.append({
                    "image_path": str(sar_path),
                    "mask_path": str(gt_path),
                    "class_folder": folder,
                    "class_name": class_name,
                })
            else:
                print(f"Missing GT mask for: {sar_path.name}")

    df = pd.DataFrame(rows)

    output_path = PROCESSED_DATA_DIR / "image_index.csv"
    df.to_csv(output_path, index=False)

    print(f"Image index saved to: {output_path}")
    print(f"Total samples: {len(df)}")

    print("\nClass distribution:")
    print(df["class_name"].value_counts())


if __name__ == "__main__":
    main()
