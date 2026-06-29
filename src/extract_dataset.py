from pathlib import Path
import tarfile

from src.config import RAW_DATA_DIR


def extract_dataset():

    archives = list(RAW_DATA_DIR.glob("*.tar.gz"))

    if not archives:
        print("No dataset archive found inside data/raw")
        return

    archive = archives[0]

    print(f"Found archive:\n{archive.name}")

    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(RAW_DATA_DIR)

        print("\nDataset extracted successfully.")

    except Exception as error:
        print(f"\nExtraction failed:\n{error}")


if __name__ == "__main__":
    extract_dataset()
