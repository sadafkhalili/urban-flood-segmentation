import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_DIR,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def main():
    index_path = PROCESSED_DATA_DIR / "image_index.csv"

    if not index_path.exists():
        print("Image index does not exist.")
        print("Run create_image_index first.")
        return

    df = pd.read_csv(index_path)

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["class_name"],
    )

    valid_size_adjusted = VALID_RATIO / (VALID_RATIO + TEST_RATIO)

    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_size_adjusted,
        random_state=RANDOM_SEED,
        stratify=temp_df["class_name"],
    )

    train_path = PROCESSED_DATA_DIR / "train.csv"
    valid_path = PROCESSED_DATA_DIR / "validation.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Dataset split completed.")
    print(f"Train samples     : {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples      : {len(test_df)}")

    print("\nTrain distribution:")
    print(train_df["class_name"].value_counts())

    print("\nValidation distribution:")
    print(valid_df["class_name"].value_counts())

    print("\nTest distribution:")
    print(test_df["class_name"].value_counts())


if __name__ == "__main__":
    main()
