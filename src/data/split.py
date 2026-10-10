from pathlib import Path
import pandas as pd
import re
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def create_dataframe(image_dir, labels_csv_path):
    image_dir = Path(image_dir)
    labels_csv_path = Path(labels_csv_path)

    # Load the single CSV and create an integer-to-count mapping
    df_labels = pd.read_csv(labels_csv_path)
    label_map = dict(zip(df_labels['id'], df_labels['count']))

    records = []

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Extract the numeric ID from the filename (e.g., "seq 0001" -> 1)
        match = re.search(r'\d+', image_path.stem)
        if not match:
            continue

        lookup_key = int(match.group())

        # Check if this image's ID has a corresponding label in the CSV
        if lookup_key not in label_map:
            continue

        count = int(label_map[lookup_key])

        records.append({
            "image": str(image_path), # Keeps your structure of storing the relative path
            "count": count
        })

    return pd.DataFrame(records)


def create_splits():
    # Updated paths to match your actual data layout
    df = create_dataframe(
        "data/raw/frames",
        "data/raw/count/labels.csv"
    )

    # 70% Train, 30% Temp (which becomes 15% Val / 15% Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42
    )

    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    create_splits()