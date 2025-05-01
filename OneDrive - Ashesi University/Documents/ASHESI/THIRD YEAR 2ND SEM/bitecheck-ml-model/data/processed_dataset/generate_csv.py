"""
import os
import pandas as pd

# Define paths
base_dir = "processed_dataset"
splits = ["train", "test", "validation"]
data = []  # Initialize the list to store rows

for split in splits:
    split_path = os.path.join(base_dir, split)
    for label in os.listdir(split_path):  # Each subfolder = class name
        label_path = os.path.join(split_path, label)
        for image_name in os.listdir(label_path):
            if image_name.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(split, label, image_name)  # Relative path
                data.append({"image_path": image_path, "label": label, "split": split})


df = pd.DataFrame(data)
df.to_csv("processed_dataset.csv", index=False)
print(f"CSV generated with {len(df)} entries!")
"""

import os
import pandas as pd

# Define paths
base_dir = "processed_dataset"
splits = ["train", "test"]
data = []  # Initialize the list to store rows

for split in splits:
    split_path = os.path.join(base_dir, split)
    for label in os.listdir(split_path):  # Each subfolder = class name
        label_path = os.path.join(split_path, label)
        for image_name in os.listdir(label_path):
            if image_name.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(split, label, image_name)  # Relative path
                data.append({"image_path": image_path, "label": label, "split": split})

df = pd.DataFrame(data)
df.to_csv("train_test_dataset.csv", index=False)
print(f"CSV generated with {len(df)} entries!")
