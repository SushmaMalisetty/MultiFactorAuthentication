import shutil
import os

dataset_path = "dataset/user1"

if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)
    print("Dataset deleted successfully!")
else:
    print("Dataset folder not found!")
