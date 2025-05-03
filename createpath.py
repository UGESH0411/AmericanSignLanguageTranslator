import os
# Base path
base_path = r"D:\SignLanguageTranslator"

# Folder structure
folders = [
    "Data\Train",
    "Data\Test",
    "Data\Validation",
    "Model",
    "Scripts",
    "Logs",
]

# Create folders
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created folder: {folder_path}")