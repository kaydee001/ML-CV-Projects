import torch
from pathlib import Path

def train():
    pass

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device : {device}")

    if device == "cuda":
        print(f"gpu : {torch.cuda.get_device_name(0)}")