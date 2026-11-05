
import torch

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    device = torch.device("mps")

    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    result = torch.matmul(x, y)

    print("GPU device:", device)
    print("Matrix multiplication successful!")
    print("Result shape:", result.shape)
else:
    print("MPS unavailable. Check PyTorch and macOS setup.")
