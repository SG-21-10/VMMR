import torch
import time

# Load your trained model — adjust this to match your actual checkpoint/loading code
model = torch.load("your_model.pt", map_location="cpu")
model.eval()

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Inference time benchmark
dummy_input = torch.randn(1, 3, 224, 224)  # match your input size
with torch.no_grad():
    for _ in range(5):
        model(dummy_input)
    start = time.time()
    for _ in range(50):
        model(dummy_input)
    avg_time = (time.time() - start) / 50

print(f"Avg inference time: {avg_time*1000:.2f} ms")
