import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Define the Forward Process (Diffusion)
class DiffusionProcess:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)  # Noise schedule
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # Cumulative alpha
    
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise, noise  # Noisy image + actual noise

# 2. Define the U-Net Model for Denoising (Simple Version)
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t):
        return self.model(x)  # Predicts noise in the input

# 3. Training the Model
def train(model, diffusion, dataloader, optimizer, epochs=5):
    model.train()
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for x0, _ in dataloader:
            x0 = x0.to(device)
            t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device)
            x_t, noise = diffusion.add_noise(x0, t)
            predicted_noise = model(x_t, t)

            loss = criterion(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 4. Sampling (Generating New Images)
def sample(model, diffusion, img_size=(1, 28, 28), num_samples=16):
    model.eval()
    with torch.no_grad():
        x_t = torch.randn((num_samples,) + img_size, device=device)
        for t in reversed(range(diffusion.T)):
            predicted_noise = model(x_t, torch.full((num_samples,), t, device=device))
            x_t = (x_t - predicted_noise * torch.sqrt(diffusion.beta[t])) / torch.sqrt(diffusion.alpha[t])
        return x_t

# Load Dataset (MNIST)
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize Model and Train
diffusion = DiffusionProcess()
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(model, diffusion, dataloader, optimizer, epochs=5)

# Generate Images
samples = sample(model, diffusion)
plt.imshow(samples[0].squeeze().cpu(), cmap="gray")
plt.show()
