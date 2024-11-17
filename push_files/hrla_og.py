import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BlipProcessor, BlipForConditionalGeneration
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from PIL import Image

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Encoder for Image Feature Extraction
class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=64):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ).to(device)

        # Calculate flattened dimension dynamically
        dummy_input = torch.zeros(1, input_channels, 96, 96).to(device)
        conv_output = self.conv_layers(dummy_input).to(device)
        self.flatten_dim = conv_output.view(-1).size(0)
        self.fc = nn.Linear(self.flatten_dim, latent_dim).to(device)  # Ensure this is on the same device

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Decoder for Goal-Conditioned Text Generation
class Decoder(nn.Module):
    def __init__(self, latent_dim, vocab_size, embedding_dim, hidden_dim, seq_len):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)  # Ensure embedding layer is on the same device
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True).to(device)  # Ensure LSTM is on the same device
        self.fc = nn.Linear(hidden_dim, vocab_size).to(device)  # Ensure output layer is on the same device
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim).to(device)  # Ensure this is on the same device
        self.seq_len = seq_len

    def forward(self, z):
        hidden = self.latent_to_hidden(z).unsqueeze(0)
        cell = torch.zeros_like(hidden).to(device)  # Move to GPU
        outputs = []
        input_token = torch.zeros(z.size(0), 1, dtype=torch.long, device=z.device)

        for _ in range(self.seq_len):
            embedded = self.embedding(input_token)
            lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            logits = self.fc(lstm_out.squeeze(1))
            input_token = logits.argmax(dim=1).unsqueeze(1)
            outputs.append(logits)

        return torch.stack(outputs, dim=1)

class ImageRLWrapper(gym.Env):
    def __init__(self, base_env, encoder, decoder, blip_processor, blip_model, latent_dim, vocab_size):
        super(ImageRLWrapper, self).__init__()
        self.base_env = base_env
        self.encoder = encoder.to(device)  # Move encoder to GPU
        self.decoder = decoder.to(device)  # Move decoder to GPU
        self.blip_processor = blip_processor
        self.blip_model = blip_model.to(device)  # Move BLIP model to GPU
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # Update observation space to match encoder output dimension
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)
        self.action_space = base_env.action_space

    def reset(self, seed = None, **kwargs):
        raw_obs = self.base_env.reset()
        encoded_obs = self.process_image(raw_obs[0])
        return encoded_obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)  # Ensure action is within valid bounds
        action = action.astype(np.float32)  # Ensure the action is float32
        obs, reward, done, _, info = self.base_env.step(action)
        encoded_obs = self.process_image(obs)
        return encoded_obs, reward, done, _, info

    def process_image(self, image):
        if isinstance(image, tuple):
            image = image[0]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError("Invalid image type. Must be a numpy array or PIL Image.")

        image_tensor = torch.tensor(np.array(image).copy().transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0  # Move to GPU
        image_tensor = image_tensor.to(device)
        question = "Where should the car move next?"
        inputs = self.blip_processor(image, question, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU
        blip_output = self.blip_model.generate(**inputs)
        generated_text = self.blip_processor.decode(blip_output[0], skip_special_tokens=True)

        latent_vector = self.update_vqa(image_tensor, generated_text)
        # print("Generated Caption:", generated_text)
        
        return latent_vector.cpu().numpy()  # Return to CPU for numpy conversion

    def update_vqa(self, image_tensor, target_text):
        latent_vector = self.encoder(image_tensor).detach()
        inputs = self.blip_processor.tokenizer(target_text, return_tensors="pt").input_ids.to(device)  # Move to GPU
        output = self.decoder(latent_vector)
        
        output_len, target_len = output.shape[1], inputs.shape[1]
        
        if output_len > target_len:
            output = output[:, :target_len, :]
        elif output_len < target_len:
            padding = torch.zeros((output.shape[0], target_len - output_len, output.shape[2]), device=output.device)
            output = torch.cat((output, padding), dim=1)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.view(-1, self.vocab_size), inputs.view(-1))
        
        loss.backward()
        decoder_optimizer.step()
        decoder_optimizer.zero_grad()
        
        return latent_vector

# Initialize Components
latent_dim = 16
seq_len = 20
vocab_size = 30522
embedding_dim = 256
hidden_dim = 256

encoder = Encoder(input_channels=3, latent_dim=latent_dim)
decoder = Decoder(latent_dim, vocab_size, embedding_dim, hidden_dim, seq_len)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(device)  # Move to GPU

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

base_env = gym.make("CarRacing-v2")

wrapped_env = ImageRLWrapper(base_env, encoder, decoder, blip_processor, blip_model, latent_dim, vocab_size)
env = DummyVecEnv([lambda: wrapped_env])

sac_model = SAC("MlpPolicy", env, verbose=1)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix='sac_model')
save_path = 'saved_models/'
for i in range(10000):
    sac_model.learn(total_timesteps=1, reset_num_timesteps=False)
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    if i % 100 == 0:
        sac_model.save(f"{save_path}/sac_model_{i}.zip")
        torch.save(encoder.state_dict(), f"{save_path}/encoder_{i}.pt")
        torch.save(decoder.state_dict(), f"{save_path}/decoder_{i}.pt")
        print(f"Saved models at timestep {i}")

obs = env.reset()
for _ in range(1000):
    action, _states = sac_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
