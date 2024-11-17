import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from stable_baselines3 import SAC
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from iphyre.simulator import IPHYRE
import logging
from iphyre.games import GAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=64):
        super(Encoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.fc_layers(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, vocab_size, embedding_dim, hidden_dim, seq_len):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.seq_len = seq_len

    def forward(self, z):
        hidden = self.latent_to_hidden(z).unsqueeze(0).to(device)
        cell = torch.zeros_like(hidden).to(device)
        outputs = []
        input_token = torch.zeros(z.size(0), 1, dtype=torch.long, device=device)

        for _ in range(self.seq_len):
            embedded = self.embedding(input_token)
            lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            logits = self.fc(lstm_out.squeeze(1))
            input_token = logits.argmax(dim=1).unsqueeze(1)
            outputs.append(logits)

        return torch.stack(outputs, dim=1)

def describe_observation(observation):
    """
    Convert an observation represented as a 2D list (matrix) into a descriptive text format.

    Args:
        observation (list of list): A 2D list where each inner list represents an object's features.

    Returns:
        str: A descriptive text of the observation.
    """
    description = []

    # General information about the environment and objective
    description.append("The observation for the scene is processed in the symbolic space, represented as a 12 x 9 matrix.")
    description.append("Each row denotes one object's features in the scene.")
    description.append("The goal of the environment is to make the ball (Object 3) fall freely by removing specific objects that are allowed to be eliminated, without disturbing the fixed structures supporting the ball.")
    description.append("Below is a detailed description of each object in the scene:\n")

    for i, obj in enumerate(observation):
        if len(obj) != 9:
            continue 

        position_x1, position_y1, position_x2, position_y2, size, eliminable, fixed, joint, spring = obj
        
        eliminable_str = "can be eliminated" if eliminable else "cannot be eliminated"
        fixed_str = "is stationary" if fixed else "can be moved"
        joint_str = "is connected to a joint" if joint else "is not connected to a joint"
        spring_str = "is connected to a spring" if spring else "is not connected to a spring"
        
        if i == 2:  # Object 3, assumed to be the ball
            description.append("Object 3 (Ball):")
            description.append(f"  - Position: ({position_x1}, {position_y1}) to ({position_x2}, {position_y2})")
            description.append(f"  - Size (radius): {size}")
            description.append("  - This is the target object, and the goal is to allow it to fall by clearing obstacles beneath it.\n")
        else:
            description.append(f"Object {i + 1}:")
            description.append(f"  - Position: ({position_x1}, {position_y1}) to ({position_x2}, {position_y2})")
            description.append(f"  - Size (radius): {size}")
            description.append(f"  - Eliminable Indicator: This object {eliminable_str}.")
            description.append(f"  - Fixed Object Indicator: This object {fixed_str}.")
            description.append(f"  - Joint Indicator: This object {joint_str}.")
            description.append(f"  - Spring Indicator: This object {spring_str}.")
            description.append("")  # Blank line for readability

    return "\n".join(description)


class TextRLWrapper(gym.Env):
    def __init__(self, base_env, encoder, decoder, tokenizer, bert_model, latent_dim, vocab_size):
        super(TextRLWrapper, self).__init__()
        self.base_env = base_env
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.tokenizer = tokenizer
        self.bert_model = bert_model.to(device)
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)
        self.action_space = spaces.Discrete(7)

    def reset(self, seed=None, **kwargs):
        raw_obs = self.base_env.reset()
        raw_text = describe_observation(raw_obs)
        encoded_obs = self.process_text(raw_text)
        return encoded_obs, {}

    def step(self, action):
        action_space = self.base_env.get_action_space()
        action_1 = action_space[action]
        obs, reward, done = self.base_env.step(action_1)
        text_obs = describe_observation(obs)
        encoded_obs = self.process_text(text_obs)
        return encoded_obs, reward, done

    def process_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding='max_length'
        ).to(device)
        
        with torch.no_grad():
            bert_output = self.bert_model(**inputs)
            latent_vector = self.encoder(bert_output.last_hidden_state.mean(dim=1)).detach()
        
        return latent_vector.cpu().numpy()

    def update_vqa(self, latent_vector, target_text):
        inputs = self.tokenizer(target_text, return_tensors="pt").input_ids.to(device)
        output = self.decoder(latent_vector)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.view(-1, self.vocab_size), inputs.view(-1))
        
        loss.backward()

        
        return latent_vector

latent_dim = 16
seq_len = 20
vocab_size = 30522
embedding_dim = 256
hidden_dim = 256

encoder = Encoder(input_dim=768, latent_dim=64).to(device)
decoder = Decoder(latent_dim, vocab_size, embedding_dim, hidden_dim, seq_len).to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.00001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.00001)

base_env = IPHYRE(game='support')
wrapped_env = TextRLWrapper(base_env, encoder, decoder, tokenizer, bert_model, latent_dim, vocab_size)
env = DummyVecEnv([lambda: wrapped_env])

sac_model = DQN("MlpPolicy", env, verbose=1)
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='dqn_model')

def evaluate_game_on_angle(model, num_episodes=3):
    rewards = []
    base_env = IPHYRE(game='support')
    wrapped_env = TextRLWrapper(base_env, encoder, decoder, tokenizer, bert_model, latent_dim, vocab_size)
    env = DummyVecEnv([lambda: wrapped_env])
    obs = env.reset()
    
    for episode in range(num_episodes):
        total_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                obs = env.reset()
        
        rewards.append(total_reward)
    
    return rewards

if __name__ == '__main__':
    curr_time = time.time()

    for i in range(1000):
        print(i)
        sac_model.learn(total_timesteps=30, reset_num_timesteps=False,callback=checkpoint_callback)
        encoder_optimizer.step()
        # decoder_optimizer.step()
        encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        if i%100 == 0:
            print(i,"steps")
    print("Time taken :", time.time() - curr_time)
    total_reward_angle = evaluate_game_on_angle(sac_model)
    print(f'Total Reward on Angle: {total_reward_angle}')


# Training loop
curr_time = time.time()
for i in range(1000):
    print(i)
    sac_model.learn(total_timesteps=30, reset_num_timesteps=False)
    encoder_optimizer.step()
    # decoder_optimizer.step()
    encoder_optimizer.zero_grad()
    # decoder_optimizer.zero_grad()
    if i%100 == 0:
        print(i,"steps")
        
print("Time taken :", time.time() - curr_time)

# Evaluation
obs = env.reset()
for _ in range(1000):
    action, _states = sac_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

# Plotting the rewards
rewards = evaluate_game_on_angle(sac_model, num_episodes=10)
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward over Episodes")
plt.show()


