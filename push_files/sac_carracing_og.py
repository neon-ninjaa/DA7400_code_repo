import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
import matplotlib.pyplot as plt
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Create the environment
env = gym.make("CarRacing-v2", render_mode="rgb_array")
env = Monitor(env)
env = DummyVecEnv([lambda: env])


model = SAC(
    "CnnPolicy",
    env,
    verbose=1,
    buffer_size=10000,         # Reduced replay buffer size
    batch_size=64,             # Reduced batch size
    learning_starts=100,       # Start learning after collecting 100 steps
    tensorboard_log="./sac_carracing_tensorboard/",
    device=device              # Set device to GPU if available
)

# Train the model
curr_time = time.time()
model.learn(total_timesteps=10000)  # Adjust based on available memory
print("Training completed: ", time.time() - curr_time)

# Save the model
# model.save("sac_carracing")

# Load and test the model
model = SAC.load("sac_carracing", env=env, device=device)

# Enjoy the trained agent

episode_returns = []

for _ in range(10):
    obs = env.reset()
    episode_return = 0
    i = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        episode_return += reward[0]
        i += 1
        if done or i > 500:
            break
    episode_returns.append(episode_return)

env.close()

# Plotting the overall returns from each episode
plt.plot(episode_returns)
plt.xlabel('Episode')
plt.ylabel('Total Return')
plt.title('Total Return per Episode')
plt.show()
