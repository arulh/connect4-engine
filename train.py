import random
import torch
# from src.env import Connect4Board
# from src.model_utils import ConnectFourDQN
from src.connect4 import Connect4Board, ConnectFourDQN
from src.connect4.model_utils import ReplayMemory
from src.connect4.logger import draw_connect4_board
import torch.optim as optim
import torch.nn as nn


env = Connect4Board()
num_episodes = 1000
batch_size = 25
gamma = 0.9
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995
learning_rate = 1e-3
memory_capacity = 10000

dqn = ConnectFourDQN().to("cuda")
target_dqn = ConnectFourDQN().to("cuda")
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()

optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

memory = ReplayMemory(memory_capacity)
epsilon = epsilon_start

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            with torch.no_grad():
                # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cuda")
                state_tensor = state.clone().detach().unsqueeze(0).to("cuda")
                q_values = dqn(state_tensor)
                action = q_values.argmax().to("cpu")
        else:
            action = random.choice(env.get_valid_moves())

        # Step environment
        next_state, reward, done, winner  = env.step(action)
        total_reward += reward.item()

        # Store transition in memory
        memory.push((state, action, reward, next_state, done))

        done = done.item()

        # Train only if enough samples are in memory
        if len(memory) > batch_size:
            transitions = memory.sample(batch_size)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

            batch_state = torch.stack(batch_state).to("cuda")
            batch_action = torch.stack(batch_action).to("cuda")
            batch_reward = torch.stack(batch_reward).to("cuda")
            batch_next_state = torch.stack(batch_next_state).to("cuda")
            batch_done = torch.stack(batch_done).to("cuda")

            # Calculate current Q-values and target Q-values
            current_q_values = dqn(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_dqn(batch_next_state).max(1)[0]
                target_q_values = batch_reward + (gamma * next_q_values * (1 - batch_done))

            # Optimize the model
            loss = criterion(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state

    # Update epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # Update target network periodically
    if episode % 5 == 0:
        target_dqn.load_state_dict(dqn.state_dict())

    if (episode+1) % 50 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
        draw_connect4_board(env.board, f"./data/episode_{episode + 1}.png")

torch.save(dqn.state_dict(), "./models/C4_modelV1.pth")