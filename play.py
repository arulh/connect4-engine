import torch
from src.connect4 import Connect4Board, ConnectFourDQN


env = Connect4Board()
model = ConnectFourDQN()
model.load_state_dict(torch.load("./models/C4_modelV1.pth"))
model.eval()

done = False
state = env.reset()

print("You are playing Connect Four against the AI!")
print(env)

while not done:
    # Human player move
    valid_actions = env.get_valid_moves()
    print(f"Valid actions: {valid_actions}")

    # Get input from the human player
    action = int(input(f"Your turn! Choose a column (0-{env.NUM_COLUMNS - 1}): "))
    while action not in valid_actions:
        print(f"Invalid action! Choose a valid column: {valid_actions}")
        action = int(input(f"Choose a column (0-{env.NUM_COLUMNS - 1}): "))

    # Step environment with human move
    state, reward, done, info = env.step(action)
    print(env)

    if done:
        if info['winner'] == 1:
            print("You win!")
        elif info['winner'] == -1:
            print("The AI wins!")
        else:
            print("It's a draw!")
        break

    # AI's turn (DQN model)
    with torch.no_grad():
        # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        state_tensor = state.clone().detach().unsqueeze(0)
        q_values = model(state_tensor)
        ai_action = q_values.argmax().item()

    print(f"AI chooses action: {ai_action}")

    # Step environment with AI move
    state, reward, done, info = env.step(ai_action)
    print(env)

    if done:
        if info['winner'] == 1:
            print("You win!")
        elif info['winner'] == -1:
            print("The AI wins!")
        else:
            print("It's a draw!")