import matplotlib.pyplot as plt
import numpy as np


def draw_connect4_board(tensor, filename: str):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Get the shape of the tensor (should be 6 rows x 7 columns for a Connect 4 board)
    rows, cols = tensor.shape
    
    # Loop over each cell in the tensor
    for row in range(rows):
        for col in range(cols):
            if tensor[row, col] == 0:
                color = 'white'  # Empty spots
            elif tensor[row, col] == 1:
                color = 'red'    # Player 1
            elif tensor[row, col] == 2:
                color = 'yellow' # Player 2
            
            # Draw a circle in each position
            circle = plt.Circle((col, rows - row - 1), 0.4, color=color, ec='black')
            ax.add_patch(circle)
    
    # Set the limits and aspect
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal')

    # Set the grid and hide axis labels
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(True, which='both')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add column indices at the top after flipping
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels(np.arange(1, cols + 1))

    # Add a legend for the players
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Player 1')
    yellow_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=15, label='Player 2')
    ax.legend(handles=[red_patch, yellow_patch], loc='upper right')

    plt.savefig(filename, bbox_inches='tight')