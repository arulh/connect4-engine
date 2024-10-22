import torch


class Connect4Board:
    """
    Connect4Board class for the Connect4 environment
    """

    NUM_ROWS = 6
    NUM_COLUMNS = 7

    def __init__(self):
        self.board = torch.zeros(self.NUM_ROWS, self.NUM_COLUMNS, dtype=torch.float32)
        self.col_indices = torch.zeros(self.NUM_COLUMNS, dtype=torch.int8)
        self.turn = 0
        self.action_space = self.NUM_COLUMNS
        self.current_player = 1 # 1 for agent, 2 for the opponent
        
    def __repr__(self) -> str:
        board_str = '\n'.join([' | '.join(map(lambda r: str(int(r.item())), row)) for row in self.board])
        column_headers = " 0   1   2   3   4   5   6"
        separator = "-----------------------------"
        return f"Turn: {self.turn}\n{column_headers}\n{separator}\n{board_str}"
    
    def step(self, action: int):
        """Places piece for current_player in column specified by action"""

        if self.col_indices[action] == self.NUM_ROWS: # no more room in column
            return self.board, torch.tensor(-0.5, dtype=torch.float32), torch.tensor(0), {}
        
        # makes the move
        self.board[self.NUM_ROWS-1-self.col_indices[action], action] = self.current_player
        self.col_indices[action] += 1

        # check if there is a winner
        if (self._check_winner()):
            return self.board, torch.tensor(1.0, dtype=torch.float32), torch.tensor(1), {"winner": self.current_player}

        # check if there is a draw
        if (not self.get_valid_moves().numel()):
            return self.board, torch.tensor(0.0, dtype=torch.float32), torch.tensor(1), {"winner": 0}

        self.current_player = self.current_player ^ 3 # switch the player
        self.turn += 1

        return self.board, torch.tensor(0.0, dtype=torch.float32), torch.tensor(0), {}

    def reset(self):
        self.board = torch.zeros(self.NUM_ROWS, self.NUM_COLUMNS, dtype=torch.float32)
        self.turn = 0
        self.current_player = 1
        self.col_indices = torch.zeros(self.NUM_COLUMNS, dtype=torch.int8)

        return self.board

    def load_board(self, board: torch.Tensor):
        pass

    def _check_winner(self):
        rows, cols = self.board.shape
        player_pieces = torch.where(self.board == self.current_player, 1, 0)
        
        # Check horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if self.board[r, c] == player_pieces[r, c+1] == player_pieces[r, c+2] == player_pieces[r, c+3] and player_pieces[r, c] != 0:
                    return self.board[r, c]
        
        # Check vertical
        for r in range(rows - 3):
            for c in range(cols):
                if player_pieces[r, c] == player_pieces[r+1, c] == player_pieces[r+2, c] == player_pieces[r+3, c] and player_pieces[r, c] != 0:
                    return True
        
        # Check diagonal (bottom-left to top-right)
        for r in range(rows - 3):
            for c in range(cols - 3):
                if player_pieces[r, c] == player_pieces[r+1, c+1] == player_pieces[r+2, c+2] == player_pieces[r+3, c+3] and player_pieces[r, c] != 0:
                    return True
        
        # Check diagonal (top-left to bottom-right)
        for r in range(3, rows):
            for c in range(cols - 3):
                if player_pieces[r, c] == player_pieces[r-1, c+1] == player_pieces[r-2, c+2] == player_pieces[r-3, c+3] and player_pieces[r, c] != 0:
                    return True
        
        # If no winner, return 0
        return False

    def get_valid_moves(self):
        return torch.where(self.col_indices < self.NUM_ROWS, 1, 0).nonzero().flatten()