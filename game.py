from __future__ import print_function
from random_agent import random_agent
from policy_agent import policy_agent
import numpy as np
import pandas as pd


class Board(object):

    def __init__(self):
        self.tic = -1
        self.tac = 1
        self.board = np.zeros([3, 3])

    def print_board(self):
        print("=======================")
        df = pd.DataFrame(self.board)
        df = df.replace({0: '-', 1: 'X', -1: "O"})
        print(df)
        print("=======================")

    def check_valid(self, loc_x, loc_y):
        if self.board[loc_x][loc_y] == 0:
            return True
        else:
            return False

    def play_tic(self, loc_x, loc_y):
        if self.check_valid(loc_x, loc_y):
            self.board[loc_x][loc_y] = self.tic
            return True
        else:
            return False

    def play_tac(self, loc_x, loc_y):
        if self.check_valid(loc_x, loc_y):
            self.board[loc_x][loc_y] = self.tac
            return True
        else:
            return False

    def get_feature_vec(self, player):
        return np.ravel(self.board) * player

    def check_win(self):
        col_sum = np.sum(self.board, axis=1)
        row_sum = np.sum(self.board, axis=0)
        if np.any(col_sum == 3) or np.any(row_sum == 3):
            return 1
        elif np.any(col_sum == -3) or np.any(row_sum == -3):
            return -1
        elif self.board[0][0] + self.board[1][1] + self.board[2][2] == 3:
            return 1
        elif self.board[0][0] + self.board[1][1] + self.board[2][2] == -3:
            return -1
        elif self.board[0][2] + self.board[1][1] + self.board[2][0] == 3:
            return 1
        elif self.board[0][2] + self.board[1][1] + self.board[2][0] == -3:
            return -1
        else:
            return 0


if __name__ == '__main__':
    board = Board()
    RA1 = random_agent()
    RA2 = policy_agent(10, 0)
    while(board.check_win() == 0 and np.any(board.board == 0)):
        episodes = 0
        print(board.get_feature_vec(board.tic))
        while board.play_tac(*(RA1.get_move())) is False:
            episodes = episodes + 1
            if episodes == 9:
                break
            pass
        if not np.any(board.board == 0):
            break
        episodes = 0
        print(board.get_feature_vec(board.tac))
        while board.play_tic(*(RA2.get_move(
                board.get_feature_vec(board.tac)))) is False:
            if episodes == 9:
                break
            episodes = episodes + 1
            pass
        board.print_board()

    board.print_board()
    print(board.check_win())
