from __future__ import print_function
from random_agent import random_agent
from policy_agent import policy_agent
from game import Board
from pastalog import Log
import numpy as np

if __name__ == '__main__':
    log_a = Log('http://localhost:8120', 'modelb')
    board = Board()
    RA1 = random_agent()
    RA2 = policy_agent(1000, 0.01)
    ratio_list = []
    random_wins = 0
    policy_wins = 0
    draws = 0
    doomed = 0
    exit = 1
    win_ratio = 0
    for i in range(1000000):
        wrong_move = 0
        board = Board()
        while(board.check_win() == 0 and np.any(board.board == 0)):
            episodes = 0
            exit = 1
            while board.play_tac(*(RA1.get_move())) is False:
                pass

            if exit == 0:
                break
            if not np.any(board.board == 0):
                break
            episodes = 0
            while board.play_tic(*(RA2.get_move(
                    board.get_feature_vec(board.tic)))) is False:
                pass
            if exit == 0:
                break
        RA2.update_params((board.check_win() - 0.5) * -200 * board.tic)
        if board.check_win() == board.tic:
            policy_wins = policy_wins + 1
        elif board.check_win() == 0:
            draws = draws + 1
        else:
            random_wins = random_wins + 1
        if i % 100 == 0:
            board.print_board()
            print("Number of policy_wins " + str(policy_wins))
            print("Number of random_wins " + str(random_wins))
            print("Number of draw " + str(draws))
            print("Number of dooms " + str(doomed))
            win_ratio = policy_wins*1.0 / (random_wins+0.000001)
            
            log_a.post('Policy Wins', value=policy_wins, step=i/100)
            log_a.post('Random Wins Wins', value=random_wins, step=i/100)
            log_a.post('Draws', value=draws, step=i/100)
            random_wins = 0
            policy_wins = 0
            draws = 0
            doomed = 0
