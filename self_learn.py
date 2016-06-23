from __future__ import print_function
from policy_agent import policy_agent
from game import Board
from pastalog import Log
import numpy as np

if __name__ == '__main__':
    log_a = Log('http://localhost:8120', 'modelb')
    board = Board()
    RA1 = policy_agent(1000, 0.01)
    RA2 = policy_agent(1000, 0.01)
    policy2_wins = 0
    policy1_wins = 0
    draws = 0
    doomed = 0
    exit = 1
    win_ratio = 0
    for i in range(1000000):
        wrong_move = 0
        board = Board()
        start = np.random.uniform() > 0.5
        first = False
        while(board.check_win() == 0 and np.any(board.board == 0)):
            episodes = 0
            if first or start:
                while board.play_tac(*(RA1.get_move(
                        board.get_feature_vec(board.tac)))) is False:
                    pass
                if not np.any(board.board == 0):
                    break
            episodes = 0
            while board.play_tic(*(RA2.get_move(
                    board.get_feature_vec(board.tic)))) is False:
                pass
            first = True
        RA2.update_params((board.check_win() - 0) * -200 * board.tic)
        RA1.update_params((board.check_win() + 0) * -200 * board.tac)
        if board.check_win() == board.tic:
            policy1_wins = policy1_wins + 1
        elif board.check_win() == 0:
            draws = draws + 1
        else:
            policy2_wins = policy2_wins + 1
        if i % 100 == 0:
            board.print_board()
            print("Number of Agent1_wins " + str(policy1_wins))
            print("Number of Agent2_wins " + str(policy2_wins))
            print("Number of draw " + str(draws))
            print("Number of dooms " + str(doomed))

            log_a.post('Policy1 Wins', value=policy1_wins, step=i / 100)
            log_a.post('Policy2 Wins Wins', value=policy2_wins, step=i / 100)
            log_a.post('Draws', value=draws, step=i / 100)
            policy2_wins = 0
            policy1_wins = 0
            draws = 0
            doomed = 0
