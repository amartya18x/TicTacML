from __future__ import print_function
from random_agent import random_agent
from policy_agent import policy_agent
from game import Board
from pastalog import Log
import numpy as np

if __name__ == '__main__':
    log_a = Log('http://localhost:8120', '')
    board = Board()
    RA1 = policy_agent(1000, 0.01)
    RA2 = policy_agent(1000, 0.01)
    RA3 = random_agent()
    policy2_wins = 0
    policy1_wins = 0
    draws = 0
    doomed = 0
    exit = 1
    win_ratio = 0
    random_wins = 0
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
            log_a.post('Policy2 Wins ', value=policy2_wins, step=i / 100)
            log_a.post('P1 - P2 Draws', value=draws, step=i / 100)
            policy2_wins = 0
            policy1_wins = 0
            draws = 0
            doomed = 0
            for j in range(1, 101):
                wrong_move = 0
                board = Board()
                while(board.check_win() == 0 and np.any(board.board == 0)):
                    episodes = 0
                    exit = 1
                    while board.play_tac(*(RA3.get_move())) is False:
                        pass
                    if not np.any(board.board == 0):
                        break
                    while board.play_tic(*(RA2.get_move(
                            board.get_feature_vec(board.tic)))) is False:
                        pass
                    RA2.update_params((board.check_win() -
                                       0.5) * -200 * board.tic)
                if board.check_win() == board.tic:
                    policy2_wins = policy2_wins + 1
                elif board.check_win() == 0:
                    draws = draws + 1
                else:
                    random_wins = random_wins + 1
            board.print_board()
            print("Number of policy2_wins " + str(policy2_wins))
            print("Number of random_wins " + str(random_wins))
            print("Number of draw " + str(draws))
            print("Number of dooms " + str(doomed))
            log_a.post('Policy2 Wins vs Random_',
                       value=policy2_wins, step=i/100)
            log_a.post('Random Wins vs policy2 ',
                       value=random_wins, step=i/100)
            log_a.post('Draws P2 vs R', value=draws, step=i/100)
            random_wins = 0
            policy2_wins = 0
            draws = 0
            doomed = 0

            for j in range(1, 101):
                wrong_move = 0
                board = Board()
                while(board.check_win() == 0 and np.any(board.board == 0)):
                    episodes = 0
                    exit = 1
                    while board.play_tic(*(RA3.get_move())) is False:
                        pass
                    if not np.any(board.board == 0):
                        break
                    while board.play_tac(*(RA1.get_move(
                            board.get_feature_vec(board.tac)))) is False:
                        pass
                    RA1.update_params((board.check_win() +
                                       0.5) * -200 * board.tac)
                if board.check_win() == board.tac:
                    policy1_wins = policy1_wins + 1
                elif board.check_win() == 0:
                    draws = draws + 1
                else:
                    random_wins = random_wins + 1
            board.print_board()
            print("Number of policy1_wins vs Random " + str(policy1_wins))
            print("Number of random_wins vs Policy 1" + str(random_wins))
            print("Number of draw " + str(draws))
            print("Number of dooms " + str(doomed))
            log_a.post('Policy1 Wins vs Random',
                       value=policy1_wins, step=i/100)
            log_a.post('Random Wins vs policy1',
                       value=random_wins, step=i/100)
            log_a.post('Draws P1 vs R', value=draws, step=i/100)
            random_wins = 0
            policy1_wins = 0
            draws = 0
            doomed = 0
