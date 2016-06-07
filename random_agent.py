import numpy as np


class random_agent(object):

    def __init__(self):
        print("Created Random Agent")

    def get_random_move(self):
        return (np.random.random_integers(3)-1, np.random.random_integers(3)-1)
