from blocks.bricks import MLP
from blocks.bricks import LeakyRectifier, Softmax
from blocks.initialization import IsotropicGaussian, Constant
import theano.tensor as T
import theano
from blocks.graph import ComputationGraph
from optimization import Adam
import numpy as np


rng = theano.tensor.shared_randomstreams.RandomStreams(2)


class policy_agent(object):
    '''
    This is the policy agent class. It follows the
    policy gradient theorem to select policy from the action space.
    '''

    def __init__(self, hidden_dim, decay):
        ''' Class Constructor

        Parameters
        ----------

        hidden_dim : Dimension of the hidden layers

        decay : Decay of the effects of the previous
        steps on the final answer'''

        self.hidden_dim = hidden_dim
        board_input, mlp, output, chosen, probs =\
            self.build_model(self.hidden_dim)
        self.cg = ComputationGraph(output)
        gradient_temp = T.grad(output, self.cg.parameters)
        self.gradient = []
        for param in self.cg.parameters:
            self.gradient = self.gradient + \
                [theano.shared(np.zeros_like(
                    param.get_value(), dtype=np.float64))]
        self.update = []
        for idx, gr in enumerate(self.gradient):
            self.update = self.update + \
                [(self.gradient[idx], self.gradient[idx] + gradient_temp[idx])]
        self.f = theano.function([board_input],
                                 [chosen, probs, output],
                                 updates=self.update)
        self.param_update = Adam(self.gradient, self.cg.parameters)
        self.update_fn = theano.function([], updates=self.param_update)

        print("Created Policy agent")

    def build_model(self, hidden_dim):
        board_input = T.vector('input')
        mlp = MLP(activations=[LeakyRectifier(0.1), LeakyRectifier(0.1)],
                  dims=[9, hidden_dim,  9],
                  weights_init=IsotropicGaussian(0.00001),
                  biases_init=Constant(0.01))
        output = mlp.apply(board_input)
        masked_output = Softmax().apply(output * T.eq(board_input, 0) * 1000)
        mlp.initialize()
        cost, chosen = self.get_cost(masked_output)
        return board_input, mlp, cost, chosen, output

    def get_cost(self, output):
        chosen = T.flatten(rng.multinomial(pvals=output))
        chosen.name = 'chosen'
        chosen_id = T.argmax(chosen)
        chosen = theano.gradient.disconnected_grad(chosen)
        return T.sum(chosen * T.log(output)), chosen_id

    def new_game(self):
        self.gradient = []
        for param in self.cg.parameters:
            self.gradient = self.gradient + \
                [theano.shared(np.zeros_like(
                    param.get_value(), dtype=np.float64))]

    def get_move(self, input_feature):
        move, probs, cost = self.f(input_feature)
        t = (int(move / 3), move % 3)
        return t

    def update_params(self, weight):
        for idx, param in enumerate(self.cg.parameters):
            self.gradient[idx] = weight * self.gradient[idx]
        self.update_fn()
        self.new_game()
