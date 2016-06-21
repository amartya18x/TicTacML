from blocks.bricks import MLP
from blocks.bricks import Rectifier, Softmax
from blocks.initialization import IsotropicGaussian, Constant
import theano.tensor as T
import theano
from blocks.graph import ComputationGraph
from optimization import Adam
import numpy as np
rng = theano.tensor.shared_randomstreams.RandomStreams(2)


class policy_agent(object):

    def __init__(self, hidden_dim, decay):
        self.hidden_dim = hidden_dim
        board_input, mlp, output, chosen = self.build_model(self.hidden_dim)
        self.cg = ComputationGraph(output)
        gradient_temp = T.grad(output, self.cg.parameters)
        self.gradient = []
        for param in self.cg.parameters:
            print(np.zeros_like(param.get_value(), dtype=np.float))
            self.gradient = self.gradient + \
                [theano.shared(np.zeros_like(
                    param.get_value(), dtype=np.float))]
        print(self.gradient)
        self.update = []
        for idx, gr in enumerate(self.gradient):
            self.update = self.update + \
                [(self.gradient[idx], self.gradient[idx] + gradient_temp[idx])]
        self.f = theano.function([board_input],
                                 chosen,
                                 updates=self.update)
        self.param_update = Adam(self.gradient, self.cg.parameters)
        self.update_fn = theano.function([], updates=self.param_update)

        print("Created Policy agent")

    def build_model(self, hidden_dim):
        board_input = T.vector('input')
        mlp = MLP(activations=[Rectifier(), Softmax()],
                  dims=[9, hidden_dim, 9],
                  weights_init=IsotropicGaussian(),
                  biases_init=Constant(0.01))
        output = mlp.apply(board_input)
        mlp.initialize()
        cost, chosen = self.get_cost(output)
        return board_input, mlp, cost, chosen

    def get_cost(self, output):
        chosen = T.flatten(rng.multinomial(pvals=output))
        chosen.name = 'chosen'
        chosen_id = T.argmax(chosen)
        chosen = theano.gradient.disconnected_grad(chosen)
        return T.sum(chosen * output), chosen_id

    def new_game(self):
        self.gradient = []
        for param in self.cg.parameters:
            self.gradient = self.gradient + [T.zeros_like(param)]

    def get_move(self, input_feature):
        print(input_feature)
        move = self.f(input_feature)
        return (int(move / 3), move % 3)

    def update_params(self, weight):
        for param in self.cg.parameters:
            self.gradient = weight * self.gradient
        self.update_fn()
        self.new_game()
