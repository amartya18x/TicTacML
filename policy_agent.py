from blocks.bricks import MLP
from blocks.bricks import Rectifier, Softmax
from blocks.initialization import IsotropicGaussian, Constant
import theano.tensor as T
import theano
from blocks.graph import ComputationGraph

rng = theano.tensor.shared_randomstreams.RandomStreams(2)


class policy_agent(object):

    def __init__(self, hidden_dim, decay):
        self.hidden_dim = hidden_dim
        board_input, mlp, output, chosen = self.build_model(self.hidden_dim)
        self.cg = ComputationGraph(output)
        gradient_temp = T.grad(output, self.cg.parameters)
        self.gradient = []
        for param in self.cg.parameters:
            self.gradient = self.gradient + [T.zeros_like(param)]
        self.update = []
        for idx, gr in enumerate(self.gr):
            self.update = self.update + \
                [(self.gradient[idx], self.gradient[idx] + gradient_temp[idx])]
        self.f = theano.function([board_input],
                                 chosen,
                                 updates=self.update)
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
        chosen = theano.gradient.disconnected_grad(chosen)
        return chosen * output, T.argmax(chosen)

    def new_game(self):
        self.gradient = []
        for param in self.cg.parameters:
            self.gradient = self.gradient + [T.zeros_like(param)]
