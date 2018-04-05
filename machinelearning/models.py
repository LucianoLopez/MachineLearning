import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.025

        self.w1 = nn.Variable(1, 100)
        self.b1 = nn.Variable(100)
        self.w2 = nn.Variable(100, 1)
        self.b2 = nn.Variable(1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.w1, self.w2, self.b1, self.b2])
        input_x = nn.Input(graph, x)
        xw = nn.MatrixMultiply(graph, input_x, self.w1)
        xw_plus_b = nn.MatrixVectorAdd(graph, xw, self.b1)
        relu = nn.ReLU(graph, xw_plus_b)
        xw2 = nn.MatrixMultiply(graph, relu, self.w2)
        xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            # graph.add(xw)
            # graph.add(xw_plus_b)
            # graph.add(relu)
            # graph.add(xw2)
            # graph.add(xw2_plus_b2)
            loss = nn.SquareLoss(graph, xw2_plus_b2, input_y)
            # graph.add(loss)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            # graph.add(xw)
            # graph.add(xw_plus_b)
            # graph.add(relu)
            # graph.add(xw2)
            # graph.add(xw2_plus_b2)
            return graph.get_output(xw2_plus_b2)


class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.025

        self.w1 = nn.Variable(1, 100)
        self.b1 = nn.Variable(100)
        self.w2 = nn.Variable(100, 1)
        self.b2 = nn.Variable(1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.w1, self.w2, self.b1, self.b2])
        # ##normalx
        input_x = nn.Input(graph, x)
        xw = nn.MatrixMultiply(graph, input_x, self.w1)
        xw_plus_b = nn.MatrixVectorAdd(graph, xw, self.b1)
        relu = nn.ReLU(graph, xw_plus_b)
        xw2 = nn.MatrixMultiply(graph, relu, self.w2)
        xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.b2)
        ##negx
        neg = np.negative(np.ones((1, 1)))
        neg_input_x = nn.Input(graph, x)
        neg_input_neg = nn.Input(graph, neg)
        negx = nn.MatrixMultiply(graph, neg_input_x, neg_input_neg)
        neg_xw = nn.MatrixMultiply(graph, negx, self.w1)
        neg_xw_plus_b = nn.MatrixVectorAdd(graph, neg_xw, self.b1)
        neg_relu = nn.ReLU(graph, neg_xw_plus_b)
        neg_xw2 = nn.MatrixMultiply(graph, neg_relu, self.w2)
        neg_xw2_plus_b2 = nn.MatrixVectorAdd(graph, neg_xw2, self.b2)

        ##Now combine?
        pos_neg = nn.MatrixMultiply(graph, neg_xw2_plus_b2, neg_input_neg)
        combine = nn.MatrixVectorAdd(graph, xw2_plus_b2, pos_neg)


        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, combine, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return graph.get_output(combine)

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.17

        self.w1 = nn.Variable(784, 400)
        self.b1 = nn.Variable(400)
        self.w2 = nn.Variable(400,  10)
        self.b2 = nn.Variable(10)
        # self.w3 = nn.Variable(300, 10)
        # self.b3 = nn.Variable(10)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.w1, self.w2, self.b1, self.b2])

        input_x = nn.Input(graph, x)
        xw = nn.MatrixMultiply(graph, input_x, self.w1)
        xw_plus_b = nn.MatrixVectorAdd(graph, xw, self.b1)
        relu1 = nn.ReLU(graph, xw_plus_b)
        xw2 = nn.MatrixMultiply(graph, relu1, self.w2)
        xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.b2)
        relu2 = nn.ReLU(graph, xw2_plus_b2)
        # xw3 = nn.MatrixMultiply(graph, relu2, self.w3)
        # xw3_plus_b3 = nn.MatrixVectorAdd(graph, xw3, self.b3)
        # relu3 = nn.ReLU(graph, xw3_plus_b3)

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, relu2, input_y)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(relu2)

class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        ## .01 forsure
        self.learning_rate = .01
        self.w1 = nn.Variable(4, 100)
        self.b1 = nn.Variable(100)
        self.w2 = nn.Variable(100, 100)
        self.b2 = nn.Variable(100)
        self.w3 = nn.Variable(100, 2)
        self.b3 = nn.Variable(2)

    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
        input_states = nn.Input(graph, states)
        sw = nn.MatrixMultiply(graph, input_states, self.w1)
        sw_plus_b = nn.MatrixVectorAdd(graph, sw, self.b1)
        relu1 = nn.ReLU(graph, sw_plus_b)
        sw2 = nn.MatrixMultiply(graph, relu1, self.w2)
        sw2_plus_b2 = nn.MatrixVectorAdd(graph, sw2, self.b2)
        relu2 = nn.ReLU(graph, sw2_plus_b2)
        sw3 = nn.MatrixMultiply(graph, relu2, self.w3)
        sw3_plus_b3 = nn.MatrixVectorAdd(graph, sw3, self.b3)

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            input_q_target = nn.Input(graph, Q_target)
            loss = nn.SquareLoss(graph, sw3_plus_b3, input_q_target)
            return graph

        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(sw3_plus_b3)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .005
        self.d = 200
        self.w1 = nn.Variable(self.num_chars, self.d)
        self.b1 = nn.Variable(self.d, 200)
        # self.w5 = nn.Varianle(self.d, 200)
        self.w2 = nn.Variable(self.d, 200)
        self.b2 = nn.Variable(200)
        self.w3 = nn.Variable(200, 200)
        self.b3 = nn.Variable(200)
        self.w4 = nn.Variable(200, 5)
        self.b4 = nn.Variable(5)



    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).
-=
        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4])
        hi = nn.Input(graph, np.zeros((batch_size, self.d)))
        for x in xs:
            input_x = nn.Input(graph, x)
            xw = nn.MatrixMultiply(graph, input_x, self.w1)
            xw_mul_b = nn.MatrixMultiply(graph, hi, self.b1)
            hi = nn.Add(graph, xw, xw_mul_b)
            # xw_plus_hi = nn.Add(graph, xw, hi)
            # xw_plus_b = nn.MatrixVectorAdd(graph, xw, self.b1)
            # relu = nn.ReLU(graph, xw_plus_b)
            # hi = nn.Add(graph, hi, relu)
            hiw2 = nn.MatrixMultiply(graph, hi, self.w2)
            hiw2_plus_b2 = nn.MatrixVectorAdd(graph, hiw2, self.b2)
            hi = nn.ReLU(graph, hiw2_plus_b2)
        ##layer2

        ##layer3
        hiw3 = nn.MatrixMultiply(graph, hi, self.w3)
        hiw3_plus_b3 = nn.MatrixVectorAdd(graph, hiw3, self.b3)
        relu3 = nn.ReLU(graph, hiw3_plus_b3)
        ##layer4
        hiw4 = nn.MatrixMultiply(graph, relu3, self.w4)
        hiw4_plus_b4 = nn.MatrixVectorAdd(graph, hiw4, self.b4)

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, hiw4_plus_b4, input_y)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(hiw4_plus_b4)
