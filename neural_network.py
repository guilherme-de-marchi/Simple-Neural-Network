# My Github: https://github.com/Guilherme-De-Marchi

from random import randint
from matrix import Matrix

class NeuralNetwork:
    def __init__(self, quantity_inputs, quantity_hidden, quantity_outputs, learning_rate):
        self.weights_input_hidden = Matrix(quantity_inputs, quantity_hidden)
        self.weights_hidden_output = Matrix(quantity_hidden, quantity_outputs)

        self.bias_input_hidden = Matrix(1, quantity_hidden)
        self.bias_hidden_output = Matrix(1, quantity_outputs)

        self.learning_rate = learning_rate

    def train(self, inputs, correct_outputs):        
        inputs = Matrix(matrix_list=inputs)
        hidden = (inputs * self.weights_input_hidden + self.bias_input_hidden).sigmoid()
        outputs = (hidden * self.weights_hidden_output + self.bias_hidden_output).sigmoid()
        correct_outputs = Matrix(matrix_list=correct_outputs)

        output_errors = correct_outputs - outputs
        derivative_outputs = outputs.derivative_sigmoid()
        outputs_gradient = Matrix.hadamard(derivative_outputs, output_errors) * self.learning_rate
        self.bias_hidden_output += outputs_gradient
        transposed_hidden = hidden.transpose()
        self.weights_hidden_output += transposed_hidden * outputs_gradient

        transposed_weights_hidden_output = self.weights_hidden_output.transpose()
        hidden_errors = output_errors * transposed_weights_hidden_output
        derivative_hidden = hidden.derivative_sigmoid()
        hidden_gradient = Matrix.hadamard(derivative_hidden, hidden_errors) * self.learning_rate
        self.bias_input_hidden += hidden_gradient
        transposed_inputs = inputs.transpose()
        self.weights_input_hidden += transposed_inputs * hidden_gradient

    def predict(self, inputs):
        inputs = Matrix(matrix_list=inputs)
        hidden = (inputs * self.weights_input_hidden + self.bias_input_hidden).sigmoid()
        outputs = (hidden * self.weights_hidden_output + self.bias_hidden_output).sigmoid()

        return outputs

'''
EXAMPLE:

i = [ # Inputs
    [1, 1],
    [0, 0],
    [1, 0],
    [0, 1],
]

o = [ # Correct outputs
    [0],
    [0],
    [1],
    [1]
]

nn = NeuralNetwork(2, 3, 1, 1)

for j in range(10000): # Training 10.000 times
    x = randint(0, 3)
    nn.train([i[x]], [o[x]])

print(nn.predict([[1, 1]]))
print(nn.predict([[0, 0]]))
'''
