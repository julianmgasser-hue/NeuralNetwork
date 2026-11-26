package ch.noseryoung

import ch.noseryoung.ActivationFunctions.ActivationFunction

class Neuron(val activationFunction: ActivationFunction){
    private var bias = 0.0
    private var inputs = listOf<NeuralInput>()

    fun neuroning(newInputs : List<NeuralInput>): Double {
        inputs = newInputs // for backpropagation
        return activationFunction.activation(inputs.map{it.value * it.weight}.sum() + bias)
    }
}