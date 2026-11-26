package ch.noseryoung

import ch.noseryoung.activationFunctions.ActivationFunction

class Neuron(val activationFunction: ActivationFunction){
    private var bias = 0.0

    fun neuroning(inputs : List<Double>, weights : List<Double>): Double {
        require(inputs.size == weights.size)
        val zip = inputs.zip(weights)
        return activationFunction.activation(zip.map{it.first * it.second}.sum() + bias)
    }
}