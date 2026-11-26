package ch.noseryoung

import ch.noseryoung.activationFunctions.ActivationFunction

class Neuron(val activationFunction: ActivationFunction){
    var bias = 0.0
    var weightedSum = 0.0
        private set
    var activatedOutput = 0.0
        private set
    var delta = 0.0
        private set

    fun neuroning(inputs : List<Double>, weights : List<Double>): Double {
        require(inputs.size == weights.size)
        val zip = inputs.zip(weights)
        weightedSum = zip.map{it.first * it.second}.sum()+ bias
        activatedOutput = activationFunction.activation(weightedSum)
        return activatedOutput
    }
    fun calculateDelta(outputGradient: Double) {
        delta = outputGradient *
                activationFunction.derivative(weightedSum)
    }
}