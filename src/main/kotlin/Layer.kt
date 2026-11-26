package ch.noseryoung

import kotlin.random.Random

class Layer(val neurons : List<Neuron>, val previousLayer: Layer?){
    fun countNeurons() = neurons.size
    val inputNum = previousLayer?.countNeurons() ?: 0
    var savedInputs = listOf<Double>()
        private set

    val weights: List<MutableList<Double>> = List(countNeurons()) { // [neuron index] [inputIndex]
        MutableList(inputNum) {
            Random.nextDouble(-0.1, 0.1)
        }
    }

    fun calculateLayer(inputs : List<Double>): List<Double> {
        savedInputs = inputs.toList()
        return neurons.mapIndexed { index, neuron ->
            neuron.neuroning(inputs, weights[index])
        }
    }

    fun calculateOutputDeltas(lossGradients: List<Double>) {
        require(lossGradients.size == countNeurons())

        neurons.forEachIndexed { index, neuron ->
            neuron.calculateDelta(lossGradients[index])
        }
    }

    fun calculateHiddenDeltas(nextLayer: Layer) {
        require(nextLayer.previousLayer === this)

        neurons.forEachIndexed { neuronIndex, neuron ->

            val outputGradient = nextLayer.neurons
                .mapIndexed { nextIndex, nextNeuron ->

                    val weight = nextLayer.weights[nextIndex][neuronIndex]

                    weight * nextNeuron.delta
                }.sum()

            neuron.calculateDelta(outputGradient)
        }
    }

    fun calculateWeightGradients(): List<List<Double>> {
        require(savedInputs.size == inputNum)

        return neurons.map { neuron ->
            savedInputs.map { input ->
                neuron.delta * input
            }
        }
    }

    fun calculateBiasGradients() = neurons.map { it.delta }

    fun applyGradients(learningRate: Double) {
        require(learningRate.isFinite() && learningRate > 0.0)
        require(previousLayer != null)

        val weightGradients = calculateWeightGradients()
        val biasGradients = calculateBiasGradients()

        neurons.forEachIndexed { neuronIndex, neuron ->

            weights[neuronIndex].indices.forEach { inputIndex ->

                weights[neuronIndex][inputIndex] -=
                    learningRate * weightGradients[neuronIndex][inputIndex]
            }

            neuron.bias -= learningRate * biasGradients[neuronIndex]
        }
    }
}

