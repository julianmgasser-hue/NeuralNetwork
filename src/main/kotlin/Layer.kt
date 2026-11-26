package ch.noseryoung

import kotlin.random.Random

class Layer(val neurons : List<Neuron>, val previousLayer: Layer?){
    fun countNeurons() = neurons.size
    val inputNum = previousLayer?.countNeurons() ?: 0

    val weights: List<MutableList<Double>> = List(countNeurons()) { // [neuron index] [inputIndex]
        MutableList(inputNum) {
            Random.nextDouble(-0.1, 0.1)
        }
    }

    fun calculateLayer(inputs : List<Double>): List<Double> {
        return neurons.mapIndexed { index, neuron ->
            neuron.neuroning(inputs, weights[index])
        }
    }
}

