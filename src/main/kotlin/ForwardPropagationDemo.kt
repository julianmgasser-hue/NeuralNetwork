package ch.noseryoung

import ch.noseryoung.activationFunctions.default
import ch.noseryoung.activationFunctions.reLU

fun main() {
    val inputLayer = Layer(List(2) { Neuron(default()) }, null)
    val hiddenLayer = Layer(List(3) { Neuron(reLU()) }, inputLayer)
    val outputLayer = Layer(List(2) { Neuron(default()) }, hiddenLayer)

    val hiddenWeights = listOf(
        listOf(0.5, -1.0),
        listOf(-1.0, 1.0),
        listOf(0.25, -0.5)
    )
    val outputWeights = listOf(
        listOf(0.5, 2.0, -1.0),
        listOf(-1.0, 0.5, 1.0)
    )
    hiddenWeights.forEachIndexed { i, row ->
        row.forEachIndexed { j, weight -> hiddenLayer.weights[i][j] = weight }
    }
    outputWeights.forEachIndexed { i, row ->
        row.forEachIndexed { j, weight -> outputLayer.weights[i][j] = weight }
    }

    val network = Network(listOf(inputLayer, hiddenLayer, outputLayer))
    val inputs = listOf(2.0, -1.0)
    val outputs = network.calculateNetwork(inputs)

    println("Inputs: $inputs")
    println("Hidden outputs: ${hiddenLayer.calculateLayer(inputs)}")
    println("Network outputs: $outputs")
    println("Expected outputs: [-0.6, -1.15]")

    outputLayer.weights[0][0] = 1.0
    println("After weight update: ${network.calculateNetwork(inputs)}")
    println("Expected after update: [0.4, -1.15]")
}
