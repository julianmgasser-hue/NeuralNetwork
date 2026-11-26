package ch.noseryoung

import ch.noseryoung.activationFunctions.default
import ch.noseryoung.activationFunctions.reLU
import ch.noseryoung.lossFunctions.MeanSquaredError
import kotlin.random.Random

fun main() {
    val inputLayer = Layer(
        List(2) { Neuron(default()) },
        null
    )

    val hiddenLayer = Layer(
        List(4) { Neuron(reLU()) },
        inputLayer
    )

    val outputLayer = Layer(
        List(1) { Neuron(default()) },
        hiddenLayer
    )

    val network = Network(
        listOf(inputLayer, hiddenLayer, outputLayer)
    )

    // Reproducible weight initialization
    val random = Random(42)

    for (layer in network.layers.drop(1)) {
        for (row in layer.weights) {
            for (index in row.indices) {
                row[index] = random.nextDouble(-1.0, 1.0)
            }
        }
    }

    // XOR dataset
    val dataset = listOf(
        TrainingSample(listOf(0.0, 0.0), listOf(0.0)),
        TrainingSample(listOf(0.0, 1.0), listOf(1.0)),
        TrainingSample(listOf(1.0, 0.0), listOf(1.0)),
        TrainingSample(listOf(1.0, 1.0), listOf(0.0))
    )

    val lossFunction = MeanSquaredError()

    val history = network.train(
        dataset = dataset,
        epochs = 5000,
        lossFunction = lossFunction,
        learningRate = 0.05,
        random = Random(42)
    )

    println("First epoch: ${history.first()}")
    println("Last epoch: ${history.last()}")

    // Evaluate predictions
    for (sample in dataset) {
        val output = network
            .calculateNetwork(sample.inputs)
            .single()

        val prediction = if (output >= 0.5) 1 else 0

        println(
            "${sample.inputs} -> " +
                    "raw: $output, " +
                    "prediction: $prediction, " +
                    "target: ${sample.targets.single()}"
        )
    }

    val finalLoss = dataset.map { sample ->
        lossFunction.loss(
            network.calculateNetwork(sample.inputs),
            sample.targets
        )
    }.average()

    println("Final loss: $finalLoss")

    history.forEachIndexed { index, loss ->
        if ((index + 1) % 500 == 0) {
            println("Epoch ${index + 1}: $loss")
        }
    }
}