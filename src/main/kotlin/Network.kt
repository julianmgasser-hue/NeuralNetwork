package ch.noseryoung

import ch.noseryoung.lossFunctions.LossFunction
import kotlin.random.Random

class Network(val layers: List<Layer>) {
    val inputLayer = layers.first()
    private var forwardPassCompleted = false

    fun calculateNetwork(inputs : List<Double>):List<Double> {
        forwardPassCompleted = false
        require(inputs.size == inputLayer.countNeurons())
        var outputs = inputs
        layers.forEachIndexed{ index, layer ->
            if (index == 0) return@forEachIndexed
            outputs = layer.calculateLayer(outputs)
        }
        forwardPassCompleted = true
        return outputs
    }

    fun backwardPropagation(
        targets: List<Double>,
        lossFunction: LossFunction
    ) {
        require(layers.size >= 2)

        check(forwardPassCompleted) {
            "Forward propagation must be performed first"
        }

        val outputLayer = layers.last()

        val outputs = outputLayer.neurons.map {
            it.activatedOutput
        }

        require(outputs.size == targets.size)

        // Calculate loss gradients
        val lossGradients = lossFunction.derivative(outputs, targets)

        // Calculate output layer deltas
        outputLayer.calculateOutputDeltas(lossGradients)

        // Calculate hidden layer deltas
        for (index in layers.lastIndex - 1 downTo 1) {

            val currentLayer = layers[index]
            val nextLayer = layers[index + 1]

            currentLayer.calculateHiddenDeltas(nextLayer)
        }
    }

    fun trainStep(
        inputs: List<Double>,
        targets: List<Double>,
        lossFunction: LossFunction,
        learningRate: Double
    ): Double {
        require(learningRate.isFinite() && learningRate > 0.0)

        val outputs = calculateNetwork(inputs)

        val loss = lossFunction.loss(outputs, targets)

        backwardPropagation(targets, lossFunction)

        forwardPassCompleted = false

        layers.drop(1).forEach { layer ->
            layer.applyGradients(learningRate)
        }

        return loss
    }

    fun train(
        dataset: List<TrainingSample>,
        epochs: Int,
        lossFunction: LossFunction,
        learningRate: Double,
        random: Random = Random.Default
    ): List<Double> {
        require(dataset.isNotEmpty())
        require(epochs > 0)
        require(learningRate.isFinite() && learningRate > 0.0)
        require(layers.size >= 2)

        require(dataset.all {
            it.inputs.size == inputLayer.countNeurons() &&
                    it.targets.size == layers.last().countNeurons()
        })

        val lossHistory = mutableListOf<Double>()

        repeat(epochs) {
            var totalLoss = 0.0

            for (sample in dataset.shuffled(random)) {
                totalLoss += trainStep(
                    sample.inputs,
                    sample.targets,
                    lossFunction,
                    learningRate
                )
            }

            lossHistory.add(totalLoss / dataset.size)
        }

        return lossHistory
    }
}