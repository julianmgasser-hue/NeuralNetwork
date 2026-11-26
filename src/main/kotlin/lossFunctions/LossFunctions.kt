package ch.noseryoung.lossFunctions

import kotlin.math.pow

class MeanSquaredError(): LossFunction{
    override fun loss(
        outputs: List<Double>,
        targets: List<Double>
    ): Double {
        require(outputs.size == targets.size)
        require(outputs.isNotEmpty())
        val zip = outputs.zip(targets)
        val n = outputs.size //same amount of outputs as neurons
        val result = zip.map{(it.first - it.second).pow(2)}.sum()/n
        return result
    }

    override fun derivative(
        outputs: List<Double>,
        targets: List<Double>
    ): List<Double> {
        require(outputs.size == targets.size)
        require(outputs.isNotEmpty())

        val zip = outputs.zip(targets)
        val n = outputs.size

        return zip.map { (2.0 / n) * (it.first - it.second) }
    }

}