package ch.noseryoung

import kotlin.random.Random

data class NeuralInput (var value : Double, var weight : Double = Random.nextDouble(1.0, 2.0))