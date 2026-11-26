package ch.noseryoung.activationFunctions

interface ActivationFunction {
    fun activation(input: Double): Double
    fun derivative(input: Double): Double
}