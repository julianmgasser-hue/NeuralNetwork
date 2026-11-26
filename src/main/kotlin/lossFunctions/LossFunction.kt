package ch.noseryoung.lossFunctions

interface LossFunction {
    fun loss(outputs: List<Double>, targets: List<Double>): Double
    fun derivative(outputs: List<Double>, targets: List<Double>): List<Double>
}