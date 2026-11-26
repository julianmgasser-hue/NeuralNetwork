package ch.noseryoung.activationFunctions

public class reLU : ActivationFunction {
    override fun activation(input: Double) =  if (input <= 0) 0.1 * input else input
    override fun derivative(input: Double): Double {
        return if (input == 0.0) 1.0 else activation(input) / input
    }
}
public class default : ActivationFunction{
    override fun activation(input: Double) = input
    override fun derivative(input: Double): Double {
        return 1.0
    }
}
