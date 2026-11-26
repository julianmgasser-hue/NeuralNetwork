package ch.noseryoung.activationFunctions

import ch.noseryoung.ActivationFunctions.ActivationFunction

public class reLU : ActivationFunction{
    override fun activation(input: Double) =  if (input <= 0) 0.1 * input else input
}
public class default : ActivationFunction{
    override fun activation(input: Double) = input
}
