package ch.noseryoung

data class Layer(val neurons : List<Neuron>){
    fun countNeurons() = neurons.size
}

