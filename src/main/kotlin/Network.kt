package ch.noseryoung

class Network(val layers: List<Layer>) {
    val inputLayer = layers.first()

    fun calculateNetwork(inputs : List<Double>):List<Double> {
        require(inputs.size == inputLayer.countNeurons())
        var outputs = inputs
        layers.forEachIndexed{ index, layer ->
            if (index == 0) return@forEachIndexed
            outputs = layer.calculateLayer(outputs)
        }
        return outputs
    }
}