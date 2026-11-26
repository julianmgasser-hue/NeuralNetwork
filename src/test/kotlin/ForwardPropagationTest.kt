package ch.noseryoung

import ch.noseryoung.activationFunctions.default
import ch.noseryoung.activationFunctions.reLU
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ForwardPropagationTest {
    private val tolerance = 1e-9

    private data class Fixture(
        val input: Layer,
        val hidden: Layer,
        val output: Layer,
        val network: Network
    )

    // Fully connected 2 -> 3 -> 2 network with fixed weights and zero biases.
    private fun fixture(): Fixture {
        val input = Layer(List(2) { Neuron(default()) }, null)
        val hidden = Layer(List(3) { Neuron(reLU()) }, input)
        val output = Layer(List(2) { Neuron(default()) }, hidden)

        hidden.weights[0][0] = 0.5
        hidden.weights[0][1] = -1.0
        hidden.weights[1][0] = -1.0
        hidden.weights[1][1] = 1.0
        hidden.weights[2][0] = 0.25
        hidden.weights[2][1] = -0.5

        output.weights[0][0] = 0.5
        output.weights[0][1] = 2.0
        output.weights[0][2] = -1.0
        output.weights[1][0] = -1.0
        output.weights[1][1] = 0.5
        output.weights[1][2] = 1.0

        return Fixture(input, hidden, output, Network(listOf(input, hidden, output)))
    }

    private fun assertVectorEquals(expected: List<Double>, actual: List<Double>) {
        assertEquals(expected.size, actual.size, "Output size")
        expected.forEachIndexed { index, value ->
            assertEquals(value, actual[index], tolerance, "Output at index $index")
        }
    }

    @Test
    fun activationFunctions() {
        assertEquals(-3.0, default().activation(-3.0), tolerance)
        assertEquals(3.0, default().activation(3.0), tolerance)
        assertEquals(-0.3, reLU().activation(-3.0), tolerance)
        assertEquals(0.0, reLU().activation(0.0), tolerance)
        assertEquals(3.0, reLU().activation(3.0), tolerance)
    }

    @Test
    fun neuronCalculatesWeightedSum() {
        val neuron = Neuron(default())
        val result = neuron.neuroning(listOf(2.0, 3.0), listOf(0.5, -2.0))
        assertEquals(-5.0, result, tolerance)
    }

    @Test
    fun neuronRejectsWrongNumberOfWeights() {
        assertFailsWith<IllegalArgumentException> {
            Neuron(default()).neuroning(listOf(1.0, 2.0), listOf(0.5))
        }
    }

    @Test
    fun weightMatrixHasCorrectDimensionsAndInitialRange() {
        val input = Layer(List(2) { Neuron(default()) }, null)
        val hidden = Layer(List(3) { Neuron(reLU()) }, input)
        val output = Layer(List(2) { Neuron(default()) }, hidden)

        assertEquals(listOf(0, 0), input.weights.map { it.size })
        assertEquals(listOf(2, 2, 2), hidden.weights.map { it.size })
        assertEquals(listOf(3, 3), output.weights.map { it.size })
        assertTrue((hidden.weights.flatten() + output.weights.flatten()).all {
            it >= -0.1 && it < 0.1
        })

        val unchanged = hidden.weights[1][0]
        hidden.weights[0][0] = 0.5
        assertEquals(unchanged, hidden.weights[1][0])
    }

    @Test
    fun hiddenLayerCalculatesAllNeurons() {
        val f = fixture()
        assertVectorEquals(listOf(2.0, -0.3, 1.0),
            f.hidden.calculateLayer(listOf(2.0, -1.0)))
    }

    @Test
    fun forwardPropagationReturnsExpectedOutputs() {
        val f = fixture()
        assertVectorEquals(listOf(-0.6, -1.15),
            f.network.calculateNetwork(listOf(2.0, -1.0)))
    }

    @Test
    fun repeatedForwardPassDoesNotChangeWeightsOrOutputs() {
        val f = fixture()
        val inputs = listOf(2.0, -1.0)
        val layers = listOf(f.hidden, f.output)
        val weightsBefore = layers.map { layer -> layer.weights.map { it.toList() } }
        val first = f.network.calculateNetwork(inputs)
        val second = f.network.calculateNetwork(inputs)

        assertEquals(first, second)
        assertEquals(weightsBefore, layers.map { layer -> layer.weights.map { it.toList() } })
    }

    @Test
    fun changedWeightAffectsNextForwardPass() {
        val f = fixture()
        val inputs = listOf(2.0, -1.0)
        assertVectorEquals(listOf(-0.6, -1.15), f.network.calculateNetwork(inputs))
        f.output.weights[0][0] = 1.0
        assertVectorEquals(listOf(0.4, -1.15), f.network.calculateNetwork(inputs))
    }

    @Test
    fun networkRejectsWrongInputSize() {
        val f = fixture()
        assertFailsWith<IllegalArgumentException> {
            f.network.calculateNetwork(listOf(2.0))
        }
    }
}
