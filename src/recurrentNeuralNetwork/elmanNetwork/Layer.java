package recurrentNeuralNetwork.elmanNetwork;

import activationFunctions.*;

/**
 * Created by zorin on 15.03.2017.
 */
public class Layer {

    /* Amount of the layer inputs */
    private int inputsCount;

    /* Amount of neurons in the layer */
    private int neuronsCount;

    /* Neurons of the network layer */
    private Neuron[] neurons;

    /* Context neurons of the network layer */
    private Neuron[] contextNeurons;

    /* Output vector of the layer */
    private double[] output;

    public Layer(int inputsCount, int neuronsCount, SigmoidFunction function) {
        this.inputsCount = inputsCount;
        this.neuronsCount = neuronsCount;
        this.neurons = new Neuron[this.neuronsCount];
        this.contextNeurons = new Neuron[this.neuronsCount];
        this.output = new double[this.neuronsCount];

        for (int i = 0; i < this.neuronsCount; i++) {
            this.neurons[i] = new Neuron(this.inputsCount, function);
            this.contextNeurons[i] = new Neuron(this.inputsCount, function);
        }
    }

    public double[] compute(double[] input){
        for(int i = 0; i < neuronsCount; i++)
            output[i] = neurons[i].compute(input) + contextNeurons[i].computeContext(i);

        for(int i = 0; i < neuronsCount; i++)
            contextNeurons[i].output=neurons[i].output;

        return output;
    }
}
