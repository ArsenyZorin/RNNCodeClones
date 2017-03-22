/**
 * Created by zorin on 15.03.2017.
 */
package recurrentNeuralNetwork.elmanNetwork;
import activationFunctions.*;

public class Neuron {

    /* Amount of neuron inputs */
    protected int inputCount;

    /* Neuron weights */
    protected double [] weights;

    /* Neuron output value */
    public double output;

    /* Activation function */
    protected SigmoidFunction func;

    /* Neuron class constructor */
    public Neuron(int inputCount, SigmoidFunction func){
        this.inputCount = inputCount;
        this.weights = new double[this.inputCount];
        this.func = func;
        randomize();
    }

    /* Randomize neuron */
    protected void randomize(){
        for (int i = 0; i < this.inputCount; i ++)
            this.weights[i] = Math.random();
        this.output = Math.random();
    }

    /* Computes output value of neuron */
    public double compute(double[] input){
        double sum = 0.0;

        for(int i = 0; i < this.inputCount; i++)
            sum += this.weights[i] * input[i];
        this.output = this.func.function(sum);
        return this.output;
    }

    public double computeContext(int neuronNumber){
        return (this.weights[neuronNumber] * output);
    }
}
