/**
 * Created by zorin on 14.03.2017.
 */
package recurrentNeuralNetwork.elmanNetwork;
public class RecurrentNetwork {


    /* Numbers of input, hidden and output nodes */
    /*private int numInput;
    private int numHidden;
    private int numOutput;

    /* Values of input, hidden, context and output nodes */
    /*private double [] inputNodes;
    private double [] hiddenNodes;
    private double [] contextNodes;
    private double [] outputNodes;

    /* Weights of input-hidden, context-hidden and hidden-output */
    /*private double [][] inpHidWeights;
    private double [][] contHidWeights;
    private double [][] hidOutWeights;

    private Random rnd;

    //How to represent hidden-hidden layers

    public RecurrentNetwork(int numInput, int numHidden, int numOutput, int seed) {
        this.numInput = numInput;
        this.numHidden = numHidden;
        this.numOutput = numOutput;

        this.inputNodes = new double[numInput];
        this.hiddenNodes = new double[numHidden];
        this.outputNodes = new double[numOutput];

        this.inpHidWeights = makeMatr(numInput, numHidden);
        this.contHidWeights = makeMatr(numHidden, numHidden);
        this.hidOutWeights = makeMatr(numHidden, numOutput);

        this.rnd = new Random(seed);

    }

    private static double[][] makeMatr(int rows, int col){
        double[][] newArr = new double[rows][col];
        for (int i = 0; i < rows; i++)
            newArr[i] = new double[col];

        return newArr;
    }*/

    /* Amount of the network inputs */
    private int inputsCount;

    /* Amount of the network layers */
    private int layersCount;

    /* Layers of the network */
    private Layer[] layers;

    /* Output vector of the network */
    private double[] output;

    public RecurrentNetwork(int inputsCount, int layersCount){
        this.inputsCount = inputsCount;
        this.layersCount = layersCount;

        this.layers = new Layer[this.layersCount];
    }

    public double[] compute(double[] input){
        for(Layer layer : layers)
            output = layer.compute(input);
        return output;
    }
}
