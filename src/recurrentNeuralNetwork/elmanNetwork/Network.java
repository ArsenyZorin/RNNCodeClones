/**
 * Created by zorin on 14.03.2017.
 */
package recurrentNeuralNetwork.elmanNetwork;
public class Network {

    /* Amount of the network inputs */
    private int inputsCount;

    /* Amount of the network layers */
    private int layersCount;

    /* Layers of the network */
    private Layer[] layers;

    /* Output vector of the network */
    private double[] output;

    public Network(int inputsCount, int layersCount){
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
