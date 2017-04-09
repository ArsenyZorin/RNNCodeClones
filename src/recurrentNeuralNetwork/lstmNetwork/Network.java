package recurrentNeuralNetwork.lstmNetwork;

import additionalClasses.Matrix;
import java.util.ArrayList;

/**
 * Created by arseny on 23.03.17.
 */
public class Network {

    private ArrayList<Layer> layers;
    private Matrix input;

    public Network(ArrayList<Layer> layers){
        this.layers = layers;
    }

    public Matrix getInput() {
        return input;
    }

    public void setInput(Matrix input) {
        this.input = input;
    }

    /**
     * LSTM network creation
     * @param inputDimension Dimension of input vector
     * @param hiddenDimension Dimension of hidden layer
     * @param hiddenLayers Amount of hidden layers
     * @return LSTM network
     */
    public Network makeLSTM(int inputDimension, int hiddenDimension, int hiddenLayers){
        layers = new ArrayList<>();
        for (int i = 0; i < hiddenLayers; i++){
            if (i == 0)
                layers.add(new Layer(inputDimension, hiddenDimension));
            else
                layers.add(new Layer(hiddenDimension, hiddenDimension));
        }
        return new Network(layers);
    }

    /**
     * LSTM network activation
     * @param input
     * @return
     */
    public Matrix activate(Matrix input){
        Matrix prev = input;
        for (Layer layer : this.layers)
            prev = layer.activate(prev);
        return prev;
    }

    /**
     *
     */
    public void reset(){
        for(Layer layer : layers)
            layer.reset();
    }

    /**
     *
     * @return
     */
    public ArrayList<Matrix> getParams(){
        ArrayList<Matrix> res = new ArrayList<>();
        for (Layer layer : layers)
            res.addAll(layer.getParams());
        return res;
    }

}
