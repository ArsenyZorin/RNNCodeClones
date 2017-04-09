package recurrentNeuralNetwork.lstmNetwork;

import additionalClasses.Matrix;
import java.util.ArrayList;

/**
 * Created by arseny on 23.03.17.
 */
public class Network {

    private ArrayList<Layer> layers;
    private Matrix input;
/*
    public Network(Matrix input, ArrayList<Layer> layers){
        this.input = input;
        this.layers = layers;
    }*/

    public Network(ArrayList<Layer> layers){
        this.layers = layers;
    }

    public Matrix getInput() {
        return input;
    }

    public void setInput(Matrix input) {
        this.input = input;
    }

    public Network makeLSTM(int inputAmount, int hiddenAmount, int hiddenLayers, int outputAmount){
        layers = new ArrayList<>();
        for (int i = 0; i < hiddenLayers; i++){
            if (i == 0)
                layers.add(new Layer(/*this.input,*/ inputAmount, hiddenAmount/*, outputAmount*/));
            else
                layers.add(new Layer(/*this.input, */hiddenAmount, hiddenAmount/*, outputAmount*/));
        }
        return new Network(layers);
    }

    public Matrix activate(Matrix input){
        Matrix prev = input;
        for (Layer layer : this.layers)
            prev = layer.activate(prev);
        return prev;
    }

    public void reset(){
        for(Layer layer : layers)
            layer.reset();
    }

    public ArrayList<Matrix> getParams(){
        ArrayList<Matrix> res = new ArrayList<>();
        for (Layer layer : layers)
            res.addAll(layer.getParams());
        return res;
    }

}
