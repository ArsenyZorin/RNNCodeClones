package testing;
import additionalClasses.Matrix;
import data.Set;
import recurrentNeuralNetwork.lstmNetwork.*;
import recurrentNeuralNetwork.training.Trainer;

import java.util.ArrayList;

/**
 * Created by arseny on 23.03.17.
 */
public class Test {
    public static void main(String[] args) {

        Set data = new XORdata();
        ArrayList<Layer> layers = new ArrayList<Layer>();
        //Matrix input = new Matrix(inp);
        Network lstm = new Network(/*input, */layers);

        lstm = lstm.makeLSTM(2,3, 1, 1);

        int epoch = 100;
        double learningRate = 0.001;

        Trainer.train(epoch, learningRate, lstm, data);

        System.out.println("Training completed");
        System.out.println("Test: 1,1");
        Matrix input = new Matrix(new double[] {1,1});
        Matrix output = lstm.activate(input);
        System.out.println("Test: 1, 1. Output: " + output.getArrayElem(0));

        System.out.println("Test: 0, 1");
        input = new Matrix(new double[] {0,1});
        output = lstm.activate(input);
        System.out.println("Test: 1, 1. Output: " + output.getArrayElem(0));

        System.out.println("DONE");
    }

}
