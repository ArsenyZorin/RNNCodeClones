package testing;
import additionalClasses.Matrix;
import recurrentNeuralNetwork.lstmNetwork.*;

import java.util.ArrayList;

/**
 * Created by arseny on 23.03.17.
 */
public class Test {
    public static void main(String[] args) {

        double []inp = {0,0,0};
        ArrayList<Layer> layers = new ArrayList<Layer>();
        Matrix input = new Matrix(inp);
        Network lstm = new Network(input, layers);

        lstm = lstm.makeLSTM(3, 3, 3);


    }

}
