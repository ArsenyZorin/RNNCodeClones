package recurrentNeuralNetwork.training;

import additionalClasses.Matrix;

import java.lang.*;

/**
 * Created by arseny on 01.04.17.
 */
public class Loss {

    public void backward(Matrix output, Matrix target){
        for(int i = 0; i < target.getArrSize(); i++){
            double errDelta = output.getArrayElem(i) - target.getArrayElem(i);
            output.setBackArrayElem(i, output.getArrayElem(i) + errDelta);
        }
    }

    public double measure(Matrix output, Matrix target){
        double sum = 0.0;
        for (int i = 0; i < target.getArrSize(); i++){
            double errDelta = output.getArrayElem(i) - target.getArrayElem(i);
            sum += 0.5 * Math.pow(errDelta, 2.0);
        }
        return sum;
    }
}
