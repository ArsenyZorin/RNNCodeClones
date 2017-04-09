package recurrentNeuralNetwork.training;

import additionalClasses.Matrix;
import data.Sequence;
import data.Set;
import data.Step;
import recurrentNeuralNetwork.lstmNetwork.Network;

import java.util.ArrayList;

/**
 * Created by arseny on 31.03.17.
 */
public class Trainer {

    private static double decRate = 0.999;
    private static int gradClipVal = 5;
    private static double smoothEps = 1E-8;
    private static double regul = 1E-6;

    public static double train(int epochs, double rate, Network network, Set data){
        Double res = 0.0;
        for (int epoch = 0; epoch < epochs; epoch++){

            res = pass(rate, network, data.getTraining(),
                    data.getLossTraining(), data.getLossReporting(), true);
            if(res.isNaN() || res.isInfinite())
                throw new ArithmeticException("Exception: invalid value");

            double repLosValid = 0.0;
            double repLosTest = 0.0;

            if(data.getValidation() != null){
                repLosValid = pass(rate, network, data.getValidation(),
                        data.getLossTraining(), data.getLossReporting(), false);
                res = repLosValid;
            }
            if(data.getTesting() != null){
                repLosTest = pass(rate, network, data.getTesting(),
                        data.getLossTraining(), data.getLossReporting(), false);
                res = repLosTest;
            }

            System.out.println("epoch[" + (epoch + 1) + "/" + epochs + "]");
        }

        return res;
    }

    private static double pass(double rate, Network network,
                                    ArrayList<Sequence> sequences, Loss training,
                                    Loss reporting, boolean apply){

        double numerator = 0.0;
        double denomenator = 1.0;

        for (Sequence seq : sequences){
            network.reset();

            for (Step step : seq.getSteps()){
                Matrix output = network.activate(step.getInput());
                if(step.getTargetOutput() != null){
                    double loss = reporting.measure(output, step.getTargetOutput());
                    if (Double.isNaN(loss) || Double.isInfinite(loss))
                        return loss;

                    numerator += loss;
                    denomenator++;

                    if(apply)
                        training.backward(output, step.getTargetOutput());
                }
            }

            ArrayList<Sequence> thisSeq = new ArrayList<>();
            thisSeq.add(seq);
            // Updation LSTM values
            if (apply)
                update(network, rate);
        }

        return numerator / denomenator;
    }

    private static void update(Network network, double size){
        for (Matrix matrix : network.getParams()){
            for (int i = 0; i < matrix.getArrSize(); i++){
                double matrixBackAr = matrix.getBackArrayElem(i);
                matrix.setCacheElem(i, matrix.getCacheElem(i) * decRate + (1 - decRate) * Math.pow(matrixBackAr, 2.0));

                if (matrixBackAr > 0)
                    matrixBackAr = gradClipVal;

                if (matrixBackAr < - gradClipVal)
                    matrixBackAr = - gradClipVal;

                double elem = - size * matrixBackAr / Math.sqrt(matrix.getCacheElem(i) + smoothEps) - regul * matrix.getArrayElem(i);
                matrix.setArrayElem(i, matrix.getArrayElem(i) + elem);
                matrix.setBackArrayElem(i, 0.0);
            }
        }
    }

}
