package recurrentNeuralNetwork.training;

import additionalClasses.Matrix;
import data.Sequence;
import data.Step;
import recurrentNeuralNetwork.lstmNetwork.Network;

import java.util.ArrayList;

/**
 * Created by arseny on 31.03.17.
 */
public class Trainer {
    public static double train(int epochs, double rate, Network network){
        Double res = 0.0;
        Double reportedLossValidation = 0.0;
        Double reportedLossTesting = 0.0;
        for (int epoch = 0; epoch < epochs; epoch++){

            res = methodName();
            if(res.isNaN() || res.isInfinite())
                throw new ArithmeticException("Exception: invalid value");

        }

        return res;
    }

    public static double methodName(double rate, Network network,
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

            ArrayList<Sequence> thisSeq = new ArrayList<Sequence>();
            thisSeq.add(seq);
            // Updation LSTM values
            if (apply){

            }
        }

        return numerator / denomenator;
    }
}
