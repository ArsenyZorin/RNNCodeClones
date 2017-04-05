package testing;

import data.*;
import recurrentNeuralNetwork.training.Loss;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by arseny on 05.04.17.
 */
public class XORdata extends Set {
    public XORdata() {
        this.inputDim = 2;
        this.outputDim = 1;
        this.lossTraining = new Loss();
        this.lossReporting = new Loss();
        this.training = getTrainingData();
        this.validation = getTrainingData();
        this.testing = getTrainingData();
    }

    private static ArrayList<Sequence> getTrainingData(){
        ArrayList<Sequence> res = new ArrayList<>();

        ArrayList<Step> stepList = new ArrayList<>();
        stepList.add(new Step(new double[] { 1, 0 }, new double[] { 1 }));
        res.add(new Sequence(stepList));
        stepList = new ArrayList<>();
        stepList.add(new Step(new double[] { 0, 1 }, new double[] { 1 }));
        res.add(new Sequence(stepList));
        stepList = new ArrayList<>();
        stepList.add(new Step(new double[] { 0, 0 }, new double[] { 0 }));
        res.add(new Sequence(stepList));
        stepList = new ArrayList<>();
        stepList.add(new Step(new double[] { 1, 1 }, new double[] { 0 }));
        res.add(new Sequence(stepList));

        return res;
    }

}
