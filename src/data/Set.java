package data;

import recurrentNeuralNetwork.training.Loss;

import java.util.ArrayList;

/**
 * Created by arseny on 31.03.17.
 */
public class Set {
    private int inputDim;
    private int outputDim;
    private Loss lossTraining;
    private Loss lossReporting;

    public Loss getLossTraining() {
        return lossTraining;
    }

    public void setLossTraining(Loss lossTraining) {
        this.lossTraining = lossTraining;
    }

    public Loss getLossReporting() {
        return lossReporting;
    }

    public void setLossReporting(Loss lossReporting) {
        this.lossReporting = lossReporting;
    }

    private ArrayList<Sequence> training;
    private ArrayList<Sequence> validation;
    private ArrayList<Sequence> testing;

    public int getInputDim() {
        return inputDim;
    }

    public void setInputDim(int inputDim) {
        this.inputDim = inputDim;
    }

    public int getOutputDim() {
        return outputDim;
    }

    public void setOutputDim(int outputDim) {
        this.outputDim = outputDim;
    }

    public ArrayList<Sequence> getTraining() {
        return training;
    }

    public void setTraining(ArrayList<Sequence> training) {
        this.training = training;
    }

    public ArrayList<Sequence> getValidation() {
        return validation;
    }

    public void setValidation(ArrayList<Sequence> validation) {
        this.validation = validation;
    }

    public ArrayList<Sequence> getTesting() {
        return testing;
    }

    public void setTesting(ArrayList<Sequence> testing) {
        this.testing = testing;
    }
}
