package data;

import recurrentNeuralNetwork.training.Loss;

import java.util.ArrayList;

/**
 * Created by arseny on 31.03.17.
 */
public class Set {
    protected int inputDim;
    protected int outputDim;
    protected Loss lossTraining;
    protected Loss lossReporting;
    protected ArrayList<Sequence> training;
    protected ArrayList<Sequence> validation;
    protected ArrayList<Sequence> testing;

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
