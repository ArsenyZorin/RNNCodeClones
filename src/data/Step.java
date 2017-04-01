package data;

import additionalClasses.Matrix;

/**
 * Created by arseny on 31.03.17.
 */
public class Step {
    private Matrix input = null;
    private Matrix targetOutput = null;

    public Step(){}

    public Step(double [] input, double [] targetOutput){
        this.input = new Matrix(input);
        if(targetOutput != null)
            this.targetOutput = new Matrix(targetOutput);
    }

    /*
    @Override
    public String toString(){

    }
    */

    public Matrix getInput() {
        return input;
    }

    public void setInput(Matrix input) {
        this.input = input;
    }

    public Matrix getTargetOutput() {
        return targetOutput;
    }

    public void setTargetOutput(Matrix targetOutput) {
        this.targetOutput = targetOutput;
    }
}
