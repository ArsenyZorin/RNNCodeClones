package recurrentNeuralNetwork.lstmNetwork;
import activationFunctions.*;
import additionalClasses.*;
import sun.text.resources.ro.FormatData_ro;

import java.util.ArrayList;

/**
 * Created by arseny on 21.03.17.
 */
public class Gate {

    /* Value of previous cell */
    private Matrix prevHiddenValue;

    /* Weight of previous - current vector */
    private Matrix prevHiddenWeight;

    /* Input value of the current cell */
    private Matrix inputValue;

    /* Weight of the input - current vector*/
    private Matrix inputWeight;

    private int inputAmount;
    private int outputAmount;

    public Gate(Matrix inputWeight, Matrix prevHiddenWeight, int inputAmount, int outputAmount) {

        this.prevHiddenWeight = prevHiddenWeight;
        this.inputWeight = inputWeight;

        this.inputAmount = inputAmount;
        this.outputAmount = outputAmount;

        this.prevHiddenValue = Matrix.random(this.outputAmount, this.outputAmount);
        this.inputValue = Matrix.random(this.inputAmount, this.outputAmount);
    }

    /**
     * Gets value of gate
     * @param func Activation function
     * @return Value of gate
     */
    public Matrix getGateValue(IActivationFunction func){
        Matrix prod1 = this.prevHiddenValue.multiply(this.prevHiddenWeight);
        Matrix prod2 = this.inputValue.multiply(this.inputWeight);

        return (prod1.add(prod2)).applyActFunc(func);
    }

    public ArrayList<Matrix> getParams(){
        ArrayList<Matrix> res = new ArrayList<>();
        res.add(this.prevHiddenValue);
        res.add(this.inputValue);
        return res;
    }

}
