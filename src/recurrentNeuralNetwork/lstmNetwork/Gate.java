package recurrentNeuralNetwork.lstmNetwork;
import activationFunctions.*;
import additionalClasses.*;

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

    /* Activation functions */
    private IActivationFunction func;

    public Gate(Matrix prevHiddenWeight, Matrix inputWeight, int outputAmount) {

        this.prevHiddenWeight = prevHiddenWeight;
        this.inputWeight = inputWeight;

        this.inputAmount = inputWeight.getSize();
        this.outputAmount = outputAmount;

        this.prevHiddenValue = Matrix.random(this.inputAmount, this.outputAmount);
        this.inputValue = Matrix.random(this.inputAmount, this.outputAmount);
    }

    public Matrix getGateValue(IActivationFunction func){
        Matrix prod1 = this.prevHiddenValue.multiply(this.prevHiddenWeight);
        Matrix prod2 = this.inputValue.multiply(this.inputWeight);

        return (prod1.add(prod2)).getActFunc(func);
    }
}
