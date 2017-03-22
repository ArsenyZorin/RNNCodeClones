package recurrentNeuralNetwork.lstmNetwork;
import activationFunctions.*;
import additionalClasses.*;

/**
 * Created by arseny on 21.03.17.
 */
public class Cell {

    /* Weight of previous - current vector */
    private Matrix prevHiddenWeight;

    /* Weight of the input - current vector*/
    private Matrix inputWeight;

    private int inputAmount;
    private int outputAmount;

    /* Cell gates */
    private Gate inputGate;
    private Gate forgetGate;
    private Gate outputGate;
    private Gate candidateValue;

    public Cell(Matrix inputWeight, Matrix prevHiddenWeight, int inputAmount, int outputAmount){

        this.inputWeight = inputWeight;
        this.prevHiddenWeight = prevHiddenWeight;
        this.inputAmount = inputAmount;
        this.outputAmount = outputAmount;

        this.inputGate = new Gate(this.inputAmount, this.outputAmount
                , this.prevHiddenWeight, this.inputWeight);
        this.forgetGate = new Gate(this.inputAmount, this.outputAmount
                , this.prevHiddenWeight, this.inputWeight);
        this.outputGate = new Gate(this.inputAmount, this.outputAmount
                , this.prevHiddenWeight, this.inputWeight);
        this.candidateValue = new Gate(this.inputAmount, this.outputAmount
                , this.prevHiddenWeight, this.inputWeight);
    }

    public Matrix getCellState(Matrix prevCellState){
        Matrix inputGateValue = inputGate.
    }



}
