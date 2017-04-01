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
    private Gate candidate;

    public Cell(Matrix inputWeight, Matrix prevHiddenWeight, int outputAmount){

        this.inputWeight = inputWeight;
        this.prevHiddenWeight = prevHiddenWeight;

        this.inputAmount = inputWeight.getArrSize();
        this.outputAmount = outputAmount;

        this.inputGate = new Gate(this.prevHiddenWeight, this.inputWeight, this.outputAmount);
        this.forgetGate = new Gate(this.prevHiddenWeight, this.inputWeight, this.outputAmount);
        this.outputGate = new Gate(this.prevHiddenWeight, this.inputWeight, this.outputAmount);
        this.candidate = new Gate(this.prevHiddenWeight, this.inputWeight, this.outputAmount);
    }

    public Matrix getCellState(Matrix prevCellState){

        Matrix inputGateValue = inputGate.getGateValue(new SigmoidFunction(1.0));
        Matrix forgetGateValue = forgetGate.getGateValue(new SigmoidFunction(1.0));
        Matrix candidateValue = candidate.getGateValue(new HyperbolicTanFunction(1.0));

        Matrix firstCellValue = prevCellState.elementMultyply(forgetGateValue);
        Matrix secondCellValue = inputGateValue.elementMultyply(candidateValue);

        return firstCellValue.add(secondCellValue);
    }

    public Matrix getHiddenValue(Matrix cellStateValue){
        Matrix outputGateValue = outputGate.getGateValue(new SigmoidFunction(1.0));
        Matrix hiddenFirstValue = cellStateValue.applyActFunc(new HyperbolicTanFunction(1.0));

        return hiddenFirstValue.elementMultyply(outputGateValue);
    }
}
