package recurrentNeuralNetwork.lstmNetwork;
import activationFunctions.*;
import additionalClasses.*;

import java.util.ArrayList;

/**
 * Created by arseny on 21.03.17.
 */
public class Cell {

    private int inputAmount;
    private int outputAmount;

    /* Cell gates */
    private Gate inputGate;
    private Gate forgetGate;
    private Gate outputGate;
    private Gate candidate;

    public Cell(int inputAmount, int outputAmount, Matrix inputWeight, Matrix prevHiddenWeight){

        this.inputAmount = inputAmount;
        this.outputAmount = outputAmount;

        this.inputGate = new Gate(inputWeight, prevHiddenWeight, this.inputAmount, this.outputAmount);
        this.forgetGate = new Gate(inputWeight, prevHiddenWeight, this.inputAmount, this.outputAmount);;
        this.outputGate = new Gate(inputWeight, prevHiddenWeight, this.inputAmount, this.outputAmount);;
        this.candidate = new Gate(inputWeight, prevHiddenWeight, this.inputAmount, this.outputAmount);;
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

    public ArrayList<Matrix> getParams(){
        ArrayList<Matrix> res = new ArrayList<>();
        for (Matrix matrix : inputGate.getParams())
            res.add(matrix);

        for (Matrix matrix : forgetGate.getParams())
            res.add(matrix);

        for (Matrix matrix : outputGate.getParams())
            res.add(matrix);

        for (Matrix matrix : candidate.getParams())
            res.add(matrix);

        return  res;
    }
}
