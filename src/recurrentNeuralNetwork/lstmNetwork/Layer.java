package recurrentNeuralNetwork.lstmNetwork;

import additionalClasses.Matrix;

/**
 * Created by arseny on 22.03.17.
 */
public class Layer {

    /* Previous cell state value */
    private Matrix prevCellState;
    /* Current cell state value */
    private Matrix cellState;
    /*  */
    private Matrix inputWeight;
    private Matrix prevHiddenWeight;
    private int outputAmount;
    private int hiddenAmount;
    private int inputAmount;

    public Layer(Matrix inputWeight, int inputAmount, int hiddenAmount, int outputAmount){
        this.inputWeight = inputWeight;
        this.inputAmount = inputAmount;
        this.outputAmount = outputAmount;
        this.hiddenAmount = hiddenAmount;
    }

    public Matrix activate(Matrix inputWeight){

        Cell cell = new Cell(inputWeight, this.prevHiddenWeight, this.outputAmount);

        this.cellState = cell.getCellState(prevCellState);
        this.prevCellState = this.cellState;

        Matrix output = cell.getHiddenValue(this.cellState);
        this.prevHiddenWeight = output;

        return output;
    }

    public void reset(){
        this.prevHiddenWeight = new Matrix(this.outputAmount);
        this.prevCellState = new Matrix(this.outputAmount);
    }
}
