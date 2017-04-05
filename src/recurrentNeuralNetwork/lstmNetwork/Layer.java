package recurrentNeuralNetwork.lstmNetwork;

import additionalClasses.Matrix;

import java.util.ArrayList;

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

    private Cell cell;

    public Layer(/*Matrix inputWeight, */int iinputAmount, int hhiddenAmount, int ooutputAmount){
        //this.inputWeight = inputWeight;
        this.inputAmount = iinputAmount;
        this.outputAmount = ooutputAmount;
        this.hiddenAmount = hhiddenAmount;
    }

    public Matrix activate(Matrix inputWeight){

        this.cell = new Cell(this.inputAmount, this.outputAmount, inputWeight, this.prevHiddenWeight);

        this.cellState = this.cell.getCellState(prevCellState);
        this.prevCellState = this.cellState;

        Matrix output = this.cell.getHiddenValue(this.cellState);
        this.prevHiddenWeight = output;

        return output;
    }

    public void reset(){
        this.prevHiddenWeight = new Matrix(this.outputAmount);
        this.prevCellState = new Matrix(this.outputAmount);
    }

    public ArrayList<Matrix> getParams(){
        return this.cell.getParams();
    }
}
