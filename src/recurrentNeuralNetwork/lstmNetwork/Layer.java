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

    //private int outputAmount;
    private int hiddenAmount;
    private int inputAmount;

    private Cell cell;

    public Layer(/*Matrix inputWeight, */int inputAmount, int hiddenAmount/*, int outputAmount*/){
        //this.inputWeight = inputWeight;
        this.inputAmount = inputAmount;
        //this.outputAmount = outputAmount;
        this.hiddenAmount = hiddenAmount;
    }

    public Matrix activate(Matrix inputWeight){

        this.cell = new Cell(this.inputAmount, this.hiddenAmount, inputWeight, this.prevHiddenWeight);

        this.cellState = this.cell.getCellState(prevCellState);
        this.prevCellState = this.cellState;

        Matrix output = this.cell.getHiddenValue(this.cellState);
        this.prevHiddenWeight = output;

        return output;
    }

    public void reset(){
        this.prevHiddenWeight = new Matrix(this.hiddenAmount);
        this.prevCellState = new Matrix(this.hiddenAmount);
    }

    public ArrayList<Matrix> getParams(){
        return this.cell.getParams();
    }
}
