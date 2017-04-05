package additionalClasses;

import activationFunctions.IActivationFunction;
import recurrentNeuralNetwork.lstmNetwork.Layer;

/**
 * Created by arseny on 22.03.17.
 */
public class Matrix {

    private int rows;
    private int cols;
    private double [] array;
    private double [] backArray;
    private double [] cache;

    public Matrix(int rows){
        this.rows = rows;
        this.cols = 1;
        this.array = new double [this.rows];
        this.backArray = new double [this.rows];
        this.cache = new double [this.rows];
    }

    public Matrix(int rows, int cols){
        this.rows = rows;
        this.cols = cols;
        this.array = new double[this.rows * this.cols];
        this.backArray = new double[this.rows * this.cols];
        this.cache = new double [this.rows * this.cols];
    }

    public Matrix(double [] array){
        this.rows = array.length;
        this.cols = 1;
        this.array = array;
        this.backArray = new double[array.length];
        this.cache = new double[array.length];
    }

    public int getArrSize(){
        return this.array.length;
    }

    public double getArrayElem (int index){
        return this.array[index];
    }

    public void setArrayElem (int index, double elem){ this.array[index] = elem; }

    public double getBackArrayElem (int index){
        return this.backArray[index];
    }

    public void setBackArrayElem (int index, double elem){ this.backArray[index] = elem; }

    public void setCacheElem (int index, double elem){ this.backArray[index] = elem; }

    public double getCacheElem (int index){ return this.backArray[index]; }

    public Matrix add (Matrix ar2){
        if(this.rows != ar2.rows || this.cols != ar2.cols)
            throw new ArithmeticException("Matrix dimensions error");

        Matrix res = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.array.length; i++)
            res.array[i] = this.array[i] + ar2.array[i];

        for(int i = 0; i < this.array.length; i++){
            this.backArray[i] += res.backArray[i];
            ar2.backArray[i] += res.backArray[i];
        }
        return res;
    }

    public Matrix multiply (Matrix ar2){
        if(this.cols != ar2.rows){
            throw new ArithmeticException("Matrix dimension error!");
        }

        Matrix res = new Matrix(this.rows, ar2.cols);

        for (int i = 0; i < this.rows; i++) {
            int ar1Col = this.cols * i;
            for (int j = 0; j < ar2.cols; j++) {
                double dot = 0.0;
                for (int k = 0; k < this.cols; k++)
                    dot += this.array[ar1Col + k] * ar2.array[ar2.cols * k + j];

                res.array[ar2.cols * i + j] = dot;
            }
        }

        for (int i = 0; i < this.rows; i++){
            int outcol = ar2.cols * i;
            for (int j = 0; j < ar2.cols; j++){
                double bw = res.backArray[outcol + j];
                for (int k = 0; k < this.cols; k++){
                    this.backArray[this.cols * i + k] += ar2.array[ar2.cols * k + j] * bw;
                    ar2.backArray[ar2.cols * k +j] += this.array[this.cols * i + k] * bw;
                }
            }
        }
        return res;
    }

    public Matrix elementMultyply (Matrix ar2){
        if(this.rows != ar2.rows || this.cols != ar2.cols){
            throw new ArithmeticException("Matrix dimension error!");
        }

        Matrix res = new Matrix(this.rows, this.cols);
        for(int i = 0; i < this.array.length; i++)
            res.array[i] = this.array[i] * ar2.array[i];

        for (int i = 0; i < this.array.length; i++){
            this.backArray[i] += ar2.array[i] * res.backArray[i];
            ar2.backArray[i] += this.array[i] * res.backArray[i];
        }

        return res;
    }

    public Matrix applyActFunc(IActivationFunction func){
        Matrix res = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.array.length; i++)
            res.array[i] = func.function(this.array[i]);

        for (int i = 0; i < this.array.length; i++)
            this.backArray[i] += func.derivative(this.array[i]) * res.backArray[i];

        return res;
    }

    public static Matrix random(int cols, int rows) {
        Matrix res = new Matrix(rows, cols);
        for (int i = 0; i < res.array.length; i++)
            res.array[i] = Math.random();
        return res;
    }
}