package additionalClasses;

import activationFunctions.IActivationFunction;

/**
 * Created by arseny on 22.03.17.
 */
public class Matrix {

    private int rows;
    private int cols;
    private double [] array;

    public Matrix(int rows){
        this.rows = rows;
        this.cols = 1;
        this.array = new double [this.rows * this.cols];
    }

    public Matrix(int rows, int cols){
        this.rows = rows;
        this.cols = cols;
        this.array = new double[this.rows * this.cols];
    }

    public Matrix(double [] array){
        this.rows = array.length;
        this.cols = 1;
        this.array = array;
    }

    public int getSize(){
        return this.array.length;
    }

    public Matrix add (/* Matrix ar1, */Matrix ar2){
        if(this.rows != ar2.rows || this.cols != ar2.cols)
            throw new ArithmeticException("Matrix dimensions error");

        Matrix res = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.array.length; i++)
            res.array[i] = this.array[i] + ar2.array[i];

        return res;
    }

    public Matrix multiply(Matrix ar2){
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
        return res;
    }

    public Matrix elementMultyply(Matrix ar1){
        if(this.rows != ar1.rows || this.cols != ar1.cols){
            throw new ArithmeticException("Matrix dimension error!");
        }

        Matrix res = new Matrix(this.rows, this.cols);
        for(int i = 0; i < this.array.length; i++)
            res.array[i] = this.array[i] * ar1.array[i];

        return res;
    }

    public Matrix getActFunc(IActivationFunction func){
        Matrix res = new Matrix(this.rows, this.cols);
        for (int i = 0; i < array.length; i++)
            res.array[i] = func.function(this.array[i]);

        return res;
    }

    public static Matrix random(int inputAmount, int outputAmount) {
        Matrix res = new Matrix(inputAmount, outputAmount);
        for (int i = 0; i < res.array.length; i++)
            res.array[i] = Math.random();
        return res;
    }
}