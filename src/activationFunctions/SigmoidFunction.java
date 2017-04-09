package activationFunctions;

/**
 * Created by zorin on 15.03.2017.
 */
public class SigmoidFunction implements IActivationFunction {

    /* Function slope parameter */
    private double alpha;

    public SigmoidFunction(double alpha){
        this.alpha = alpha;
    }

    /**
     * Function computation
     * @param x function value
     * @return function result
     */
    @Override
    public double function(double x){
        return (1 / (1 + Math.exp(-this.alpha * x)));
    }

    /**
     * Computation of function derivative
     * @param x function value
     * @return function derivative result
     */
    @Override
    public double derivative(double x){
        double y = function(x);
        return (this.alpha * y * (1 - y));
    }
}
