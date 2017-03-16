package activationFunction;

/**
 * Created by zorin on 15.03.2017.
 */
public class SigmoidFunction implements IActivationFunction {

    /* Function slope parameter */
    private double alpha;

    public SigmoidFunction(double alpha){
        this.alpha = alpha;
    }

    public double function(double x){
        return (1 / (1 + Math.exp(-this.alpha * x)));
    }

    public double derivative(double x){
        double y = function(x);
        return (this.alpha * y * (1 - y));
    }
}
