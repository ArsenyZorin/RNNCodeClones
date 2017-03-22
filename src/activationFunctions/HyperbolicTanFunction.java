package activationFunctions;

/**
 * Created by arseny on 22.03.17.
 */
public class HyperbolicTanFunction implements IActivationFunction {

    /* Function slope parameter */
    private double alpha;

    public HyperbolicTanFunction(double alpha){
        this.alpha = alpha;
    }

    @Override
    public double function(double x) {
        double numerator;
        double denominator;

        numerator = Math.exp(x / this.alpha) - Math.exp(-x / this.alpha);
        denominator = Math.exp(x / this.alpha) + Math.exp(-x / this.alpha);
        return (numerator/denominator);
    }

    @Override
    public double derivative(double x) {
        double numerator;
        double denominator;

        numerator = 4 * Math.exp(2 * x / this.alpha);
        denominator = this.alpha * (Math.exp(2 * x / this.alpha));

        return (numerator/denominator);
    }
}
