package data;

import java.util.ArrayList;

/**
 * Created by arseny on 31.03.17.
 */
public class Sequence {
    private ArrayList<Step> steps;

    public Sequence(){}
    public Sequence(ArrayList<Step> steps){
        this.steps = steps;
    }

    public ArrayList<Step> getSteps() {
        return steps;
    }

    public void setSteps(ArrayList<Step> steps) {
        this.steps = steps;
    }

    /* toString?*/

}
