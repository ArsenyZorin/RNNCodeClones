package arguments;

import com.beust.jcommander.Parameter;
import java.io.File;

public class Arguments {
    @Parameter(names = {"--evalType", "--type"}, description = "Evaluation type. Could be:" +
            "\n\t\ttrain - given for training all neural networks" +
            "\n\t\teval - given when only analysis of source code is required" +
            "\n\t\tmutate - given when only mutation of source code, given by inputDir, is required" +
            "\n\t\tfull - given when full calculation is required",
            validateWith = EvalTypeValidator.class)
    private String evalType = "full";

    @Parameter(names = {"--outputDir", "--output"}, description = "Specify output dir for data saving", validateWith = DirValidator.class)
    private String outputDir = System.getProperty("user.home");

    @Parameter(names = {"--inputDir", "--input"}, description = "Specify input dir for data analyzing. " +
            "Not required when evaluation type is train", validateWith = DirValidator.class)
    private String inputDir;

    @Parameter(names = {"--help", "-h"}, help = true, description = "Prints help message")
    private boolean help;

    public String getEvalType() {
        return evalType;
    }

    public String getInputDir() {
        return new File(inputDir).getAbsolutePath();
    }

    public String getOutputDir() {
        return new File(outputDir).getAbsolutePath();
    }

    public boolean getHelp(){
        return help;
    }
}


