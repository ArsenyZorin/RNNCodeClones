package arguments;

import com.beust.jcommander.Parameter;

import javax.annotation.ParametersAreNonnullByDefault;

public class Arguments {
    @Parameter(names = {"--evaltype"}, description = "Evaluation type", validateWith = EvalTypeValidator.class, required = true)
    private String evalType = "full";

    @Parameter(names = {"--outputDir"}, description = "Specify output dir for data saving")
    private String outputDir;

    @Parameter(names = {"--inputDir"}, description = "Specify input data", required = true)
    private String inputDir;

    @Parameter(names = "--help", help = true)
    private boolean help;

    public String getEvalType() {
        return evalType;
    }

    public String getInputDir() {
        return inputDir;
    }

    public String getOutputDir() {
        return outputDir;
    }
}


