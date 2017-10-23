package arguments;

import com.beust.jcommander.Parameter;

import javax.annotation.ParametersAreNonnullByDefault;
import java.io.File;

public class Arguments {
    @Parameter(names = {"--evaltype"}, description = "Evaluation type", validateWith = EvalTypeValidator.class)
    private String evalType = "full";

    @Parameter(names = {"--outputDir"}, description = "Specify output dir for data saving")
    private String outputDir;

    @Parameter(names = {"--inputDir"}, description = "Specify input data", validateWith = DirValidator.class, required = true)
    private String inputDir;

    @Parameter(names = {"--help", "-h"}, help = true)
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
}


