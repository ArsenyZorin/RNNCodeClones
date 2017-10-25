    package arguments;

import com.beust.jcommander.IParameterValidator;
import com.beust.jcommander.ParameterException;

import java.io.File;
import java.nio.file.Files;

public class DirValidator implements IParameterValidator{
    public void validate(String name, String value) throws ParameterException {
        File file = new File(value);

        if(Files.exists(file.toPath())) {
            if (Files.isDirectory(file.toPath())) {
                if ("--outputDir".equals(name))
                    System.out.println("Specified dir in parameter " + name + "already exists.\nFiles will be rewritten!");
            }
            else
                throw new ParameterException("Specified path in " + name + " parameter is file");
        } else {
            if ("--outputDir".equals(name))
                file.mkdirs();
            else
                throw new ParameterException("Invalid path to analyzing directory");
        }
    }
}
