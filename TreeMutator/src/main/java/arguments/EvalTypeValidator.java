package arguments;

import com.beust.jcommander.IParameterValidator;
import com.beust.jcommander.ParameterException;

import java.util.Arrays;

public class EvalTypeValidator implements IParameterValidator {
    public void validate(String name, String value) throws ParameterException{
        StringBuilder validValue = new StringBuilder();
        for (EvalType type : EvalType.values()) {
            validValue.append("\n\t\t")
                    .append(type.toString().toLowerCase());
            if (type.toString().equals(value))
                return;
        }

        throw new ParameterException("Invalid value of parameter " + name + ".\nShould be:" + validValue.toString());
    }
}
