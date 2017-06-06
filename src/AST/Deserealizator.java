package AST;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Deserealizator{
    /*
     * Wrong deserealizator. File was written through pickle. It's another stream
     */
    public static Object deserealization(String path){
        List<Node> nodes = new ArrayList<>();

        try{
            FileInputStream fis = new FileInputStream(path);
            ObjectInputStream ois = new ObjectInputStream(fis);
            Object obj = ois.readObject();
            return obj;
        } catch(Exception e){
            e.printStackTrace();
        }
        return nodes;
    }
}
