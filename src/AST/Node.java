package AST;

import recurrentNeuralNetwork.training.Loss;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by arseny on 06.06.17.
 */
public class Node implements Serializable{
    private List<Token> allNodes;
    private List<Token> nonLeafs;
    private Token rootNode;

    public Node(){
        allNodes = new ArrayList<>();
        nonLeafs = new ArrayList<>();
        rootNode = new Token();
    }

    public List<Token> getAllNodes() { return this.allNodes; }
    public List<Token> getNonLeafs() { return this.nonLeafs; }
    public Token getRootNode() { return this.rootNode; }

}
