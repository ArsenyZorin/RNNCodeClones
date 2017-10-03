import com.intellij.lang.ASTNode;
import com.intellij.openapi.editor.Document;
import com.intellij.psi.impl.source.tree.CompositeElement;
import com.intellij.psi.impl.source.tree.LeafElement;
import org.apache.commons.lang.ObjectUtils;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class ASTEntry {
    public String nodeName;
    public final short nodeIndex;
    public int sourceStart;
    public int sourceEnd;
    public final ASTEntry parent;
    public List<ASTEntry> children;
    public final boolean isTerminal;
    public String text;

    public ASTEntry(ASTEntry entry) {
        this.nodeName = entry.nodeName;
        this.nodeIndex = entry.nodeIndex;
        this.sourceStart = entry.sourceStart;
        this.sourceEnd = entry.sourceEnd;
        this.parent = entry.parent;
        this.children = entry.children;
        this.isTerminal = entry.isTerminal;
        this.text = entry.text;
    }

    public ASTEntry(String nodeName, short nodeIndex, int sourceStart, int sourceEnd,
                    ASTEntry parent, List<ASTEntry> children, boolean isTerminal, String text) {
        this.nodeName = nodeName;
        this.nodeIndex = nodeIndex;
        this.sourceStart = sourceStart;
        this.sourceEnd = sourceEnd;
        this.parent = parent;
        this.children = children;
        this.isTerminal = isTerminal;
        this.text = text;
    }

    public ASTEntry(ASTNode astNode, ASTEntry parent, Document doc) {
        this.nodeName = astNode.getElementType().toString();
        this.nodeIndex = astNode.getElementType().getIndex();
        this.sourceStart = doc.getLineNumber(astNode.getTextRange().getStartOffset());
        this.sourceEnd = doc.getLineNumber(astNode.getTextRange().getEndOffset());
        this.parent = parent;
        this.children = new LinkedList<ASTEntry>();
        this.isTerminal = (astNode instanceof LeafElement) || ((CompositeElement) astNode).countChildren(null) == 0;
        this.text = astNode.getText();
    }

    private String shiftedString(String shift) {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("\n")
                .append(shift).append("┗ ")
                .append(nodeName);/*
                .append(" [").append(sourceStart)
                .append(":").append(sourceEnd)
                .append("]");//.append(text);*/

        for (ASTEntry child : children) {
            if (child.isTerminal) {
                stringBuilder.append(" ");
                stringBuilder.append(child.nodeName);
            } else {
                stringBuilder.append(child.shiftedString(shift + " "));
            }
        }
        return stringBuilder.toString();
    }

    public ASTEntry removeSpaces(List<String> blackList){
        children = children.stream().filter(p->!blackList.contains(p.nodeName)).collect(Collectors.toList());
        for(ASTEntry child : children) {
            child.removeSpaces(blackList);
        }
        return this;
    }

    public int getNodesAmount(){
        int amount = 0;
        if(children.size() == 0) {
            amount++;
            return amount;
        }
        for(ASTEntry node : children){
            amount += node.getNodesAmount();
        }
        return amount;
    }

    public List<String> getAllTokensList(){
        List<String> nodesTokens = new ArrayList<>();
        if(children.size() == 0){
            nodesTokens.add(nodeName);
            return nodesTokens;
        }
        for(ASTEntry node : children){
            nodesTokens.addAll(node.getAllTokensList());
        }
        return nodesTokens;
    }

    public String getAllTokensString(){
        StringBuilder nodesTokens = new StringBuilder("");
        if(children.size() == 0){
            nodesTokens.append(nodeName).append(" ");
            return nodesTokens.toString();
        }

        for(ASTEntry node : children){
            nodesTokens.append(node.getAllTokensString());
        }
        return nodesTokens.toString();
    }

    public void mutate(List<String> blackList){
        Random rnd = new Random();
        int func = rnd.nextInt(3);

        if(children.size() < 5)
            func = 1;
        if(children.size() < 4)
            func = 2;

        switch(func) {
            case 0:
                deleteNode(blackList);
                break;
            case 1:
                copyNode();
                break;
            default:
                break;
        }
    }

    private void deleteNode(List<String> blackList){
        System.out.println("Node deletion");
        int[] pos = getStartEndMethod();

        Random rnd = new Random();

        int amountOfLines = rnd.nextInt(children.size() - 3);

        for (int i = 0; i < amountOfLines; i++) {
            int line;
            if(pos[1] == 0)
                line = rnd.nextInt(pos[1]) + pos[0] + 1;
            else
                line = rnd.nextInt(pos[1] - 1) + pos[0] + 1;

            children.forEach(child -> {
                if (children.indexOf(child) == line) {
                    child.nodeName = blackList.stream()
                            .filter(p -> p.contains("WHITE")).findFirst().get();
                }
            });
        }
    }

    private void copyNode(){
        System.out.println("Node copy-paste");
        int[] pos = getStartEndMethod();
        Random rnd = new Random();
        int amountOfLines = rnd.nextInt(children.size() - 2);

//        int[] copyLine = new int[amountOfLines];
//        int[] pasteLine = new int[amountOfLines];

        for(int i = 0; i < amountOfLines; i++){
            int copyLine;

            System.out.println(pos[1]);
            System.out.println(pos[0]);

            if (pos[1] == 0)
                copyLine = rnd.nextInt(pos[1]) + pos[0] + 1;
            else
                copyLine = rnd.nextInt(pos[1] - 1) + pos[0] + 1;
            int pasteLine = rnd.nextInt(pos[1] - 1) + pos[0] + 1;

            while (pasteLine == copyLine){
                pasteLine = rnd.nextInt(pos[1] - 1) + pos[0] + 1;
            }

            ASTEntry copyNode = null;
            for(ASTEntry child : children){
                if (children.indexOf(child) == copyLine) {
                    copyNode = new ASTEntry(child);
                    break;
                }
            }
            children.add(pasteLine, copyNode);
        }
    }

    private int[] getStartEndMethod(){
        int pos[] = new int[2];
        children.forEach(p->{
            if("LBRACE".equals(p.nodeName))
                pos[0] = children.indexOf(p);
            if("RBRACE".equals(p.nodeName))
                pos[1] = children.indexOf(p);
        });

        if(pos[1] == 0)
            System.out.println("WTF???");
        return pos;
    }

    @Override
    public String toString() {
        return shiftedString("");
    }

}
