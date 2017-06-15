import com.intellij.lang.ASTNode;
import com.intellij.openapi.editor.Document;
import com.intellij.psi.impl.source.tree.CompositeElement;
import com.intellij.psi.impl.source.tree.LeafElement;

import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

public class ASTEntry {
    public final String nodeName;
    public final short nodeIndex;
    public final int sourceStart;
    public final int sourceEnd;
    public final ASTEntry parent;
    public List<ASTEntry> children;
    public final boolean isTerminal;
    public String text;
    //private String spacelessText = "";

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

    public ASTEntry getSpacelessText(){
        //text = appendSpacelessText(spacelessText);
        return this;
    }

    /*private String appendSpacelessText(String spacelessText){
        for(ASTEntry child: children)
            spacelessText = child.appendSpacelessText(spacelessText);
        return spacelessText + text;
    }*/

    @Override
    public String toString() {
        return shiftedString("");
    }

}
