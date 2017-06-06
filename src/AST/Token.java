package AST;

import java.util.List;

/**
 * Created by arseny on 26.05.17.
 */
public class Token {
    private String type;
    private Token parent;
    private List<Token> children;
    private boolean isLeaf;
    private int position;
    private int leftRate;
    private int rightRate;
    private int leafNum;
    private int childrenNum;
    private int index;
    private int startLine;
    private int endLine;
    private Author author; //?

    public Token(){}

    public Token(String type, Token parent, List<Token> children,
                 boolean isLeaf, int index, int startLine,
                 int endLine, Author author) {
        this.type = type;
        this.parent = parent;
        this.children = children;
        this.isLeaf = isLeaf;
        this.position = 0;
        this.leftRate = 0;
        this.rightRate = 0;
        this.leafNum = 0;
        this.childrenNum = 0;
        this.index = index;
        this.startLine = startLine;
        this.endLine = endLine;
        this.author = author;
    }

    public String getType() { return type; }
    public Token getParent() { return parent; }
    public List<Token> getChildren() { return children; }
    public boolean isLeaf() { return isLeaf; }
    public int getPosition() { return position; }
    public int getLeftRate() { return leftRate; }
    public int getRightRate() { return rightRate; }
    public int getLeafNum() { return leafNum; }
    public int getChildrenNum() { return childrenNum; }
    public int getIndex() { return index; }
    public int getStartLine() { return startLine; }
    public int getEndLine() { return endLine; }
    public Author getAuthor() { return author; }

    public void setType(String type) { this.type = type; }
    public void setParent(Token parent) { this.parent = parent; }
    public void setChildren(List<Token> children) { this.children = children; }
    public void setLeaf(boolean leaf) { isLeaf = leaf; }
    public void setPosition(int position) { this.position = position; }
    public void setLeftRate(int leftRate) { this.leftRate = leftRate; }
    public void setRightRate(int rightRate) { this.rightRate = rightRate; }
    public void setLeafNum(int leafNum) { this.leafNum = leafNum; }
    public void setChildrenNum(int childrenNum) { this.childrenNum = childrenNum; }
    public void setIndex(int index) { this.index = index; }
    public void setStartLine(int startLine) { this.startLine = startLine; }
    public void setEndLine(int endLine) { this.endLine = endLine; }
    public void setAuthor(Author author) { this.author = author;}

    public String toString(){
        return this.type + "_" + this.index;
    }
}
