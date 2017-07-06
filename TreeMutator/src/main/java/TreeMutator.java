import com.intellij.psi.impl.source.SourceTreeToPsiMap;
import com.intellij.util.containers.InternalIterator;

import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class TreeMutator {
    private static final String JAVA_EXTENSION = ".java";
    private static final String METHOD_TOKEN = "METHOD";
    private static final String CODEBLOCK_TOKEN= "CODE_BLOCK";
    private static List<String> blackList;
    private static List<String> whiteList;
    private final PsiGen psiGenerator;

    public TreeMutator(final PsiGen psiGenerator) {
        this.psiGenerator = psiGenerator;
    }

    public TreeMutator(final PsiGen psiGenerator, final List<String> blackList, final List<String> whiteList) {
        this.psiGenerator = psiGenerator;
        this.blackList = blackList;
        this.whiteList = whiteList;
    }

    private static boolean checkFileExtension(Path filePath) {
        String fileName = filePath.getFileName().toString();
        return fileName.endsWith(JAVA_EXTENSION);
    }

    private List<ASTEntry> analyzeDirectory(String repoPath) throws IOException{

        List<ASTEntry> repoTree = new ArrayList<>();
        final List<String> javaFiles = Files.walk(Paths.get(repoPath),
                FileVisitOption.FOLLOW_LINKS).
                filter(f -> Files.isRegularFile(f))
                .filter(TreeMutator::checkFileExtension)
                .map(Path::toString)
                .map(s -> s.replace(repoPath, ""))
                .collect(Collectors.toList());

        int i = 0;
        for(String file : javaFiles){
            repoTree.add(Main.buildPSI(repoPath + file).removeSpaces(blackList));
            System.out.println("Completed " + (++i) + " / " + javaFiles.size());
        }
        return getMethodBlocks(repoTree);
    }

    List<ASTEntry> treeMutator(List<ASTEntry> trees){
        List<ASTEntry> methodList = getMethodBlocks(trees);

        int i = 0;
        for(ASTEntry node : methodList) {
            node.mutate(blackList);
            System.out.println("Mutates completed: " + (++i) + "/" + methodList.size());

        }
        return methodList;
    }

    void oneHotCreation(List<ASTEntry> tree){
        int size;
        for (ASTEntry node : tree) {
            size = nodesAmount(node);
            
        }
        
    }



    private List<ASTEntry> getMethodBlocks(List<ASTEntry> tree){
        List<ASTEntry> methodsList = new ArrayList<>();
        for(ASTEntry node : tree){
            if(!node.nodeName.contains(CODEBLOCK_TOKEN))
                methodsList.addAll(getMethodBlocks(node.children));
            else {
                methodsList.add(node);
            }
        }
        return methodsList;
    }

    List<ASTEntry> analyzeDir(String repoPath) {
        List<ASTEntry> methods = null;
        try {
            methods = analyzeDirectory(repoPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return methods;
    }

    private int nodesAmount(ASTEntry tree){
        int amount = 0;
        if(tree.children.size() == 0)
            amount ++;

        for(ASTEntry node : tree.children){
            amount += nodesAmount(node);
        }

        return amount;
    }

}
