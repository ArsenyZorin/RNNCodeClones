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
    private static List<String> blackList;
    private List<ASTEntry> removeList = new ArrayList<>();
    private final PsiGen psiGenerator;

    public TreeMutator(final PsiGen psiGenerator) {
        this.psiGenerator = psiGenerator;
    }

    public TreeMutator(final PsiGen psiGenerator, final List<String> blackList) {
        this.psiGenerator = psiGenerator;
        this.blackList = blackList;
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
        return repoTree;
    }

    public List<ASTEntry> treeMutator(List<ASTEntry> trees){
        for(ASTEntry tree : trees){
            if(!tree.nodeName.contains("CODE_BLOCK"))
                treeMutator(tree.children);
            else
                tree.mutate(blackList);
        }
        return trees;
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

}
