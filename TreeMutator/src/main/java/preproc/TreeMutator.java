package preproc;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.ObjectUtils;
import trees.ASTEntry;
import trees.PsiGen;

import java.io.File;
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

        for(String file : javaFiles){
            System.out.print(String.format("Analyzing: %s/%s\tFile name: %s",
                    javaFiles.indexOf(file), javaFiles.size(), file));
            //changeEncoding(repoPath + file);
            ASTEntry tree;
            try {
                tree = this.psiGenerator.parseFile(repoPath + file).removeSpaces(blackList);
            } catch (NullPointerException ex){
                continue;
            }
            repoTree.add(tree);
            System.out.print("\r\b");
        }
        return getMethodBlocks(repoTree);
    }

    public List<ASTEntry> treeMutator(List<ASTEntry> trees){
        for(ASTEntry node : trees) {
            node.mutate(blackList);
            System.out.print(String.format("Mutates completed: %s/%s", trees.indexOf(node), trees.size()));
            System.out.print("\r\b");
        }
        System.out.println();
        return trees;
    }

    private void changeEncoding(String fileName){
        try {
            File file = new File(fileName);
            String content = FileUtils.readFileToString(file, "ISO8859_1");
            FileUtils.write(file, content, "UTF-8");
        } catch (Exception e){
            System.out.println(e.getMessage());
        }
    }

    private List<ASTEntry> getMethodBlocks(List<ASTEntry> trees){
        List<ASTEntry> methodsList = new ArrayList<>();
        for(ASTEntry tree : trees){
            if(!tree.nodeName.contains(METHOD_TOKEN))
                methodsList.addAll(getMethodBlocks(tree.children));
            else
                for(ASTEntry node : tree.children)
                    if(node.nodeName.contains(CODEBLOCK_TOKEN)) {
                        methodsList.add(node);
                        break;
                    }
        }

        return methodsList;
    }

    public List<ASTEntry> analyzeDir(String repoPath) {
        List<ASTEntry> methods = null;
        try {
            methods = analyzeDirectory(repoPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return methods;
    }
}
