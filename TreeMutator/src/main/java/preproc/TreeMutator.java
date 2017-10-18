package preproc;

import org.apache.commons.io.FileUtils;
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

        //int i = 0;
        for(String file : javaFiles){
            System.out.print(String.format("\rAnalyzing: %s/%s\tFile name: %s",
                    javaFiles.indexOf(file), javaFiles.size(), file));
            changeEncoding(repoPath + file);
            repoTree.add(this.psiGenerator.parseFile(repoPath + file).removeSpaces(blackList));
        }
        return getMethodBlocks(repoTree);
    }

    public List<ASTEntry> treeMutator(List<ASTEntry> trees){
        List<ASTEntry> methodList = getMethodBlocks(trees);

        int i = 0;
        for(ASTEntry node : methodList) {
            node.mutate(blackList);
            System.out.println("Mutates completed: " + (++i) + "/" + methodList.size());

        }
        return methodList;
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
        /*for(trees.ASTEntry node : tree){
            for (trees.ASTEntry childNode : node.children)
                if(childNode.nodeName.contains(METHOD_TOKEN)){
                    for(trees.ASTEntry child : node.children)
                        if(child.nodeName.contains(CODEBLOCK_TOKEN))
                            methodsList.add(child);
            }
        }*/

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