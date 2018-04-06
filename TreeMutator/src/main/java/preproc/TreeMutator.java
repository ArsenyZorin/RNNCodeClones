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
    private static List<String> black_list;
    private static List<String> white_list;
    private final PsiGen psi_generator;

    public TreeMutator(final PsiGen psi_generator) {
        this.psi_generator = psi_generator;
    }

    public TreeMutator(final PsiGen psi_generator, final List<String> black_list, final List<String> white_list) {
        this.psi_generator = psi_generator;
        this.black_list = black_list;
        this.white_list = white_list;
    }

    /***
     * Checks whether file has java extension
     * @param file_path Path of file for extension checking
     * @return True if file is .java
     */
    private static boolean checkFileExtension(Path file_path) {
        String file_name = file_path.getFileName().toString();
        return file_name.endsWith(JAVA_EXTENSION);
    }

    /***
     * Analyze all files in directory and create AST
     * @param repo_path Path to directory for analysis
     * @return List of AST for every method in directory
     * @throws IOException
     */
    private List<ASTEntry> analyzeDirectory(String repo_path) throws IOException{

        List<ASTEntry> repo_tree = new ArrayList<>();
        final List<String> java_files = Files.walk(Paths.get(repo_path),
                FileVisitOption.FOLLOW_LINKS).
                filter(f -> Files.isRegularFile(f))
                .filter(TreeMutator::checkFileExtension)
                .map(Path::toString)
                .map(s -> s.replace(repo_path, ""))
                .collect(Collectors.toList());

        if(java_files.isEmpty()){
            System.out.println("Dir for analyzing is empty");
            System.exit(1);
        }

        for(String file : java_files){
            System.out.print(String.format("Analyzing: %s/%s",
                    java_files.indexOf(file), java_files.size()));
            ASTEntry tree;
            try {
                tree = this.psi_generator.parseFile(repo_path + file).removeSpaces(black_list);
            } catch (NullPointerException ex){
                continue;
            }
            repo_tree.add(tree);
            System.out.print("\r\b");
        }
        return getMethodBlocks(repo_tree);
    }

    /***
     * Mutates every node in a tree
     * @param trees List of trees for mutation
     * @return List of mutated trees
     */
    public List<ASTEntry> treeMutator(List<ASTEntry> trees){
        for(ASTEntry node : trees) {
            node.mutate(black_list);
            System.out.print(String.format("Mutates completed: %s/%s", trees.indexOf(node), trees.size()));
            System.out.print("\r\b");
        }
        System.out.println();
        return trees;
    }

    /***
     * Extracts methods from every tree in a list of trees
     * @param trees List of trees for methods extraction
     * @return List of all methods in the tree
     */
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
