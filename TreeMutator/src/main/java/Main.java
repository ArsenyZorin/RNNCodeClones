import arguments.Arguments;
import arguments.EvalType;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FileUtils;
import preproc.Embedding;
import gitrepos.Repository;
import preproc.TreeMutator;
import trees.ASTEntry;
import trees.PsiGen;

import java.io.*;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

public class Main {
    private final static PsiGen generator = new PsiGen();
    private final static List<String> spaces = Arrays.asList(
            "DOC_", "COMMENT", "PACKAGE", "IMPORT",
            "SPACE", "IMPLEMENTS", "EXTENDS", "THROWS",
            "PARAMETER_LIST");

    public static void main(String[] argv) {
        Arguments args = new Arguments();

        try {
            JCommander.newBuilder().programName("RNNCodeClones").addObject(args).build().parse(argv);
            args.globalValidation();
        } catch (ParameterException ex) {
            System.out.println(ex.getMessage());
            System.out.println("For additional info type: --help or -h");
            return;
        }

        if (args.getHelp()) {
            new JCommander(args, null, argv).usage();
            return;
        }

        List<String> whiteList = getAllAvailableTokens();
        List<String> blackList = whiteList.stream()
                .filter(p -> contains(spaces, p)).collect(Collectors.toList());

        TreeMutator treeMutator = new TreeMutator(generator, blackList, whiteList);
        whiteList.removeAll(blackList);
        Embedding emb = new Embedding(treeMutator, args.getEvalType(), args.getOutputDir());
        String saveFile = args.getOutputDir() + "/vectors";

        if (EvalType.MUTATE.toString().equals(args.getEvalType().toUpperCase())) {
            System.out.println("Directory for mutation: " + args.getInputDir());
            String repoPath = args.getInputDir();
            mutate(treeMutator, emb,
                    evaluate(treeMutator, emb, repoPath, saveFile + "/EvalCode"),
                    args.getOutputDir() + "/EvalMutatedCode");
            evaluate(treeMutator, emb, "/home/arseny/evals/jdbc", saveFile + "/EvalNonClone");

        } else if (EvalType.EVAL.toString().equals(args.getEvalType().toUpperCase())) {
            String repoPath = args.getInputDir();
            System.out.println("Start analyzing repo : " + repoPath);
            evaluate(treeMutator, emb, repoPath, saveFile + "/indiciesOriginCode");
        } else if (EvalType.TRAIN.toString().equals(args.getEvalType().toUpperCase())) {
            train(treeMutator, emb, saveFile);
        } else {
            System.out.println("Directory for analysis: " + args.getInputDir());
            String repoPath = args.getInputDir();
            train(treeMutator, emb, saveFile);

            List<ASTEntry> tree = evaluate(treeMutator, emb, repoPath, saveFile + "/EvalCode");
            mutate(treeMutator, emb, tree, saveFile + "/EvalMutatedCode");
            evaluate(treeMutator, emb, "/home/arseny/evals/jdbc", saveFile + "/EvalNonClone");
        }

        String path = Main.class.getResource("/clonesRecognition.py").getPath();
        System.out.println(path);

        String pythonArgs = "--type full --data " + args.getOutputDir();
//        pythonExec("/home/arseny/Repos/RNNCodeClones/Networks/clonesRecognition.py", pythonArgs);
    }

    public Main getMain() {
        return this;
    }

    private static boolean contains(List<String> blackList, String nodeName){
        for(String blackElem : blackList)
            if(nodeName.contains(blackElem))
                return true;

        return false;
    }

    private static void train(TreeMutator treeMutator, Embedding emb, String savePath){
        File dir = emb.getIdeaRepo();
        Repository repository = null;
        if(dir == null)
            repository = new Repository("/tmp/intellij-community", "https://github.com/JetBrains/intellij-community.git");
        List<ASTEntry> originTree = evaluate(treeMutator, emb, "/tmp/intellij-community", savePath + "/indiciesOriginCode");
        mutate(treeMutator, emb, originTree, savePath + "/indiciesMutatedCode");
        if(repository != null)
            repository.removeRepo();
        else
            try {
                FileUtils.deleteDirectory(dir);
            } catch (IOException ex){
                ex.printStackTrace();
            }

        repository = new Repository("/tmp/netbeans", "https://github.com/apache/incubator-netbeans.git");
        evaluate(treeMutator, emb, "/tmp/netbeans", savePath + "/indiciesNonClone");
        repository.removeRepo();


    }

    private static List<ASTEntry> evaluate(TreeMutator treeMutator, Embedding emb, String repoPath, String fileName){
        List<ASTEntry> tree = treeMutator.analyzeDir(repoPath);
        emb.createEmbedding(tree, fileName);
        return tree;
    }

    private static void mutate(TreeMutator treeMutator, Embedding emb, List<ASTEntry> originTree, String path){
        List<ASTEntry> mutatedTree = treeMutator.treeMutator(originTree);
        emb.createEmbedding(mutatedTree, path);
    }

    private static List<String> getAllAvailableTokens() {
        return generator.getAllAvailableTokens();
    }

    public static String parsePSIText(String filename) {
        return generator.parsePSIText(filename);
    }

    public static ASTEntry buildPSI(String filename) {
        return generator.parseFile(filename);
    }

    private static void pythonExec(String pythonCode, String args){
        try {
            Runtime rt = Runtime.getRuntime();
            String[] cmds = {"python", pythonCode, args};
            Process proc = rt.exec(cmds);

            BufferedReader stdInput = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            BufferedReader stdError = new BufferedReader(new InputStreamReader(proc.getErrorStream()));

            System.out.println("Output:");
            String s = null;

            while ((s = stdInput.readLine()) != null)
                System.out.println(s);

            while ((s=stdError.readLine()) != null)
                System.out.println(s);

        } catch (IOException ex){
            ex.printStackTrace();
        }
    }
}
