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
import java.util.Arrays;
import java.util.List;
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

        List<String> white_list = getAllAvailableTokens();
        List<String> black_list = white_list.stream()
                .filter(Main::contains).collect(Collectors.toList());

        TreeMutator tree_mutator = new TreeMutator(generator, black_list, white_list);
        white_list.removeAll(black_list);
        Embedding emb = new Embedding(tree_mutator, args.getEvalType(), args.getOutputDir());
        String save_file = args.getOutputDir() + "/vectors";
        String repo_path = args.getInputDir();

        if (EvalType.MUTATE.toString().equals(args.getEvalType().toUpperCase())) {
            System.out.println("Directory for mutation: " + args.getInputDir());
            mutate(tree_mutator, emb,
                    evaluate(tree_mutator, emb, repo_path, save_file + "/EvalCode"),
                    args.getOutputDir() + "/EvalMutatedCode");
            evaluate(tree_mutator, emb, repo_path, save_file + "/EvalNonClone");

        } else if (EvalType.EVAL.toString().equals(args.getEvalType().toUpperCase())) {
            System.out.println("Start analyzing repo : " + repo_path);
            evaluate(tree_mutator, emb, repo_path, save_file + "/originCode");
        } else if (EvalType.TRAIN.toString().equals(args.getEvalType().toUpperCase())) {
            train(tree_mutator, emb, save_file);
        } else {
            System.out.println("Directory for analysis: " + args.getInputDir());
            train(tree_mutator, emb, save_file);

            List<ASTEntry> tree = evaluate(tree_mutator, emb, repo_path, save_file + "/originCode");
            //mutate(tree_mutator, emb, tree, save_file + "/EvalMutatedCode");
            //evaluate(tree_mutator, emb, "/home/arseny/evals/jdbc", save_file + "/EvalNonClone");
        }

        // String path = Main.class.getResource("/clonesRecognition.py").getPath();
        // System.out.println(path);

        // String pythonArgs = "--type full --data " + args.getOutputDir();
        // pythonExec("/home/arseny/Repos/RNNCodeClones/Networks/clonesRecognition.py", pythonArgs);
    }

    public Main getMain() {
        return this;
    }

    private static boolean contains(String nodeName){
        for(String black_elem : Main.spaces)
            if(nodeName.contains(black_elem))
                return true;

        return false;
    }

    private static void train(TreeMutator tree_mutator, Embedding emb, String save_path){
        File dir = emb.getIdeaRepo();
        Repository repository = null;
        if(dir == null)
            repository = new Repository("/tmp/intellij-community", "https://siemens.spbpu.com/arseny/intellij-community.git");
        List<ASTEntry> origin_tree = evaluate(tree_mutator, emb, /*"/tmp/intellij-community"*/"/tmp/w2v_train", save_path + "/indiciesOriginCode");
        mutate(tree_mutator, emb, origin_tree, save_path + "/indiciesMutatedCode");
        if(repository != null)
            repository.removeRepo();
        else
            try {
                FileUtils.deleteDirectory(dir);
            } catch (IOException ex){
                ex.printStackTrace();
            }

        repository = new Repository("/tmp/netbeans", "https://siemens.spbpu.com/arseny/incubator-netbeans.git");
        evaluate(tree_mutator, emb, "/tmp/netbeans", save_path + "/indiciesNonClone");
        repository.removeRepo();
    }

    private static List<ASTEntry> evaluate(TreeMutator tree_mutator, Embedding emb, String repo_path, String file_name){
        List<ASTEntry> tree = tree_mutator.analyzeDir(repo_path);
        emb.createEmbedding(tree, file_name);
        return tree;
    }

    private static void mutate(TreeMutator tree_mutator, Embedding emb, List<ASTEntry> origin_tree, String path){
        List<ASTEntry> mutated_tree = tree_mutator.treeMutator(origin_tree);
        emb.createEmbedding(mutated_tree, path);
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
