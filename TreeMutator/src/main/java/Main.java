import arguments.Arguments;
import arguments.EvalType;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;
import org.bytedeco.javacpp.presets.opencv_core;
import preproc.Embedding;
import preproc.TreeMutator;
import trees.ASTEntry;
import trees.PsiGen;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
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
        if(argv.length < 1)
            return;

        Arguments args = new Arguments();
        try {
            JCommander.newBuilder().addObject(args).build().parse(argv);
        }catch (ParameterException ex){
            System.out.println(ex.getMessage());
            System.out.println("For additional info type: --help or -h");
            return;
        }

        System.out.println("Analyzed dir: " + args.getInputDir());
        String repoPath = args.getInputDir();

        List<String> whiteList = getAllAvailableTokens();
        List<String> blackList = whiteList.stream()
                .filter(p->contains(spaces, p)).collect(Collectors.toList());

        TreeMutator treeMutator = new TreeMutator(generator, blackList, whiteList);
        whiteList.removeAll(blackList);
        Embedding emb = new Embedding(treeMutator, args.getEvalType(), args.getOutputDir());

        if(!EvalType.MUTATE.toString().equals(args.getEvalType().toUpperCase())) {
            System.out.println("Start analyzing repo : " + repoPath);
            List<ASTEntry> originTree = treeMutator.analyzeDir(repoPath);
            emb.createEmbedding(originTree, args.getOutputDir() + "/indiciesOriginCode");

            if (!EvalType.EVAL.toString().equals(args.getEvalType().toUpperCase())) {
                System.out.println("Start tree mutation:");
                List<ASTEntry> mutatedTree = treeMutator.treeMutator(originTree);
                emb.createEmbedding(mutatedTree, args.getOutputDir() + "/indiciesMutatedCode");

                System.out.println("NonClone Methods");
                List<ASTEntry> nonClone = treeMutator
                        .analyzeDir("/home/arseny/deeplearning4j");
                emb.createEmbedding(nonClone, args.getOutputDir() + "/indiciesNonClone");
            }
        } else {
            System.out.println("Start analyzing repo : " + repoPath);
            List<ASTEntry> originTree = treeMutator.analyzeDir(repoPath);
            emb.createEmbedding(originTree, args.getOutputDir() + "/EvalCode");

            System.out.println("Start tree mutation:");
            List<ASTEntry> mutatedTree = treeMutator.treeMutator(originTree);
            emb.createEmbedding(mutatedTree, args.getOutputDir() +"/EvalMutatedCode");

            System.out.println("NonClone Methods");
            List<ASTEntry> nonClone = treeMutator
                    .analyzeDir("/home/arseny/evals/jdbc");
            emb.createEmbedding(nonClone, args.getOutputDir() + "/EvalNonClone");
        }

        pythonExec("../Autoencoders/clonesRecognition.py", args.getOutputDir());

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
            String[] cmds = {"python", pythonCode, args}; //"/home/arseny/Repos/RNNCodeClones/Autoencoders/seq2seq.py", "/home/arseny/Repos/RNNCodeClones/data/"};
            /*List<String> cmdss = new ArrayList<>();
            cmdss.add("python");
            cmdss.addAll(args);

            String[] cmds = new String[cmdss.size()];
            cmds = cmdss.toArray(cmds);*/

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
