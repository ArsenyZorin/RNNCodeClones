/**
 * Created by sobol on 3/15/17.
 */

import py4j.GatewayServer;
import sun.rmi.server.ActivatableServerRef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static py4j.GatewayServer.DEFAULT_PORT;

public class Main {
    private final static PsiGen generator = new PsiGen();
    private final static List<String> spaces = Arrays.asList(
            "DOC_", "COMMENT", "PACKAGE", "IMPORT",
            "SPACE", "IMPLEMENTS", "EXTENDS", "THROWS",
            "PARAMETER_LIST");

    public static void main(String[] args) {
        int port = DEFAULT_PORT;
        if(args.length < 1)
            return;
        System.out.println(args[0]);
        String repoPath = args[0];
        GatewayServer gatewayServer = new GatewayServer(new Main(), port);
        gatewayServer.start();
        System.out.println("Gateway Server Started " + port);

        List<String> whiteList = getAllAvailableTokens();
        List<String> blackList = whiteList.stream()
                .filter(p->contains(spaces, p)).collect(Collectors.toList());

        System.out.println("Start analyzing repo : " + repoPath);
        TreeMutator treeMutator = new TreeMutator(generator, blackList, whiteList);
        List<ASTEntry> originTree = treeMutator.analyzeDir(repoPath);

        System.out.println("Start tree mutation:");
        List<ASTEntry> mutatedTree = treeMutator.treeMutator(originTree);

    }

    public Main getMain() {
        return this;
    }

    /*private static List<ASTEntry> analyzeDir(String repoPath, List<String> blackList) {

        return analyzer.analyzeDir(repoPath);
    }

    private static List<ASTEntry> mutateTree(List<ASTEntry> tree) {
        System.out.println("Start tree mutation:");
        TreeMutator mutator = new TreeMutator(generator);
        return mutator.treeMutator(tree);
    }

    public static List<String> getBlackList(){
        return getAllAvailableTokens().stream()
                .filter(p->contains(spaces, p)).collect(Collectors.toList());
    }*/

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
}
