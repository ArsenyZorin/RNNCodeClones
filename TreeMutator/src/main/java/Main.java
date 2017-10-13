//import py4j.GatewayServer;
//import sun.rmi.server.ActivatableServerRef;

//import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

//import static py4j.GatewayServer.DEFAULT_PORT;

public class Main {
    private final static PsiGen generator = new PsiGen();
    private final static List<String> spaces = Arrays.asList(
            "DOC_", "COMMENT", "PACKAGE", "IMPORT",
            "SPACE", "IMPLEMENTS", "EXTENDS", "THROWS",
            "PARAMETER_LIST");

    public static void main(String[] args) {
        //int port = DEFAULT_PORT;
        if(args.length < 1)
            return;
        System.out.println(args[0]);
        String repoPath = args[0];
        //GatewayServer gatewayServer = new GatewayServer(new Main(), port);
        //gatewayServer.start();
        //System.out.println("Gateway Server Started " + port);

        List<String> whiteList = getAllAvailableTokens();
        List<String> blackList = whiteList.stream()
                .filter(p->contains(spaces, p)).collect(Collectors.toList());

        TreeMutator treeMutator = new TreeMutator(generator, blackList, whiteList);
        whiteList.removeAll(blackList);
        Embedding emb = new Embedding(treeMutator);

        System.out.println("Start analyzing repo : " + repoPath);
        List<ASTEntry> originTree = treeMutator.analyzeDir(repoPath);
        emb.createEmbedding(originTree, "OriginCode");

        System.out.println("Start tree mutation:");
        List<ASTEntry> mutatedTree = treeMutator.treeMutator(originTree);

        emb.createEmbedding(mutatedTree, "MutatedCode");

        System.out.println("NonClone Methods");
        List<ASTEntry> nonClone = treeMutator
                .analyzeDir("/home/arseny/mediahdd/Repos/deeplearning4j");
        emb.createEmbedding(nonClone, "NonClone");



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
}
