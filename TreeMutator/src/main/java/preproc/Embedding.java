package preproc;

import arguments.EvalType;
import com.google.gson.Gson;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import trees.ASTEntry;


import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Embedding {
    Word2Vec mainVec;
    TreeMutator treeMutator;
    String workingDir = System.getProperty("user.dir");

    public Embedding(TreeMutator treeMutator, String evalType, String outputDir) {
        this.treeMutator = treeMutator;
        if(outputDir != null)
            this.workingDir = outputDir;
        File mainEmb = new File(workingDir + "/word2Vec");
        if(EvalType.EVAL.toString().toUpperCase().equals(evalType.toUpperCase())){
            if (mainEmb.exists()) {
                System.out.println("Tokens file was found. Reading values from it");
                mainVec = WordVectorSerializer.readWord2VecModel(mainEmb);
                System.out.println("Successful");
            } else {
                train();
            }
        } else
            train();
    }

    public void train() {
        File cloneDir = new File("/tmp/intellij-community/");
        /*try {
            if (cloneDir.exists())
                FileUtils.deleteDirectory(cloneDir);

            System.out.println("Clonning intellij-community repo");
            Git.cloneRepository()
                    .setProgressMonitor(new TextProgressMonitor(new PrintWriter(System.out)))
                    .setURI("https://github.com/JetBrains/intellij-community.git")
                    .setDirectory(cloneDir)
                    .call();

        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
*/
        System.out.println("Additional analysis : " + cloneDir.getAbsolutePath());
        List<ASTEntry> tree = treeMutator.analyzeDir(cloneDir.getAbsolutePath());
        List<String> treeTokens = new ArrayList<>();

        for (ASTEntry token : tree) {
            treeTokens.add(token.getAllTokensString());
            System.out.print("\rTokensString: " + tree.indexOf(token) + "/" + tree.size());
        }

        System.out.println(treeTokens);
        TokenizerFactory t = new DefaultTokenizerFactory();
        SentenceIterator iter = new CollectionSentenceIterator(treeTokens);
        System.out.println("Building model...");
        mainVec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        System.out.println("Fitting Word2Vec model...");

        mainVec.fit();

        /*try {
            FileUtils.deleteDirectory(cloneDir);
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }*/

        try {
            WordVectorSerializer.writeWord2VecModel(mainVec, workingDir + "/word2Vec");
            gsonSerialization(mainVec.getLookupTable().getWeights(), workingDir + "/tokensWeight");
            ArrayList<double[]> weights = new ArrayList<>();
            for (int j = 0; j < mainVec.getVocab().numWords(); j++)
                weights.add(mainVec.getWordVector(mainVec.getVocab().wordAtIndex(j)));

            gsonSerialization(weights, workingDir + "/pretrainedWeights");
        } catch (Exception ex) {
            System.out.println(ex.toString());
        }

        System.out.println("ADDITIONAL ANALYSIS COMPLETE");
    }

    public void createEmbedding(List<ASTEntry> codeTokens, String embeddingTree) {
        System.out.println("Embedding creation started");
        List<List<Integer>> allIndexes = new ArrayList<>();

        for (ASTEntry tokenList : codeTokens) {
            List<Integer> tokenIndexes = new ArrayList<>();
            for (String token : tokenList.getAllTokensList())
                tokenIndexes.add(mainVec.indexOf(token));
            allIndexes.add(tokenIndexes);
            System.out.print("\r" + codeTokens.indexOf(tokenList) + "/" + codeTokens.size());
        }

        gsonSerialization(allIndexes, workingDir + "/indicies" + embeddingTree);
        System.out.println("Embedding created");
    }

    private void gsonSerialization(Object obj, String path) {
        Gson gson = new Gson();

        FileWriter fw = null;
        BufferedWriter bw = null;
        try {
            fw = new FileWriter(path);
            bw = new BufferedWriter(fw);
            bw.write(gson.toJson(obj));
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null)
                    bw.close();
                if (fw != null)
                    fw.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }
}