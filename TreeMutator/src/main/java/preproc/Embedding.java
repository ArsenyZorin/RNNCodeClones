package preproc;

import arguments.EvalType;
import com.google.gson.Gson;
import gitrepos.Repository;
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
    String workingDir;
    File ideaRepo;

    public Embedding(TreeMutator treeMutator, String evalType, String outputDir) {
        this.treeMutator = treeMutator;
        this.workingDir = outputDir + "/networks/word2vec";

        if(EvalType.FULL.toString().toUpperCase().equals(evalType.toUpperCase()))
            train();
        else {
            File mainEmb = new File(workingDir + "/word2Vec");
            if (mainEmb.exists()) {
                System.out.println("Tokens file was found. Reading values from it");
                mainVec = WordVectorSerializer.readWord2VecModel(mainEmb);
            } else {
                train();
            }
        }
    }

    public File getIdeaRepo() {
        return ideaRepo;
    }

    public void train() {
        //Repository repository = new Repository("/tmp/intellij-community",
        //        "https://github.com/JetBrains/intellij-community.git");
        Repository repository = new Repository("/tmp/intellij-community");
        ideaRepo = repository.getRepoFile();
        System.out.println("Additional analysis : " + ideaRepo.getAbsolutePath());
        List<ASTEntry> tree = treeMutator.analyzeDir(ideaRepo.getAbsolutePath());
        List<String> treeTokens = new ArrayList<>();

        for (ASTEntry token : tree) {
            treeTokens.add(token.getAllTokensString());
            System.out.print("\rTokensString: " + tree.indexOf(token) + "/" + tree.size());
        }

        TokenizerFactory t = new DefaultTokenizerFactory();
        SentenceIterator iter = new CollectionSentenceIterator(treeTokens);
        System.out.println("\nBuilding model...");
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

    public void createEmbedding(List<ASTEntry> codeTokens, String savePath) {
        System.out.println("Embedding creation started");
        List<List<Integer>> allIndexes = new ArrayList<>();

        for (ASTEntry tokenList : codeTokens) {
            List<Integer> tokenIndexes = new ArrayList<>();
            for (String token : tokenList.getAllTokensList())
                tokenIndexes.add(mainVec.indexOf(token));
            allIndexes.add(tokenIndexes);
            System.out.print(String.format("\rEmbedding creation: %s/%s", codeTokens.indexOf(tokenList) + 1, codeTokens.size()));
        }
        System.out.println();
        gsonSerialization(allIndexes, savePath);
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