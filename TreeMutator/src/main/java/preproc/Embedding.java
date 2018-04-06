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

    /***
     * Method for word2vec model training.
     *
     * Uses openjdk from siemens.spbpu.com for training model
     */
    public void train() {
        Repository repository = new Repository("/tmp/intellij-community",
                "https://github.com/JetBrains/intellij-community.git");
        ideaRepo = repository.getRepoFile();
        System.out.println("Additional analysis : " + ideaRepo.getAbsolutePath());
        List<ASTEntry> tree = treeMutator.analyzeDir(ideaRepo.getAbsolutePath());
        //List<String> treeTokens = new ArrayList<>();
        AbstractMap<String, String> treeTokens = new HashMap<>();

        for (ASTEntry token : tree) {
            String ident = String.format("Path: %s Start: %d End: %d", token.filePath, token.sourceStart, token.sourceEnd);
            treeTokens.put(ident, token.getAllTokensString());
            //treeTokens.add(token.getAllTokensString());
            System.out.print("\rTokensString: " + tree.indexOf(token) + "/" + tree.size());
        }

        TokenizerFactory t = new DefaultTokenizerFactory();
        SentenceIterator iter = new CollectionSentenceIterator(treeTokens.values());
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

    /***
     * Creates vector representations of tokens based on word2vec model
     *
     * @param code_tokens Dict of tokens for embeddings creation
     * @param save_path Path where to save embeddings
     */
    public void createEmbedding(List<ASTEntry> code_tokens, String save_path) {
        System.out.println("Embedding creation started");
        //List<List<Integer>> allIndexes = new ArrayList<>();
        AbstractMap<String, List<Integer>> all_indices = new HashMap<>();

        for (ASTEntry token_list : code_tokens) {
            List<Integer> tokenIndexes = new ArrayList<>();
            for (String token : token_list.getAllTokensList())
                tokenIndexes.add(mainVec.indexOf(token));
            String ident = String.format("Path: %s Start: %d End: %d", token_list.filePath, token_list.sourceStart, token_list.sourceEnd);
            all_indices.put(ident, tokenIndexes);
            //allIndexes.add(tokenIndexes);
            System.out.print(String.format("\rEmbedding creation: %s/%s", code_tokens.indexOf(token_list) + 1, code_tokens.size()));
        }
        System.out.println();
        gsonSerialization(all_indices , save_path);
    }

    /***
     * Method for JSON serialization of object
     *
     * @param obj Object for serialization
     * @param path Path where to save file
     */
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