import com.google.gson.Gson;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.bytedeco.javacv.FrameFilter;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

public class Embedding {
    Word2Vec  mainVec;
    TreeMutator treeMutator;
    String workingDir = System.getProperty("user.dir");

    public Embedding(TreeMutator treeMutator) {
        File mainEmb = new File(workingDir + "/word2Vec");
        if (mainEmb.exists()) {
            System.out.println("Tokens file was found. Reading values from it");
            mainVec = WordVectorSerializer.readWord2VecModel(mainEmb);
            System.out.println("Successful");
        } else {
            this.treeMutator = treeMutator;
            String repoPath = "/home/arseny/Repos/intellij-community";
            System.out.println("Additional analysis : " + repoPath);
            List<ASTEntry> tree = treeMutator.analyzeDir(repoPath);
            List<String> treeTokens = new ArrayList<>();

            int i = 0;
            for (ASTEntry token : tree){
                treeTokens.add(token.getAllTokensString());
                System.out.println("TokensString: " + (++i) + "/" + tree.size());
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

            try {
                WordVectorSerializer.writeWord2VecModel(mainVec, workingDir + "/word2Vec");
            } catch (Exception ex) {
                System.out.println(ex.toString());
            }

            System.out.println("ADDITIONAL ANALYSIS COMPLETE");
        }
    }

    public void createEmbedding(List<ASTEntry> codeTokens, String embeddingTree) {
        System.out.println("Embedding creation started");

        List<List<double[]>> embeddings = new ArrayList<>();

        for (ASTEntry tokenList : codeTokens) {
            List<double[]> embed = new ArrayList<>();
            for (String token : tokenList.getAllTokensList())
                embed.add(mainVec.getWordVector(token));
            embeddings.add(embed);
        }

        Gson gson = new Gson();

        FileWriter fw = null;
        BufferedWriter bw = null;
        try {
            fw = new FileWriter(workingDir + "/embedding" + embeddingTree);
            bw = new BufferedWriter(fw);
            bw.write(gson.toJson(embeddings));
        }catch (Exception e){
            e.printStackTrace();
        }finally {
            try {
                if (bw != null)
                    bw.close();
                if (fw != null)
                    fw.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }

        System.out.println("Embedding created");
    }
}
