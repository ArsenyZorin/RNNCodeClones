import org.deeplearning4j.models.word2vec.Word2Vec;
import org.bytedeco.javacv.FrameFilter;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Embedding {
    Word2Vec  mainVec;
    TreeMutator treeMutator;

    public Embedding(TreeMutator treeMutator) {
        String workingDir = System.getProperty("user.dir");
        File mainEmb = new File(workingDir + "/word2Vec");
        if (mainEmb.exists()) {
            System.out.println("Tokens file was found. Reading values from it");
            mainVec = WordVectorSerializer.readWord2VecModel(mainEmb);
            System.out.println("Successful");
        } else {
            this.treeMutator = treeMutator;
            String repoPath = "/home/arseny/Repos/intellij-community";
            System.out.println("Additional analysis : " + repoPath);
            List<ASTEntry> originTree = treeMutator.analyzeDir(repoPath);

            List<String> treeTokens = new ArrayList<>();
            int i = 0;
            for (ASTEntry tree : originTree) {
                treeTokens.add(tree.getAllTokensString());
                System.out.println("TokensString: " + (++i) + "/" + originTree.size());
            }

            System.out.println(treeTokens);
            TokenizerFactory t = new DefaultTokenizerFactory();
            SentenceIterator iter = new CollectionSentenceIterator(treeTokens);
            System.out.println("Building model...");
            Word2Vec vec = new Word2Vec.Builder()
                    .minWordFrequency(1)
                    .iterations(1)
                    .layerSize(100)
                    .seed(42)
                    .windowSize(5)
                    .iterate(iter)
                    .tokenizerFactory(t)
                    .build();

            System.out.println("Fitting Word2Vec model...");

            vec.fit();

            try {
                WordVectorSerializer.writeWordVectors(vec, workingDir + "/word2Vec");
            } catch (Exception ex) {
                System.out.println(ex.toString());
            }

            System.out.println("ADDITIONAL ANALYSIS COMPLETE");
        }
    }

    public void createEmbedding(List<ASTEntry> codeTokens) {
        System.out.println("Embedding creation starts");
        System.out.println("Embedding creation ends");
    }
}
