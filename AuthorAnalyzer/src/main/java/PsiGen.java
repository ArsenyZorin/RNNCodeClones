import com.intellij.codeInsight.ContainerProvider;
import com.intellij.codeInsight.JavaContainerProvider;
import com.intellij.core.CoreApplicationEnvironment;
import com.intellij.core.CoreProjectEnvironment;
import com.intellij.core.JavaCoreApplicationEnvironment;
import com.intellij.core.JavaCoreProjectEnvironment;
import com.intellij.lang.ASTNode;
import com.intellij.lang.FileASTNode;
import com.intellij.lang.java.JavaLanguage;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.extensions.Extensions;
import com.intellij.psi.PsiElementFinder;
import com.intellij.psi.PsiFile;
import com.intellij.psi.PsiFileFactory;
import com.intellij.psi.PsiNameHelper;
import com.intellij.psi.impl.PsiElementFinderImpl;
import com.intellij.psi.impl.PsiModificationTrackerImpl;
import com.intellij.psi.impl.PsiNameHelperImpl;
import com.intellij.psi.impl.PsiTreeChangePreprocessor;
import com.intellij.psi.impl.source.tree.ElementType;
import com.intellij.psi.tree.IElementType;
import com.intellij.psi.tree.TokenSet;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PsiGen {
    private final PsiFileFactory fileFactory;

    PsiGen() {
        Extensions.getRootArea().registerExtensionPoint(
                ContainerProvider.EP_NAME.getName(),
                JavaContainerProvider.class.getCanonicalName()
        );

        final CoreApplicationEnvironment appEnv = new JavaCoreApplicationEnvironment(() -> {
        });

        final CoreProjectEnvironment prjEnv = new JavaCoreProjectEnvironment(() -> {
        }, appEnv) {
            @Override
            protected void preregisterServices() {
                Extensions.getArea(myProject).registerExtensionPoint(
                        PsiTreeChangePreprocessor.EP_NAME.getName(),
                        PsiModificationTrackerImpl.class.getCanonicalName()
                );
                Extensions.getArea(myProject).registerExtensionPoint(
                        PsiElementFinder.EP_NAME.getName(),
                        PsiElementFinderImpl.class.getCanonicalName()
                );
                myProject.registerService(
                        PsiNameHelper.class,
                        PsiNameHelperImpl.class
                );
            }
        };

        this.fileFactory = PsiFileFactory.getInstance(prjEnv.getProject());

    }

    private ASTEntry convertSubtree(ASTNode node, ASTEntry parent, Document doc) {
        ASTEntry rootEntry = new ASTEntry(node, parent, doc);
        for (ASTNode child : node.getChildren(null)) {
            ASTEntry entry = convertSubtree(child, rootEntry, doc);
            rootEntry.children.add(entry);
        }
        return rootEntry;
    }

    private ASTEntry convert(PsiFile file) {
        Document doc = file.getViewProvider().getDocument();
        FileASTNode startNode = file.getNode();
        return convertSubtree(startNode, null, doc);
    }

    private List<String> elemArrayToString(IElementType[] et) {
        return Stream.of(et).map(IElementType::toString).collect(Collectors.toList());
    }

    private void checkTokens(List<String> tokens, ASTEntry node) {

        if (!tokens.contains(node.nodeName)) {
            System.out.println(node.nodeName);
        }
        for (ASTEntry child : node.children) {
            checkTokens(tokens, child);
        }

    }

    private void fillFile(ASTEntry root, Map<Integer, String> file) {
        file.merge(root.sourceStart, root.nodeName, (v, s) -> v.concat(" " + s));
        for (ASTEntry child : root.children) {
            fillFile(child, file);
        }
    }

    public void printFile(Integer line, String value) {
        System.out.println(line + ": " + value);
    }

    private String getASTText(ASTEntry root) {
        Map<Integer, String> file = new HashMap<>();
        for (ASTEntry child : root.children) {
            fillFile(child, file);
        }
        return file.entrySet().stream().map(Map.Entry::getValue).collect(Collectors.joining(" "));
    }


    private Object getField(Field f) {
        try {
            return f.get(null);
        } catch (NullPointerException | IllegalAccessException ex) {
            ex.printStackTrace();
        }
        return null;
    }

    private PsiFile parse(final String sourceCode) {
        return fileFactory.createFileFromText(JavaLanguage.INSTANCE, sourceCode);
    }


    List<String> getAllAvailableTokens() {
        Set<String> tokens = new HashSet<>();
        for (Field f : ElementType.class.getDeclaredFields())
            tokens.addAll(elemArrayToString(((TokenSet) getField(f)).getTypes()));
        for (Class cl : ElementType.class.getInterfaces()) {
            for (Field f : cl.getDeclaredFields()) {
                Object value = getField(f);
                if (value instanceof TokenSet)
                    tokens.addAll(elemArrayToString(((TokenSet) value).getTypes()));
                else
                    tokens.add(value.toString());
            }
        }
        tokens.add("java.FILE");
        return new ArrayList<>(tokens);
    }

    String parsePSIText(String filename) {
        ASTEntry root = parseFile(filename);
        return getASTText(root);
    }

    ASTEntry parseFile(String filename) {
        try {
            return convert(parse(String.join("\n", Files.readAllLines(Paths.get(filename)))));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }


}
