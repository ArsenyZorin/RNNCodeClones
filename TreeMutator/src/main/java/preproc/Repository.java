package preproc;

import org.apache.commons.io.FileUtils;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.lib.TextProgressMonitor;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

public class Repository {
    String path;
    String uri;

    public Repository(String path, String uri){
        this.path = path;
        this.uri = uri;
        cloneRepo();
    }

    private void cloneRepo(){
        File dir = new File(path);
        try {
            if (dir.exists())
                FileUtils.deleteDirectory(dir);

            System.out.println("Clonning repo from " + uri);
            Git.cloneRepository()
                    .setProgressMonitor(new TextProgressMonitor(new PrintWriter(System.out)))
                    .setURI(uri)
                    .setDirectory(dir)
                    .call();

        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public File getRepoFile(){
        File repo = new File(path);
        if(repo.exists())
            return repo;

        return null;
    }

    public void removeRepo(){
        File dir = new File(this.path);
        try {
            if(dir.exists())
                FileUtils.deleteDirectory(dir);
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }

}
