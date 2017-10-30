package preproc;

import org.apache.commons.io.FileUtils;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.lib.TextProgressMonitor;
import org.eclipse.jgit.transport.HttpTransport;
import org.eclipse.jgit.transport.http.HttpConnectionFactory;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URL;

public class Repository {
    String path;
    String url;

    public Repository(String path, String url){
        this.path = path;
        this.url = url;
        cloneRepo();
    }

    private void cloneRepo(){
        File dir = new File(path);
        try {
            if (dir.exists())
                FileUtils.deleteDirectory(dir);

            HttpConnectionFactory preservedConnectionFactory = HttpTransport.getConnectionFactory();
            HttpTransport.setConnectionFactory( new InsecureHttpConnectionFactory() );
            System.out.println("Clonning repo from " + url);
            //InsecureHttpConnectionFactory repo = new InsecureHttpConnectionFactory();
            //repo.create(new URL(this.url));
            Git.cloneRepository()
                    .setProgressMonitor(new TextProgressMonitor(new PrintWriter(System.out)))
                    .setURI(this.url)
                    .setDirectory(dir)
                    .call();
            HttpTransport.setConnectionFactory( preservedConnectionFactory );

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(e.getMessage());
            System.out.println(e.getCause());
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
