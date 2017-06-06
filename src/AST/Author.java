package AST;

/**
 * Created by arseny on 26.05.17.
 */
public class Author {
    private String name;
    private String email;

    public Author(){}

    public Author(String name, String email){
        this.name = name;
        this.email = email;
    }

    public void setName(String name){ this.name = name; }
    public void setEmail(String email) { this.email = email; }

    public String getName(){ return this.name; }
    public String getEmail(){ return this.email; }

    public boolean equals(Author author){
        return this.name.equals(author.name) || this.email.equals(author.email);
    }

    public String toString(){
        return this.name + " <" + this.email + ">";
    }
}
