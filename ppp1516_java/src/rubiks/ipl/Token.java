package rubiks.ipl;
import ibis.ipl.*;
import java.io.Serializable;

public class Token implements Serializable {
    public int id;
    public boolean white = true;
    public Token(int id) {
        this.id = id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public void setWhite(boolean white) {
        this.white = white;
    }

    public int getId() {
        return id;
    }

    public boolean getWhite() {
        return white;
    }
}