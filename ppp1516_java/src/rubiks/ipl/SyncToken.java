package rubiks.ipl;
import ibis.ipl.*;

public class SyncToken {
    boolean tokenComeBack = false;
    Token receivedToken;

    public Token waitToken() {
        while(!tokenComeBack) {
            wait();
        }
        tokenComeBack = false;
        return receivedToken;
    }

    public void arrivedToken(Token t) {
        receivedToken = t;
        tokenComeBack = true;
        notifyAll();
    }
}


