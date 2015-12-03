package rubiks.ipl;
import ibis.ipl.*;

public class SyncToken {
    boolean tokenComeBack = false;
    Token receivedToken;

    synchronized public Token waitToken() throws InterruptedException{
        while(!tokenComeBack) {
            wait();
        }
        tokenComeBack = false;
        return receivedToken;
    }

    synchronized public void arrivedToken(Token t) {
        receivedToken = t;
        tokenComeBack = true;
        notifyAll();
    }
}


