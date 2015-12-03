public class SyncToken {
    boolean tokenComeBack = false;
    Token receivedToken;

    public Token waitToken() {
        while(!tokenComeBack) {
            this.wait();
        }
        tokenComeBack = false;
    }

    public void arrivedToken(Token t) {
        receivedToken = t;
        tokenComeBack = true;
        notifyAll();
    }
}


