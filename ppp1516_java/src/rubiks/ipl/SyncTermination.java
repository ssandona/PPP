package rubiks.ipl;
import ibis.ipl.*;

public class SyncTermination {
    int busyWorkers=0;
    int receivedResults=0;

    //method called by the Server when it finish its work to wait the termination of all the Slaves
    synchronized public int waitTermination() throws InterruptedException{
        while(busyWorkers!=0) {
            wait();
        }
        int res=receivedResults;
        receivedResults=0;
        return res;
    }

    //method called by the Server to increase the result with its results or with the results of the Slaves
    synchronized public void increaseResults(int res) throws InterruptedException{
        receivedResults+=res;
        decreaseBusyWorkers();
    }

    //method called when some work is given to a Slave
    synchronized public void increaseBusyWorkers() throws InterruptedException{
        busyWorkers+=1;
        System.out.println("BUSY WORKERS: "+busyWorkers);
    }

    //method called when a new result is received, so the number of workers decrease
    public void decreaseBusyWorkers() throws InterruptedException{
        busyWorkers-=1;
        System.out.println("BUSY WORKERS: "+busyWorkers);
        if(busyWorkers==0){
            notifyAll();
        }
    }
}


