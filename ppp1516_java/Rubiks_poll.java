package rubiks.ipl;

import ibis.ipl.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.IOException;

/**
 * Solver for rubik's cube puzzle.
 *
 * @author Niels Drost, Timo van Kessel
 *
 */
//public class Rubiks implements RegistryEventHandler {
public class Rubiks {

    static PortType portType1toM = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT,
            PortType.CONNECTION_ONE_TO_MANY);

    static PortType portTypeMto1 = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT,
            PortType.CONNECTION_MANY_TO_ONE);

    static PortType portTypeMto1Poll = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.RECEIVE_POLL,
            PortType.SERIALIZATION_OBJECT, PortType.CONNECTION_MANY_TO_ONE, PortType.RECEIVE_EXPLICIT, PortType.RECEIVE_AUTO_UPCALLS);

    static PortType portType1to1 = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT,
            PortType.CONNECTION_ONE_TO_ONE);

    static PortType portType1to1Up = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.CONNECTION_ONE_TO_ONE, PortType.RECEIVE_AUTO_UPCALLS);

    static PortType requestWorkPortType = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_AUTO_UPCALLS,
            PortType.CONNECTION_ONE_TO_ONE);




    static IbisCapabilities ibisCapabilities = new IbisCapabilities(
        IbisCapabilities.ELECTIONS_STRICT, IbisCapabilities.MEMBERSHIP_TOTALLY_ORDERED);

    /*static IbisCapabilities ibisCapabilities = new IbisCapabilities(
        IbisCapabilities.ELECTIONS_STRICT);*/

    static int counter = 0;
    //static ArrayList<String> done;
    static int result = 0;
    static int nodes = 1;
    static ArrayList<Cube[]> machines;
    static IbisIdentifier[] joinedIbises;
    static int myIntIbisId;
    static boolean[] white;
    static IbisIdentifier myIbisId;
    static Integer[] cubes_per_proc;
    static Integer[] displs;
    static Ibis myIbis;
    static Cube initialCube = null;
    static int size;
    static int level;
    static int target;

    static int valuatedCubes = 0;
    static int requestsForWork = 0;

    static ReceivePort workRequestReceiver;
    static SendPort workRequestSender;
    static ReceivePort workReceiver;

    static ReceivePort tokenRequestReceiver;
    static SendPort tokenRequestSender;

    static TokenManager tokenManager;
    static WorkManager workManager;


    public static final boolean PRINT_SOLUTION = false;

    static ArrayList<Cube> toDo = new ArrayList<Cube>();
    static int toDoWeight = 0;



    /*static class WorkManager implements MessageUpcall {
        A request of work from another Ibis instance
        public void upcall(ReadMessage message) throws IOException,
            ClassNotFoundException {
            int otherIbisId = message.readInt();
            message.finish();
            System.out.println("Ibis[" + myIntIbisId + "] -> workrequest UPCALL");
            //get ibisIdentifier of the requestor
            IbisIdentifier requestor = joinedIbises[otherIbisId];
            int i;
            // create a sendport for the reply
            //SendPort replyPort = myIbis.createSendPort(portType1to1);
            SendPort replyPort = myIbis.createSendPort(portTypeMto1);
            //System.out.println("Ibis[" + myIntIbisId + "] -> send to requestor");
            // connect to the requestor's receive port
            replyPort.connect(requestor, "Work");

            // send the work to him
            WriteMessage reply = replyPort.newMessage();

            //System.out.println("Ibis[" + myIntIbisId + "] -> pool to send empty");
            reply.writeInt(0);
            reply.finish();

            replyPort.close();
        }
    }*/


    public static void printSize() {
        System.out.println("Ibis[" + myIntIbisId + "] -> SIZE: " + toDo.size());
    }

    public static void add(Cube cube) {
        /*if(cube == null) {
            System.out.println("Ibis[" + myIntIbisId + "] -> AHAHAH 6");
        }*/

        toDo.add(cube);
        toDoWeight += (cube.getBound() - cube.getTwists());



        //System.out.println("Ibis[" + myIntIbisId + "] -> added cube");
    }

    /*receive port that I check number of connections, if 0 then all are terminated*/
    /*when I think that I'm terminated I disconnect from all the other ports*/


    public static boolean askForWork() throws IOException, ClassNotFoundException {
        //System.out.println("Ibis[" + myIntIbisId + "] -> Ask");
        System.out.println("Ibis[" + myIntIbisId + "] -> askForWork");
        int i;
        Cube[] receivedWork = new Cube[0];
        //Cube[] receivedWork;
        IbisIdentifier doner;
        for(i = 0; i < nodes; i++) {
            doner = joinedIbises[target];
            if(!doner.equals(myIbisId)) {
                try {
                    requestsForWork++;
                    workRequestSender.connect(doner, "WorkReq");
                    //System.out.println("Ibis[" + myIntIbisId + "] -> send");
                    WriteMessage task = workRequestSender.newMessage();
                    task.writeInt(myIntIbisId);
                    task.finish();
                    ReadMessage r=null;
                    ReceivePort[] connections;
                    while((connections=workRequestSender.lostConnections())!=null{
                        r=workReceiver.poll();
                    }

                    //System.out.println("Ibis[" + myIntIbisId + "] -> receive");
                    ReadMessage r = workReceiver.receive();
                    int cubes = r.readInt();
                    r.finish();
                    if(cubes == 0) {
                        System.out.println("Ibis[" + myIntIbisId + "] -> no work");
                    } else {
                        r = workReceiver.receive();
                        System.out.println("Ibis[" + myIntIbisId + "] -> work!!! " + cubes + " cubes");
                        receivedWork = new Cube[cubes];
                        r.readArray(receivedWork);
                        r.finish();
                    }
                    //System.out.println("ReceivedMyWork");


                    /*int n=r.readInt();
                    System.out.println("received n");*/

                    workRequestSender.disconnect(doner, "WorkReq");
                    if(receivedWork != null && receivedWork.length != 0) {
                        //System.out.println("Ibis[" + myIntIbisId + "] -> received " + receivedWork.length + " cubes");
                        //toDo = new ArrayList<Cube>(Arrays.asList(receivedWork));
                        int j;
                        for(j = 0; j < receivedWork.length; j++) {
                            Cube c = receivedWork[j];
                            if(c == null) {
                                System.out.println("Ibis[" + myIntIbisId + "] -> AHAHHA 5");
                            }
                            add(c);
                        }
                        return true;
                    }
                } catch(Exception e) {
                    System.out.println("il bastardo si è scollegato");
                }
            }
            target = (target + 1) % nodes;
        }
        return false;
    }

    public static ArrayList<Cube> getFromPool (boolean sameNode) {
        //System.out.println("Ibis[" + myIntIbisId + "] -> getFromPool");
        ArrayList<Cube> workToReturn = new ArrayList<Cube>();
        if(toDo.size() == 0) {
            return null;
        }
        if(sameNode) {
            Cube c = null;
            int n = toDo.size() - 1;
            c = toDo.remove(n);
            toDoWeight -= (c.getBound() - c.getTwists());

            /*if(c == null) {
                System.out.println("Ibis[" + myIntIbisId + "] -> AHAHAHA 1, index -> "+n);
            }*/
            workToReturn.add(c);
        } else {
            /*int amount = toDo.size() / 2;
            boolean even = toDo.size() % 2 == 0;
            int index = even ? toDo.size() / 2 : toDo.size() / 2 + 1;
            int i;*/
            int weightToDistribute = toDoWeight / 2;
            int distributed = 0;
            while(distributed < weightToDistribute) {
                Cube c = toDo.remove(0);
                workToReturn.add(c);
                toDoWeight -= (c.getBound() - c.getTwists());
                distributed += (c.getBound() - c.getTwists());
                /*if(c == null) {
                    System.out.println("Ibis[" + myIntIbisId + "] -> AHAHAHA 2, (amount, even, index) -> (" + amount + "," + even + "," + index + ")");

                }*/
            }
            if(workToReturn.size() == 0) {
                workToReturn = null;
                //System.out.println("Ibis[" + myIntIbisId + "] -> send to the other 0 cubes");
            }


        }
        return workToReturn;
    }

    //function invokable from both actual worker or another one, if invoked by the actual worker
    //and the queue i(toDo) is empty, the function askForWork is invoked (that invoke the workRequest function
    //on the other nodes
    public static ArrayList<Cube> getWork(boolean sameNode) throws IOException, ClassNotFoundException {
        //System.out.println("Ibis[" + myIntIbisId + "] -> getWork");
        //System.out.println("Ibis[" + myIntIbisId + "] -> getWork");
        if(sameNode) {
            if(toDo.size() == 0) {
                // enable upcalls
                workRequestReceiver.enableMessageUpcalls();
                //System.out.println("Ibis[" + myIntIbisId + "] -> toDo Empty");
                boolean b = askForWork();
                if(!b) {
                    return null;
                } else {
                    // enable upcalls
                    workRequestReceiver.disableMessageUpcalls();
                }
            }
        }
        return getFromPool(sameNode);


    }

    /*A request of work from another Ibis instance*/
    public static void workRequestFromOthers(ReadMessage message) throws IOException,
        ClassNotFoundException {
        int otherIbisId = message.readInt();
        message.finish();
        System.out.println("Ibis[" + myIntIbisId + "] -> workrequestFromOther");
        //get ibisIdentifier of the requestor
        IbisIdentifier requestor = joinedIbises[otherIbisId];
        int i;
        // create a sendport for the reply
        //SendPort replyPort = myIbis.createSendPort(portType1to1);
        SendPort replyPort = myIbis.createSendPort(portTypeMto1);

        ArrayList<Cube> subPool;
        Cube[] subPoolToSend = new Cube[0];
        //Cube[] subPoolToSend = null;
        if(toDo.size() == 0) {
            subPool = null;
        } else {
            subPool = getWork(false);
        }
        //update node color in the eyes of every other node
        if(subPool != null) {
            if(myIntIbisId > otherIbisId) {
                for(i = 0; i <= otherIbisId; i++) {
                    white[i] = false;
                }
                for(i = myIntIbisId + 1; i < nodes; i++) {
                    white[i] = false;
                }
            } else {
                for(i = myIntIbisId + 1; i <= otherIbisId; i++) {
                    white[i] = false;
                }
            }
        }

        //System.out.println("Ibis[" + myIntIbisId + "] -> send to requestor");
        // connect to the requestor's receive port
        replyPort.connect(requestor, "Work");

        // send the work to him
        WriteMessage reply = replyPort.newMessage();
        if(subPool != null && subPool.size() != 0) {

            /*for(i = 0; i < subPool.size(); i++) {
                if(subPool.get(i) == null) {
                    System.out.println("Ibis[" + myIntIbisId + "] -> AHAHAHA 3");
                }
            }*/

            subPoolToSend = subPool.toArray(new Cube[subPool.size()]);
            //System.out.println("Ibis[" + myIntIbisId + "] -> pool to send not empty => " + subPoolToSend.length);
            reply.writeInt(subPoolToSend.length);
            reply.finish();
            reply = replyPort.newMessage();
            reply.writeArray(subPoolToSend);
            reply.finish();
        } else {
            //System.out.println("Ibis[" + myIntIbisId + "] -> pool to send empty");
            reply.writeInt(0);
            reply.finish();
        }
        replyPort.close();
    }


    static class TokenManager implements MessageUpcall {

        boolean tokenComeBack = false;
        Token receivedToken;
        static SyncToken sync = new SyncToken();

        public static boolean checkTermination () throws Exception {
            //System.out.println("Ibis[" + myIntIbisId + "] -> checkTermination");
            //create a new token
            Token t = new Token(myIntIbisId);
            //already connected with the next ibis instance
            //send the token to the next ibis instance
            WriteMessage term = tokenRequestSender.newMessage();
            term.writeObject(t);
            term.finish();
            Token receivedToken = null;

            receivedToken = sync.waitToken();
            return receivedToken.white;

            //wait the token comes back
            /*boolean myToken = false;
            while(!myToken) {
                ReadMessage r = tokenRequestReceiver.receive();
                t = (Token)r.readObject();
                r.finish();
                //if the received token is the one expected, its value is returned
                if(t.id == myIntIbisId) {
                    return t.white;
                }
                //if the received token is not the one expected, it is propagated
                propagateToken(t);
            }*/
            //return false;
        }


        /*After a token is received, to propagate it to the next node*/
        public static  void propagateToken(Token t) throws IOException {
            //System.out.println("Ibis[" + myIntIbisId + "] -> propagateToken");
            int tokenId = t.id;
            //if the token is black it is propagated as it is
            if(t.white) {
                t.white = white[tokenId];
            }
            white[tokenId] = true;
            //already connected with the next ibis instance
            //send the token to the next ibis instance
            WriteMessage term = tokenRequestSender.newMessage();
            term.writeObject(t);
            term.finish();
        }

        public void upcall(ReadMessage message) throws IOException,
            ClassNotFoundException {
            Token t = (Token)message.readObject();
            message.finish();
            if(t.id == myIntIbisId) {
                sync.arrivedToken(t);
            } else {
                propagateToken(t);
            }

        }
    }


    public static int solution(Cube cube, CubeCache cache) {
        valuatedCubes++;
        //System.out.println("Ibis[" + myIntIbisId + "] -> solution");
        if (cube.isSolved()) {
            //System.out.println("SOLVED");

            return 1;
        }

        if (cube.getTwists() >= cube.getBound()) {
            return 0;
        }
        //generate childrens
        Cube[] children = cube.generateChildren(cache);
        Cube child;
        int i;
        //add childrens on the toDo pool
        for(i = 0; i < children.length; i++) {
            child = children[(children.length - 1) - i];
            /*if(child == null) {
                System.out.println("Ibis[" + myIntIbisId + "] -> AHAHAHA 4");
            }*/

            add(child);

            //cache.put(child);
        }
        return 0;
    }






    public static int solutionsWorkers(Ibis ibis) throws Exception {
        //System.out.println("Ibis[" + myIntIbisId + "] -> solutionsWorkers");
        ArrayList<Cube> actual;
        CubeCache cache = null;
        boolean first = true;
        int result = 0;
        int i;
        Cube cube;
        boolean end = false;
        int elab = 0;


        while(!end) {
            workRequestReceiver.disableMessageUpcalls();
            while((actual = getWork(true)) != null && actual.size() != 0) {

                elab++;
                /*if(actual.size() != 1) {
                    System.out.println("Ibis[" + myIntIbisId + "] -> ActualSize PROBLEMS");
                }*/

                cube = actual.remove(0);
                if(cube == null) {
                    //System.out.println("Ibis[" + myIntIbisId + "] -> NULLCUBE");
                    continue;
                }
                //System.out.println("Ibis[" + myIntIbisId + "] -> ReceivedWork, twists: " + cube.getTwists() + ", bound: " + cube.getBound());
                if(first) {
                    cache = new CubeCache(cube.getSize());
                    first = false;
                }
                result += solution(cube, cache);

                /*------------------ADD HERE---------------------------------------*/
                if(cube != initialCube) {
                    cache.put(cube);
                }
                //System.out.println("Ibis[" + myIntIbisId + "] -> poll check");
                ReadMessage m = workRequestReceiver.poll();
                if(m != null) {
                    workRequestFromOthers(m);
                }

            }
            //System.out.println("Ibis[" + myIntIbisId + "] -> finio");
            //System.out.println("Ibis[" + myIntIbisId + "] -> solutionsWorkers -> No work");
            end = tokenManager.checkTermination();
        }
        //System.out.println("Ibis[" + myIntIbisId + "] -> solutionsWorkers -> FIrstTermination");
        return result;
    }

    public static int solutionsServer(ReceivePort resultsReceiver, Ibis ibis) throws Exception {
        //System.out.println("Ibis[" + myIntIbisId + "] -> SolutionsServer");
        int i;
        result = solutionsWorkers(ibis);
        System.out.println("Ibis[" + myIntIbisId + "] -> valuatedCubes: " + valuatedCubes + " workRequests: " + requestsForWork);
        //workManager.printSize();
        valuatedCubes = 0;
        requestsForWork = 0;
        for(i = 0; i < nodes - 1; i++) {
            ReadMessage r = resultsReceiver.receive();
            result += r.readInt();
            r.finish();
            //System.out.println("YEAH");
        }
        return result;
    }

    private static void solveServer(Ibis ibis) throws Exception {

        long start = System.currentTimeMillis();
        int bound = 0;
        int result = 0;
        /*ArrayList<Cube> work = workManager.getWork(true);
        Cube cube = work.get(0);*/
        //System.out.println("SolutionsServer");
        ReceivePort resultsReceiver = ibis.createReceivePort(portTypeMto1, "results");
        resultsReceiver.enableConnections();

        SendPort terminationSender = ibis.createSendPort(portType1toM);
        for (IbisIdentifier joinedIbis : joinedIbises) {
            if(joinedIbis.equals(myIbisId)) {
                continue;
            }
            terminationSender.connect(joinedIbis, "continue");
        }

        //Thread.sleep(1000);
        WriteMessage task;
        System.out.println("bound");
        while (result == 0) {
            bound++;
            initialCube.setBound(bound);
            //System.out.println("InitialCube : (" + initialCube.getBound() + ", " + initialCube.getTwists() + ")");
            add(initialCube);
            System.out.println(" " + bound);
            result = solutionsServer(resultsReceiver, ibis);
            //System.out.println("Result :" + result);

            //say to all to continue
            if(result == 0) {
                task = terminationSender.newMessage();
                task.writeBoolean(false);
                task.finish();
            }
        }

        //say to all that the work is finished
        task = terminationSender.newMessage();
        task.writeBoolean(true);
        task.finish();

        System.out.println();
        System.out.println("Solving cube possible in " + result + " ways of "
                           + bound + " steps");
        long end = System.currentTimeMillis();
        System.err.println("Solving cube took " + (end - start)
                           + " milliseconds");

        System.out.println("TERMINATE");
        terminationSender.close();
        Thread.sleep(1000);
        resultsReceiver.close();
        System.out.println("PortClosed");

    }

    public static void solveWorkers(Ibis ibis, IbisIdentifier server) throws Exception {

        //1 sender and many receivers
        ReceivePort terminationReceiver = ibis.createReceivePort(portType1toM, "continue");
        terminationReceiver.enableConnections();

        //many senders and 1 receiver
        SendPort resultsSender = ibis.createSendPort(portTypeMto1);
        resultsSender.connect(server, "results");

        //Thread.sleep(1000);
        //sender.connect(server, "results");
        //System.out.println("ConnectedToServerPort");
        boolean first = true;
        int i;
        int bound = 0;
        CubeCache cache = null;

        boolean end = false;
        while(!end) {
            result = solutionsWorkers(ibis);
            System.out.println("Ibis[" + myIntIbisId + "] -> valuatedCubes: "  + valuatedCubes + " workRequests: " + requestsForWork);
            //workManager.printSize();
            valuatedCubes = 0;
            requestsForWork = 0;
            //communicate my results
            WriteMessage resultMessage = resultsSender.newMessage();
            resultMessage.writeInt(result);
            resultMessage.finish();

            //check if I have to continue
            ReadMessage r = terminationReceiver.receive();
            end = r.readBoolean();
            r.finish();
        }


        System.out.println("FINE");
        resultsSender.close();
        Thread.sleep(1000);
        terminationReceiver.close();
        System.out.println("PortClosed");
    }



    public static void printUsage() {
        System.out.println("Rubiks Cube solver");
        System.out.println("");
        System.out
        .println("Does a number of random twists, then solves the rubiks cube with a simple");
        System.out
        .println(" brute-force approach. Can also take a file as input");
        System.out.println("");
        System.out.println("USAGE: Rubiks [OPTIONS]");
        System.out.println("");
        System.out.println("Options:");
        System.out.println("--size SIZE\t\tSize of cube (default: 3)");
        System.out
        .println("--twists TWISTS\t\tNumber of random twists (default: 11)");
        System.out
        .println("--seed SEED\t\tSeed of random generator (default: 0");
        System.out
        .println("--threads THREADS\t\tNumber of threads to use (default: 1, other values not supported by sequential version)");
        System.out.println("");
        System.out
        .println("--file FILE_NAME\t\tLoad cube from given file instead of generating it");
        System.out.println("");
    }

    /**
     * Main function.
     *
     * @param arguments
     *            list of arguments
     */


    private void run() throws Exception {
        //System.out.println("done");
        // Create an ibis instance.
        Ibis ibis = IbisFactory.createIbis(ibisCapabilities, null, portTypeMto1, portType1to1Up, portTypeMto1Poll, portType1toM, portType1to1);
        Thread.sleep(5000);
        System.out.println("Ibis created");
        myIbisId = ibis.identifier();
        myIbis = ibis;


        Runtime.getRuntime().addShutdownHook(new Thread() {
            public void run() {
                try {
                    //myIbis.registry().terminate();
                    myIbis.end();
                } catch(IOException e) {
                    System.err.println("Error");
                }
            }
        });

        // Elect a server
        System.out.println("elections");
        IbisIdentifier server = ibis.registry().elect("Server");

        System.out.println("Server is " + server);

        joinedIbises = ibis.registry().joinedIbises();
        nodes = joinedIbises.length;
        target = (myIntIbisId + 1) % nodes;
        white = new boolean[nodes];
        int i = 0;
        for (IbisIdentifier joinedIbis : joinedIbises) {
            System.err.println("Ibis joined: " + joinedIbis);
            if(joinedIbis.equals(myIbisId)) {
                myIntIbisId = i;
            }
            white[i] = true;
            i++;
        }

        tokenManager = new TokenManager();
        workManager = new WorkManager();


        workRequestReceiver = ibis.createReceivePort(portTypeMto1Poll, "WorkReq", workManager);
        // enable connections
        workRequestReceiver.enableConnections();



        //port in which new work requests will be sent
        workRequestSender = ibis.createSendPort(portTypeMto1Poll);

        //port in which new tokens will be received
        tokenRequestReceiver = ibis.createReceivePort(portType1to1Up, "TokenReq", tokenManager);
        // enable connections
        tokenRequestReceiver.enableConnections();
        // enable upcalls
        tokenRequestReceiver.enableMessageUpcalls();

        //port in which new tokens will be sent (the next ibis instance)
        tokenRequestSender = ibis.createSendPort(portType1to1Up);
        tokenRequestSender.connect(joinedIbises[(myIntIbisId + 1) % nodes], "TokenReq");

        //port in which new work is received
        //workReceiver = ibis.createReceivePort(portType1to1, "Work");
        workReceiver = ibis.createReceivePort(portTypeMto1, "Work");
        workReceiver.enableConnections();


        cubes_per_proc = new Integer[nodes];
        displs = new Integer[nodes];
        int avarage_cubes_per_proc = (6 * (size - 1)) / nodes;
        int rem = (6 * (size - 1)) % nodes;
        int sum = 0;
        for (i = 0; i < nodes; i++) {
            cubes_per_proc[i] = avarage_cubes_per_proc;
            if (rem > 0) {
                cubes_per_proc[i]++;
                rem--;
            }
            displs[i] = sum;
            sum += cubes_per_proc[i];
        }

        System.out.println("DISPL");
        for(i = 0; i < nodes; i++) {
            System.out.print(displs[i] + " ");
        }
        System.out.println("CUBESPERPROC");
        for(i = 0; i < nodes; i++) {
            System.out.print(cubes_per_proc[i] + " ");
        }

        // If I am the server, run server, else run client.
        if (server.equals(ibis.identifier())) {
            if(initialCube == null) {
                System.out.println("CUBE NULL FROM THE BEGIN");
            } else {
                System.out.println("CUBE ok");
            }
            //long start = System.currentTimeMillis();
            solveServer(ibis);
            //long end = System.currentTimeMillis();

            // NOTE: this is printed to standard error! The rest of the output is
            // constant for each set of parameters. Printing this to standard error
            // makes the output of standard out comparable with "diff"

            //terminate all workers
            // terminate the pool
            //System.out.println("Terminating pool");
            //ibis.registry().terminate();
            // wait for this termination to propagate through the system
            //ibis.registry().waitUntilTerminated();


        } else {
            solveWorkers(ibis, server);
        }

        workRequestSender.close();
        tokenRequestSender.close();
        Thread.sleep(1000);
        workRequestReceiver.close();
        tokenRequestReceiver.close();
        workReceiver.close();
        ibis.end();
    }

    public static void main(String[] arguments) {

        // default parameters of puzzle
        size = 3;
        int twists = 11;
        int seed = 0;
        String fileName = null;



        // number of threads used to solve puzzle
        // (not used in sequential version)

        for (int i = 0; i < arguments.length; i++) {
            if (arguments[i].equalsIgnoreCase("--size")) {
                i++;
                size = Integer.parseInt(arguments[i]);
            } else if (arguments[i].equalsIgnoreCase("--twists")) {
                i++;
                twists = Integer.parseInt(arguments[i]);
            } else if (arguments[i].equalsIgnoreCase("--seed")) {
                i++;
                seed = Integer.parseInt(arguments[i]);
            } else if (arguments[i].equalsIgnoreCase("--file")) {
                i++;
                fileName = arguments[i];
            } else if (arguments[i].equalsIgnoreCase("--help") || arguments[i].equalsIgnoreCase("-h")) {
                printUsage();
                System.exit(0);
            } else {
                System.err.println("unknown option : " + arguments[i]);
                printUsage();
                System.exit(1);
            }
        }

        // create cube
        if (fileName == null) {
            initialCube = new Cube(size, twists, seed);
        } else {
            try {
                initialCube = new Cube(fileName);
            } catch (Exception e) {
                System.err.println("Cannot load cube from file: " + e);
                System.exit(1);
            }
        }

        // print cube info
        System.out.println("Searching for solution for cube of size "
                           + initialCube.getSize() + ", twists = " + twists + ", seed = " + seed);
        initialCube.print(System.out);
        System.out.flush();


        try {
            System.out.println("run");
            new Rubiks().run();
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }

    }

}