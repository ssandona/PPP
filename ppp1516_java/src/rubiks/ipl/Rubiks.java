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
public class Rubiks {

    static PortType portTypeMto1 = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT,
            PortType.CONNECTION_MANY_TO_ONE);

    static PortType portType1to1 = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT,
            PortType.CONNECTION_ONE_TO_ONE);

    static PortType portTypeMto1Up = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.CONNECTION_MANY_TO_ONE, PortType.RECEIVE_AUTO_UPCALLS);

    static PortType portType1toM = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT,
            PortType.CONNECTION_ONE_TO_MANY);

    static IbisCapabilities ibisCapabilities = new IbisCapabilities(
        IbisCapabilities.ELECTIONS_STRICT, IbisCapabilities.MEMBERSHIP_TOTALLY_ORDERED);

    static ArrayList<Cube> toDo = new ArrayList<Cube>();
    static Object lock = new Object();

    static IbisIdentifier[] joinedIbises;
    static Ibis myIbis;
    static IbisIdentifier myIbisId;
    static int nodes;

    static ReceivePort workRequestReceiver;
    static SendPort workRequestSender;
    static ReceivePort workReceiver;
    //workSender create on demand
    static SendPort resultsSender;
    static ReceivePort resultsReceiver;
    static SendPort workSender;
    //static ReceivePort terminationReceiver;

    static SyncTermination syncTermination;
    static int requestsForWork = 0;
    static int valuatedCubes = 0;

    static IbisIdentifier server;
    static Cube initialCube;


    static String[] arguments;


    public static int solution(Cube cube, CubeCache cache) {
        valuatedCubes++;
        //System.out.println(myIbisId + " -> solution");
        if (cube.isSolved()) {
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
        synchronized(lock) {
            for(i = 0; i < children.length; i++) {
                child = children[(children.length - 1) - i];
                toDo.add(child);
                //cache.put(child);
            }
        }
        return 0;
    }

    public static final boolean PRINT_SOLUTION = false;



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

    //method called to ask work to the server (local work pool empty)
    public static Cube askForWork() throws IOException, ClassNotFoundException {
        System.out.println(myIbisId + " -> askForWork");
        Cube receivedWork = null;
        requestsForWork++;

        //send local work receiving port
        WriteMessage task = workRequestSender.newMessage();
        task.writeObject(workReceiver.identifier());
        task.finish();
        System.out.println(myIbisId + " -> asked");

        //get the work
        ReadMessage r = workReceiver.receive();
        receivedWork = (Cube)r.readObject();
        r.finish();
        if(receivedWork == null) {
            System.out.println(myIbisId + " -> NULLworkReceived");
        } else {
            System.out.println(myIbisId + " -> workReceived");
        }

        //workRequestSender.disconnect(doner, "WorkReq");
        return receivedWork;
    }

    //method called to ask work to the server (local work pool empty)
    public static boolean waitForInitialWork() throws IOException, ClassNotFoundException {
        System.out.println(myIbisId + " -> waitFOrInitialWork");
        Cube receivedWork = null;
        //get the work
        ReadMessage r = workReceiver.receive();
        receivedWork = (Cube)r.readObject();
        r.finish();
        toDo.add(receivedWork);
        if(receivedWork == null) {
            System.out.println(myIbisId + " -> NULLworkReceived");
            return true;
        } else {
            System.out.println(myIbisId + " -> InitialWorkReceived");
            return false;
        }


    }

    //extract the last element of the work pool and return it, null if the work pool is empty
    //method calle directly by the server and indirectly (through getWork()) by the Slaves
    synchronized public static Cube getFromPool (boolean sameNode) {
        Cube cube = null;
        synchronized(lock) {
            if(toDo.size() != 0) {
                int index;
                if(sameNode) {
                    index = toDo.size() - 1;
                } else {
                    index = 0;
                }
                cube = toDo.remove(index);
                //toDoWeight -= (c.getBound() - c.getTwists());
            }
        }
        return cube;
    }

    //method called by Slaves to getWork, if the work queue is empty, some work is asked to the server
    //if this method return null, that means that there is no more work to do
    public static Cube getWork() throws IOException, ClassNotFoundException {
        Cube cube;
        cube = getFromPool(true);
        if(cube == null) {
            cube = askForWork();
        }

        return cube;
    }


    //send the actual results, and as response receive if another bound has to be evaluated
    //or if the system can terminate
    public static void sendResults(int res) throws IOException {
        System.out.println(myIbisId + " -> send results to server");
        System.out.println(myIbisId + " -> computed " + valuatedCubes + " cubes");
        valuatedCubes = 0;
        boolean termination;
        //send local work receiving port
        WriteMessage resMsg = resultsSender.newMessage();
        resMsg.writeInt(res);
        resMsg.finish();

        //get the work
        /*ReadMessage r = terminationReceiver.receive();
        termination = r.readBoolean();
        r.finish();
        return termination;*/
    }

    public static class ResultsUpdater implements MessageUpcall {

        public void upcall(ReadMessage message) throws IOException,
            ClassNotFoundException {
            System.out.println(myIbisId + " -> received results from Slave");
            int results = message.readInt();
            try {
                syncTermination.increaseResults(results);
            } catch(InterruptedException ie) {
                ie.printStackTrace();
            }

        }


    }

    public static class WorkManager implements MessageUpcall {

        //method called when a new work request comes from a Slave
        public void upcall(ReadMessage message) throws IOException,
            ClassNotFoundException {
            ReceivePortIdentifier requestor = (ReceivePortIdentifier) message
                                              .readObject();

            System.err.println("received request from: " + requestor);

            // finish the request message. This MUST be done before sending
            // the reply message. It ALSO means Ibis may now call this upcall
            // method agian with the next request message
            message.finish();

            // connect to the requestor's receive port
            workSender.connect(requestor);

            Cube cube = getFromPool(false);
            if(cube == null) {
                System.out.println(myIbisId + " -> tryToSendNULL");
            } else {
                System.out.println(myIbisId + " -> tryToSendCube");
            }
            // create a reply message
            WriteMessage reply = workSender.newMessage();
            reply.writeObject(cube);
            reply.finish();

            workSender.disconnect(requestor);
            //workSender.close();
            System.out.println(myIbisId + " -> sent");
        }


    }


    public void solutionsWorkers() throws IOException, ClassNotFoundException {
        Cube cube = null;
        CubeCache cache = null;
        boolean first = true;
        int results = 0;
        boolean end = false;

        //while there are bounds to evaluate
        while(!end) {
            results = 0;
            if((end = waitForInitialWork())) {
                continue;
            }
            System.out.println(myIbisId + " -> another round");
            //while there is work for the actual bound
            while((cube = getWork()) != null) {
                //cache initialization with rhe first received cube
                if(first) {
                    cache = new CubeCache(cube.getSize());
                    first = false;
                }
                results += solution(cube, cache);
            }
            sendResults(results);
        }
    }

    public int sendInitialWork(boolean terminated, CubeCache cache) throws IOException, ConnectionFailedException, InterruptedException {
        Cube cube = null;
        int count = 0;
        int results = 0;

        if(terminated) {
            //send initial cubes to the slaves
            for (IbisIdentifier joinedIbis : joinedIbises) {
                if(joinedIbis.equals(myIbisId)) {
                    continue;
                }
                workSender.connect(joinedIbis, "Work");
            }
            WriteMessage reply = workSender.newMessage();
            reply.writeObject(cube);
            reply.finish();
            return 0;
        } else {

            //send initial cubes to the slaves
            for (IbisIdentifier joinedIbis : joinedIbises) {
                if(joinedIbis.equals(myIbisId)) {
                    continue;
                }
                cube = getFromPool(false);
                workSender.connect(joinedIbis, "Work");
                WriteMessage reply = workSender.newMessage();
                reply.writeObject(cube);
                reply.finish();
                workSender.disconnect(joinedIbis, "Work");
                syncTermination.increaseBusyWorkers();
            }
            System.out.println(myIbisId + " -> initialWorkSent");
            return 0;
        }
    }

    public int solutionsServer(CubeCache cache) throws InterruptedException, IOException {
        syncTermination.increaseBusyWorkers();
        int results = 0;
        Cube cube=null;
        int twist = 0;
        int bound=cube.getBound();
        cube= getFromPool(true);
        results+=solution(cube,cache);
        System.out.println(myIbisId + " -> FIRST " + toDo.size() + " cubes");
        if(results!=0){
            return results;
        }
        int n=toDo.size();
        int i;
        for(i=0;i<n;i++){
            cube=toDo.remove(0);
            results+=solution(cube,cache);
        }
        if(results!=0){
            return results;
        }
        System.out.println(myIbisId + " -> SECOND " + toDo.size() + " cubes");
        if(bound==2){
            sendInitialWork(false, cache);
        }
        //while the work pool is not empty, continue to work
        while((cube = getFromPool(true)) != null) {
            results += solution(cube, cache);
            if(cube != initialCube) {
                cache.put(cube);
            }
        }
        System.out.println(myIbisId + " -> computed " + valuatedCubes + " cubes");
        valuatedCubes = 0;
        //add my results to the cumulative results
        System.out.println(myIbisId + " -> increase results");
        syncTermination.increaseResults(results);

        System.out.println(myIbisId + " -> wait termination");
        //wait until all the slaves terminate the calculation for this bound and get the cumulative results
        int boundResult = syncTermination.waitTermination();
        System.out.println(myIbisId + " -> terminated");
        return boundResult;
    }


    public void solveServer() throws InterruptedException, IOException {
        ResultsUpdater resultsUpdater = new ResultsUpdater();
        WorkManager workManager = new WorkManager();
        //port in which new work requests will be received
        workRequestReceiver = myIbis.createReceivePort(portTypeMto1Up, "WorkReq", workManager);
        // enable connections
        workRequestReceiver.enableConnections();
        // enable upcalls
        workRequestReceiver.enableMessageUpcalls();

        resultsReceiver = myIbis.createReceivePort(portTypeMto1Up, "Results", resultsUpdater);
        // enable connections
        resultsReceiver.enableConnections();
        // enable upcalls
        workRequestReceiver.enableMessageUpcalls();

        workSender = myIbis.createSendPort(portType1toM);


        initialCube = generateCube();
        CubeCache cache = new CubeCache(initialCube.getSize());

        syncTermination = new SyncTermination();

        int bound = 0;
        int result = 0;

        WriteMessage termination;


        System.out.print("Bound now:");

        long start = System.currentTimeMillis();
        while (result == 0) {
            bound++;
            initialCube.setBound(bound);
            System.out.println("InitialCube : (" + initialCube.getBound() + ", " + initialCube.getTwists() + ")");
            toDo.add(initialCube);
            System.out.print(" " + bound);
            result = solutionsServer(cache);
        }
        long end = System.currentTimeMillis();
        sendInitialWork(true, cache);

        System.out.println();
        System.out.println("Solving cube possible in " + result + " ways of "
                           + bound + " steps");

        System.err.println("Solving cube took " + (end - start)
                           + " milliseconds");

        //close all ports
        //terminationSender.close();
        Thread.sleep(1000);
        resultsReceiver.close();
        workSender.close();
        workRequestReceiver.close();
    }

    public void solveWorkers() throws IOException, InterruptedException, ClassNotFoundException {
        //workReceiver = ibis.createReceivePort(portType1to1, "Work");
        workReceiver = myIbis.createReceivePort(portType1toM, "Work");
        workReceiver.enableConnections();

        /*terminationReceiver = myIbis.createReceivePort(portType1toM, "Termination");
        terminationReceiver.enableConnections();*/

        //port in which new work requests will be sent
        workRequestSender = myIbis.createSendPort(portTypeMto1Up);
        workRequestSender.connect(server, "WorkReq");

        resultsSender = myIbis.createSendPort(portTypeMto1Up);
        resultsSender.connect(server, "Results");

        solutionsWorkers();

        //close all the ports
        workRequestSender.close();
        Thread.sleep(1000);
        workReceiver.close();
        //terminationReceiver.close();



    }

    public static Cube generateCube() {
        Cube cube = null;

        // default parameters of puzzle
        int size = 3;
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
            cube = new Cube(size, twists, seed);
        } else {
            try {
                cube = new Cube(fileName);
            } catch (Exception e) {
                System.err.println("Cannot load cube from file: " + e);
                System.exit(1);
            }
        }

        // print cube info
        System.out.println("Searching for solution for cube of size "
                           + cube.getSize() + ", twists = " + twists + ", seed = " + seed);
        cube.print(System.out);
        System.out.flush();
        return cube;
    }

    private void run() throws Exception {
        //System.out.println("done");
        // Create an ibis instance.
        Ibis ibis = IbisFactory.createIbis(ibisCapabilities, null, portTypeMto1Up, portTypeMto1, portType1toM, portType1to1);
        Thread.sleep(5000);
        System.out.println("Ibis created");
        myIbisId = ibis.identifier();
        myIbis = ibis;

        // Elect a server
        System.out.println("elections");
        server = ibis.registry().elect("Server");

        System.out.println("Server is " + server);

        joinedIbises = ibis.registry().joinedIbises();
        nodes = joinedIbises.length;

        for (IbisIdentifier joinedIbis : joinedIbises) {
            System.err.println("Ibis joined: " + joinedIbis);
        }


        // If I am the server, run server, else run client.
        if (server.equals(myIbisId)) {
            solveServer();


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
            solveWorkers();
        }

        workRequestSender.close();
        Thread.sleep(1000);
        workRequestReceiver.close();
        workReceiver.close();
        ibis.end();
    }


    public static void main(String[] argumentsForCube) {
        arguments = argumentsForCube;
        try {
            System.out.println("run");
            new Rubiks().run();
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }
}

