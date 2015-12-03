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
            PortType.CONNECTION_MANY_TO_ONE, PortType.RECEIVE_POLL);

    static PortType portType1to1 = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT,
            PortType.CONNECTION_ONE_TO_ONE, PortType.RECEIVE_POLL);

    static PortType requestWorkPortType = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_AUTO_UPCALLS,
            PortType.CONNECTION_ONE_TO_ONE);




    static IbisCapabilities ibisCapabilities = new IbisCapabilities(
        IbisCapabilities.ELECTIONS_STRICT, IbisCapabilities.MEMBERSHIP_TOTALLY_ORDERED);

    /*static IbisCapabilities ibisCapabilities = new IbisCapabilities(
        IbisCapabilities.ELECTIONS_STRICT);*/

    static int counter = 0;
    static ArrayList<Cube> toDo;
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
    static Cube cube = null;
    static int size;
    static int level;
    static int target;

    static ReceivePort workRequestReceiver;
    static SendPort workRequestSender;
    static ReceivePort workReceiver;

    static ReceivePort tokenRequestReceiver;
    static SendPort tokenRequestSender;


    public static final boolean PRINT_SOLUTION = false;

    static class Token {
        public int id;
        public boolean white = true;
        public Token(int id) {
            this.id = id;
        }
    }


    public static boolean askForWork() throws Exception {
    	int i;
        Cube[] receivedWork = null;
        IbisIdentifier doner;
        for(i = 0; i < nodes; i++) {
            doner = joinedIbises[target];
            if(!doner.equals(myIbisId)) {
                workRequestSender.connect(doner, "WorkReq");
                WriteMessage task = workRequestSender.newMessage();
                task.writeInt(myIntIbisId);
                task.finish();

                ReadMessage r = workReceiver.receive();
                //System.out.println("ReceivedMyWork");
                r.readArray(receivedWork);
                r.finish();
                workRequestSender.disconnect(doner, "WorkReq");
                if(receivedWork.length == 0) {
                    toDo = new ArrayList<Cube>(Arrays.asList(receivedWork));
                    return true;
                }
            }
            target = (target + 1) % nodes;
        }
        return false;
    }


    //function invokable from both actual worker or another one, if invoked by the actual worker
    //and the queue i(toDo) is empty, the function askForWork is invoked (that invoke the workRequest function
    //on the other nodes
    public static ArrayList<Cube> getWork(boolean sameNode) throws Exception {
        ArrayList<Cube> workToReturn = new ArrayList<Cube>();
        if(sameNode) {
            if(toDo.size() == 0) {
                boolean b = askForWork();
                if(!b) {
                    return null;
                }
            }
            workToReturn.add(toDo.remove(toDo.size() - 1));
        } else {
            int amount = toDo.size() / 2;
            boolean even = toDo.size() % 2 == 0;
            int index = even ? toDo.size() / 2 : toDo.size() / 2 + 1;
            int i;
            for(i = 0; i < amount; i++) {
                workToReturn.add(toDo.remove(index));
            }
            if(workToReturn.size() == 0) {
                workToReturn = null;
            }

        }
        return workToReturn;
    }

    public static int solution(Cube cube, CubeCache cache) {
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
        for(i = 0; i < children.length; i++) {
            child = children[(children.length-1) - i];
            toDo.add(child);
            cache.put(cube);
        }
        return 0;
    }


    public static boolean checkTermination () throws Exception {
    	//create a new token
        Token t = new Token(myIntIbisId);
        //already connected with the next ibis instance
        //send the token to the next ibis instance
        WriteMessage term = tokenRequestSender.newMessage();
        term.writeObject(t);
        term.finish();

        //wait the token comes back
        boolean myToken = false;
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
        }
        return false;
    }


/*After a token is received, to propagate it to the next node*/
    public static  void propagateToken(Token t) throws Exception {
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

/*A request of work from another Ibis instance*/
    public static void workRequest(int otherIbisId) throws Exception {
    	//get ibisIdentifier of the requestor
        IbisIdentifier requestor = joinedIbises[otherIbisId];
        int i;
        // create a sendport for the reply
        SendPort replyPort = myIbis.createSendPort(portType1to1);
        ArrayList<Cube> subPool;
        if(toDo.size() == 0) {
            subPool = null;
        } else {
            subPool = getWork(false);
        }
        //update node color in the eyes of every other node
        if(subPool != null) {
            if(myIntIbisId>otherIbisId){
            	for(i=0;i<=otherIbisId;i++){
            		white[i]=false;
            	}
            	for(i=myIntIbisId+1;i<nodes;i++){
            		white[i]=false;
            	}
            }
            else{
            	for(i=myIntIbisId+1;i<=otherIbisId;i++){
            		white[i]=false;
            	}
            }
        }

        // connect to the requestor's receive port
        replyPort.connect(requestor, "Work");

        // send the work to him
        WriteMessage reply = replyPort.newMessage();
        reply.writeArray(subPool.toArray(new Cube[subPool.size()]));
        reply.finish();
        replyPort.close();
    }

    public static int solutionsWorkers() throws Exception {
        ArrayList<Cube> actual;
        CubeCache cache=null;
        boolean first = true;
        int result = 0;
        int i;
        Cube cube;
        boolean end = false;
        while(!end) {
            while((actual = getWork(true)) != null) {
                cube = actual.remove(0);
                if(first) {
                    cache = new CubeCache(cube.getSize());
                    first = false;
                }
                result += solution(cube, cache);

                //check for pending work requests
                ReadMessage r= workRequestReceiver.poll();
                if(r!=null){
                	workRequest(r.readInt());
                }
                r.finish();
                //check for pending token requests
                r= tokenRequestReceiver.poll();
                if(r!=null){
                	propagateToken((Token)r.readObject());
                }
                r.finish();
            }
            end = checkTermination();
        }
        return result;
    }

    public static int solutionsServer(ReceivePort resultsReceiver) throws Exception {
        int i;
        result = solutionsWorkers();
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
        Cube cube = toDo.get(0);
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

        Thread.sleep(1000);
        WriteMessage task;
        System.out.println("Bound:");
        while (result == 0) {
            bound++;
            cube.setBound(bound);
            System.out.println(" " + bound);
            result = solutionsServer(resultsReceiver);
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
        resultsReceiver.close();
        terminationSender.close();
        System.out.println("PortClosed");

    }

    public static void solveWorkers(Ibis ibis, IbisIdentifier server) throws Exception {

    	//1 sender and many receivers
        ReceivePort terminationReceiver = ibis.createReceivePort(portType1toM, "continue");
        terminationReceiver.enableConnections();

        //many senders and 1 receiver
        SendPort resultsSender = ibis.createSendPort(portTypeMto1);
        resultsSender.connect(server, "results");

        Thread.sleep(1000);
        //sender.connect(server, "results");
        //System.out.println("ConnectedToServerPort");
        boolean first = true;
        int i;
        int bound = 0;
        Cube cube = null;
        CubeCache cache = null;

        boolean end = false;
        while(!end) {
            result = solutionsWorkers();

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
        terminationReceiver.close();
        resultsSender.close();
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
        Ibis ibis = IbisFactory.createIbis(ibisCapabilities, null, portTypeMto1, portType1toM, portType1to1);
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

        //port in which new work requests will be received
        workRequestReceiver = ibis.createReceivePort(portTypeMto1, "WorkReq");
        // enable connections
        workRequestReceiver.enableConnections();
       
       	//port in which new work requests will be sent
        workRequestSender = ibis.createSendPort(portTypeMto1);

        //port in which new tokens will be received
        tokenRequestReceiver = ibis.createReceivePort(portType1to1, "TokenReq");
        // enable connections
        tokenRequestReceiver.enableConnections();
        // enable upcalls

        //port in which new tokens will be sent (the next ibis instance)
        tokenRequestSender = ibis.createSendPort(portType1to1);
        tokenRequestSender.connect(joinedIbises[(myIntIbisId+1)%nodes],"TokenReq");

        //port in which new work is received
        workReceiver = ibis.createReceivePort(portType1to1, "Work");
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
            toDo.add(cube);
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

        // End ibis.
        //ibis.registry().terminate();
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

        //end if beginner

        //done = new ArrayList<String>();
        toDo = new ArrayList<Cube>();



        try {
            System.out.println("run");
            new Rubiks().run();
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }

    }

}
