package rubiks.ipl;

import ibis.ipl.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.IOException;

import java.lang.Math;


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

    static PortType portTypeMto1Up = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_OBJECT, PortType.CONNECTION_MANY_TO_ONE, PortType.RECEIVE_AUTO_UPCALLS);

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

    static WorkManager workManager;
    static TokenManager tokenManager;

    static int children;
    static CubeCache cache = null;

    static String[] arguments;


    public static final boolean PRINT_SOLUTION = false;

    static class WorkManager implements MessageUpcall {
        static ArrayList<Cube> toDo = null;
        static ArrayList<ArrayList<Cube>> toDoTree = new ArrayList<ArrayList<Cube>>();
        static int actualTreeLevel = 0;
        static int nodesOnTree = 0;
        static int toDoWeight = 0;
        static Object lock = new Object();

        public WorkManager() {
            int i;
            for(i = 0; i < 20; i++) {
                toDoTree.add(new ArrayList<Cube>());
            }
            System.out.println("SIZE OF TREE -> " + toDoTree.size());
        }

        public static void printTree() {
            synchronized(lock) {
                String s = "";
                int i, j;
                for(i = 0; i < 20; i++) {
                    s += "\n [" + i + "]";
                    ArrayList<Cube> actual = toDoTree.get(i);
                    for(j = 0; j < actual.size(); j++) {
                        s += "* ";
                    }

                }
                System.out.println("Ibis[" + myIntIbisId + "] ----- TREE-----" + s);
            }
        }

        /*synchronized public static boolean availableWork() {
            System.out.println("Ibis[" + myIntIbisId + "] -> SIZE: " + toDo.size());
        }*/

        public static void add(Cube cube) {
            synchronized(lock) {        //Only one PrintThread at a time can call syn1.display()
                /*if(cube == null) {
                    System.out.println("Ibis[" + myIntIbisId + "] -> AHAHAH 6");
                }*/
                //toDo.add(cube);
                actualTreeLevel = cube.getTwists();
                toDo = toDoTree.get(actualTreeLevel);

                toDo.add(cube);
                nodesOnTree++;
                //printTree();
                //toDoWeight += Math.pow(children,(cube.getBound() - cube.getTwists()));
            }
            //System.out.println("Ibis[" + myIntIbisId + "] -> added cube");
        }


        public static boolean askForWork() throws IOException, ClassNotFoundException {
            //System.out.println("Ibis[" + myIntIbisId + "] -> askForWork");
            int i;
            Cube[] receivedWork = new Cube[0];
            //Cube[] receivedWork;
            IbisIdentifier doner;
            for(i = 0; i < nodes; i++) {
                doner = joinedIbises[target];
                if(!doner.equals(myIbisId)) {
                    requestsForWork++;
                    workRequestSender.connect(doner, "WorkReq");
                    WriteMessage task = workRequestSender.newMessage();
                    task.writeInt(myIntIbisId);
                    task.finish();

                    ReadMessage r = workReceiver.receive();
                    int cubes = r.readInt();
                    r.finish();
                    if(cubes == 0) {
                        //System.out.println("Ibis[" + myIntIbisId + "] -> no work");
                    } else {
                        r = workReceiver.receive();
                        //System.out.println("Ibis[" + myIntIbisId + "] -> work!!! " + cubes + " cubes");
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
                            workManager.add(c);
                        }
                        return true;
                    }
                }
                target = (target + 1) % nodes;
            }
            return false;
        }

        /*synchronized public static ArrayList<Cube> getFromLevel(int level) {
            toDo = toDoTree.get(level);
            return toDo.remove(toDo.size() - 1);
        }*/

        synchronized public static ArrayList<Cube> getFromPool (boolean sameNode) {
            ArrayList<Cube> workToReturn = new ArrayList<Cube>();
            if(nodesOnTree == 0) {
                return null;
            }
            if(sameNode) {
                Cube c = null;
                synchronized(lock) {
                    /*if(toDoTree[actualTreeLevel].size() == 0){
                        actualTreeLevel--;
                    }*/
                    //int n = toDo.size() - 1;
                    //System.out.println("Ibis[" + myIntIbisId + "nodes  on tree -> "+nodesOnTree+ " actual level "+actualTreeLevel);
                    int n = toDo.size() - 1;
                    while(n < 0) {
                        actualTreeLevel--;
                        toDo = toDoTree.get(actualTreeLevel);
                        n = toDo.size() - 1;
                    }
                    c = toDo.remove(n);
                    nodesOnTree--;
                    //printTree();
                    //c = toDo.remove(n);
                    //toDoWeight -= Math.pow(children,(c.getBound() - c.getTwists()));
                }
                /*if(c == null) {
                    System.out.println("Ibis[" + myIntIbisId + "] -> AHAHAHA 1, index -> "+n);
                }*/
                workToReturn.add(c);
            } else {
                synchronized(lock) {
                    /*int amount = toDo.size() / 2;
                    boolean even = toDo.size() % 2 == 0;
                    int index = even ? toDo.size() / 2 : toDo.size() / 2 + 1;
                    int i;*/
                    /*int weightToDistribute = toDoWeight / 2;
                    int distributed = 0;
                    while(distributed < weightToDistribute) {
                        Cube c = toDo.remove(0);
                        workToReturn.add(c);
                        toDoWeight -= Math.pow(children,(c.getBound() - c.getTwists()));
                        distributed += Math.pow(children,(c.getBound() - c.getTwists()));
                    }
                    if(workToReturn.size() == 0) {
                        workToReturn = null;
                        //System.out.println("Ibis[" + myIntIbisId + "] -> send to the other 0 cubes");
                    }*/

                    //for each tree level, distribute half of the nodes
                    int i, j;
                    ArrayList<Cube> actual;
                    int bound = actualTreeLevel; // < 4 ? actualTreeLevel:4;
                    /*for(i = 0; i < bound; i++) {
                        actual = toDoTree.get(i);
                        int amount = actual.size() / 2;
                        for(j = 0; j < amount; j++) {
                            workToReturn.add(actual.remove(0));
                            nodesOnTree--;
                        }
                    }*/
                    for(i = 0; i < bound; i++) {
                        actual = toDoTree.get(i);
                        int amount = actual.size() / 2 + 1;
                        if(amount <= 1) {
                            continue;
                        }
                        for(j = 0; j < amount; j++) {
                            workToReturn.add(actual.remove(0));
                            nodesOnTree--;
                        }
                        break;
                    }


                }
            }
            return workToReturn;
        }
        //function invokable from both actual worker or another one, if invoked by the actual worker
        //and the queue i(toDo) is empty, the function askForWork is invoked (that invoke the workRequest function
        //on the other nodes
        public static ArrayList<Cube> getWork(boolean sameNode) throws IOException, ClassNotFoundException {
            //System.out.println("Ibis[" + myIntIbisId + "] -> getWork");
            ArrayList<Cube> work = getFromPool(sameNode);
            if(sameNode) {
                if(work == null) {
                    //System.out.println("Ibis[" + myIntIbisId + "] -> toDo Empty");
                    boolean b = askForWork();
                    if(!b) {
                        return null;
                    } else {
                        work = getFromPool(sameNode);
                    }
                }
            }
            return work;

        }

        /*A request of work from another Ibis instance*/
        public void upcall(ReadMessage message) throws IOException,
            ClassNotFoundException {
            int otherIbisId = message.readInt();
            message.finish();
            //System.out.println("Ibis[" + myIntIbisId + "] -> workrequest");
            //get ibisIdentifier of the requestor
            IbisIdentifier requestor = joinedIbises[otherIbisId];
            int i;
            // create a sendport for the reply
            //SendPort replyPort = myIbis.createSendPort(portType1to1);
            SendPort replyPort = myIbis.createSendPort(portTypeMto1);

            ArrayList<Cube> subPool;
            Cube[] subPoolToSend = new Cube[0];
            //Cube[] subPoolToSend = null;
            /*if(toDo.size() == 0) {
                subPool = null;
            } else {*/
            subPool = getWork(false);
            //}
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

            workManager.add(child);
            //cache.put(child);
        }
        return 0;
    }

    public static int solutionInitial(Cube cube, CubeCache cache, ArrayList<Cube> toDo) {
        valuatedCubes++;
        //System.out.println("Ibis[" + myIntIbisId + "] -> solution");
        if (cube.isSolved()) {
            //System.out.println("SOLVED");

            return 1;
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

            toDo.add(child);
            //cache.put(child);
        }
        return 0;
    }






    public static int solutionsWorkers() throws Exception {
        //System.out.println("Ibis[" + myIntIbisId + "] -> solutionsWorkers");
        ArrayList<Cube> actual;
        int result = 0;
        int i;
        Cube cube;
        boolean end = false;
        while(!end) {
            while((actual = workManager.getWork(true)) != null && actual.size() != 0) {
                /*if(actual.size() != 1) {
                    System.out.println("Ibis[" + myIntIbisId + "] -> ActualSize PROBLEMS");
                }*/

                cube = actual.remove(0);
                if(cube == null) {
                    //System.out.println("Ibis[" + myIntIbisId + "] -> NULLCUBE");
                    continue;
                }
                //System.out.println("Ibis[" + myIntIbisId + "] -> ReceivedWork, twists: " + cube.getTwists() + ", bound: " + cube.getBound());
                if(cache == null) {
                    cache = new CubeCache(cube.getSize());
                    children = 6 * (cube.getSize() - 1);
                }
                result += solution(cube, cache);

                /*------------------ADD HERE---------------------------------------*/
                if(cube != initialCube) {
                    cache.put(cube);
                }

            }
            //System.out.println("Ibis[" + myIntIbisId + "] -> solutionsWorkers -> No work");
            end = tokenManager.checkTermination();
        }
        //System.out.println("Ibis[" + myIntIbisId + "] -> solutionsWorkers -> FIrstTermination");
        return result;
    }

    public static int solutionsServer(ReceivePort resultsReceiver) throws Exception {
        //System.out.println("Ibis[" + myIntIbisId + "] -> SolutionsServer");
        int i;
        result = solutionsWorkers();
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

    public static int generateFirstLevel(Cube cube, CubeCache cache, ArrayList<Cube> toDo) {
        int results = solutionInitial(cube, cache, toDo);
        //System.out.println(myIbisId + " -> FIRST " + toDo.size() + " cubes");
        return results;
    }

    public static int generateSecondLevel(CubeCache cache, ArrayList<Cube> toDo) {
        int n = toDo.size();
        int results=0;
        int i;
        for(i = 0; i < n; i++) {
            Cube cube = toDo.remove(0);
            results += solutionInitial(cube, cache, toDo);
        }
        //System.out.println(myIbisId + " -> SECOND " + toDo.size() + " cubes");
        return results;
    }

    private static void solveServer(Ibis ibis) throws Exception {
        ReceivePort resultsReceiver = ibis.createReceivePort(portTypeMto1, "results");
        resultsReceiver.enableConnections();

        SendPort terminationSender = ibis.createSendPort(portType1toM);
        for (IbisIdentifier joinedIbis : joinedIbises) {
            if(joinedIbis.equals(myIbisId)) {
                continue;
            }
            terminationSender.connect(joinedIbis, "continue");
        }

        long start = System.currentTimeMillis();
        int bound = 2;
        int result = 0;
        int i;
        /*ArrayList<Cube> work = workManager.getWork(true);
        Cube cube = work.get(0);*/
        //System.out.println("SolutionsServer");


        CubeCache cache = new CubeCache(initialCube.getSize());
        ArrayList<Cube> toDo;
        WriteMessage task;

        System.out.println("bound");
        while(result == 0) {
            bound++;
            initialCube.setBound(bound);
            toDo=new ArrayList<Cube>();
            result = generateFirstLevel(initialCube, cache, toDo);
            if(result == 0) {
                result = generateSecondLevel(cache, toDo);
                if(result != 0) {
                    bound = 2;
                }
            } else {
                bound = 1;
            }
            if(result != 0) {
                continue;
            }

            int mydisp = displs[myIntIbisId];
            for(i = 0; i < displs[myIntIbisId]; i++) {
                toDo.remove(0);
            }
            for(i = 0; i < cubes_per_proc[myIntIbisId]; i++) {
                workManager.add(toDo.remove(0));
            }

            //Thread.sleep(1000);
            System.out.println(" " + bound);
            result = solutionsServer(resultsReceiver);
            //System.out.println("Result :" + result);

            //say to all to continue
            if(result == 0) {
                task = terminationSender.newMessage();
                task.writeBoolean(false);
                task.finish();
            }
        }

        if(bound > 2) {
            //say to all that the work is finished
            task = terminationSender.newMessage();
            task.writeBoolean(true);
            task.finish();
        }

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

        CubeCache cache = new CubeCache(initialCube.getSize());

        int result = 0;
        boolean end = false;
        int i;
        int bound = 2;

        ArrayList<Cube> toDo=new ArrayList<Cube>();

        while(!end) {
            bound++;
            initialCube.setBound(bound);
            toDo=new ArrayList<Cube>();
            result = generateFirstLevel(initialCube, cache, toDo);
            if(result != 0) {
                end = true;
                continue;
            }
            result = generateSecondLevel(cache, toDo);
            if(result != 0) {
                end = true;
                continue;
            }

            int mydisp = displs[myIntIbisId];
            for(i = 0; i < displs[myIntIbisId]; i++) {
                toDo.remove(0);
            }
            for(i = 0; i < cubes_per_proc[myIntIbisId]; i++) {
                workManager.add(toDo.remove(0));
            }

            result = solutionsWorkers();
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
        Ibis ibis = IbisFactory.createIbis(ibisCapabilities, null, portTypeMto1, portType1to1Up, portTypeMto1Up, portType1toM, portType1to1);
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

        workManager = new WorkManager();
        tokenManager = new TokenManager();


        //port in which new work requests will be received
        workRequestReceiver = ibis.createReceivePort(portTypeMto1Up, "WorkReq", workManager);
        // enable connections
        workRequestReceiver.enableConnections();
        // enable upcalls
        workRequestReceiver.enableMessageUpcalls();

        //port in which new work requests will be sent
        workRequestSender = ibis.createSendPort(portTypeMto1Up);

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

        initialCube = generateCube();

        cubes_per_proc = new Integer[nodes];
        displs = new Integer[nodes];
        int jobs = (6 * (initialCube.getSize() - 1)) * (6 * (initialCube.getSize() - 1));
        int avarage_cubes_per_proc = jobs / nodes;
        int rem = jobs % nodes;
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