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
    static int nodes = 1;
    static IbisIdentifier[] joinedIbises;
    static int myIntIbisId;
    static IbisIdentifier myIbisId;
    static Ibis myIbis;
    static Cube initialCube = null;


    static int valuatedCubes = 0;


    static CubeCache cache = null;

    static String[] arguments;


    public static final boolean PRINT_SOLUTION = false;

    static ArrayList<Cube> toDo = new ArrayList<Cube>();
    static int nodesOnTree = 0;

    public static void printTree() {
        String s = "\n";
        int i, j;
        int level = -1;
        int nodes = 0;
        for(i = 0; i < toDo.size(); i++) {
            int actual = toDo.get(i).getTwists();
            if(actual != level) {
                level = actual;
                s += "\n [" + level + "]";
            }
            s += (toDo.get(i).getTwists() + " ");
        }
        System.out.println("Ibis[" + myIntIbisId + "] ----- TREE-----" + s);

    }

    /*synchronized public static boolean availableWork() {
        System.out.println("Ibis[" + myIntIbisId + "] -> SIZE: " + toDo.size());
    }*/

    public static void add(Cube cube) {
        toDo.add(cube);
        nodesOnTree++;
    }

    /*synchronized public static ArrayList<Cube> getFromLevel(int level) {
        toDo = toDoTree.get(level);
        return toDo.remove(toDo.size() - 1);
    }*/

    public static Cube getFromPool () {
        if(nodesOnTree == 0) {
            return null;
        }
        nodesOnTree--;
        return toDo.remove(0);
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

    public static int solutionInitial(Cube cube, CubeCache cache, ArrayList<Cube> toDo) {
        valuatedCubes++;
        //System.out.println("Ibis[" + myIntIbisId + "] -> solution");
        if (cube.isSolved()) {
            //System.out.println("SOLVED");

            return 1;
        }

        //generate childrens
        //System.out.println("Ibis[" + myIntIbisId + "] -> cube: " + cube.getTwists());
        Cube[] children = cube.generateChildren(cache);
        //System.out.println("Ibis[" + myIntIbisId + "] -> child: " + children[0].getTwists());
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
        while((cube = getFromPool()) != null) {
            result += solution(cube, cache);

            /*------------------ADD HERE---------------------------------------*/
            if(cube != initialCube) {
                cache.put(cube);
            }

        }
        //System.out.println("Ibis[" + myIntIbisId + "] -> solutionsWorkers -> FIrstTermination");
        return result;
    }

    public static int solutionsServer(ReceivePort resultsReceiver) throws Exception {
        //System.out.println("Ibis[" + myIntIbisId + "] -> SolutionsServer");
        int i;
        int result = solutionsWorkers();
        System.out.println("Ibis[" + myIntIbisId + "] -> valuatedCubes: " + valuatedCubes);
        //workManager.printSize();
        valuatedCubes = 0;
        System.out.println("Ibis[" + myIntIbisId + "] -> Wait results from other cubes");
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
        int results = 0;
        int i;
        for(i = 0; i < n; i++) {
            Cube cube = toDo.remove(0);
            results += solutionInitial(cube, cache, toDo);
        }
        //System.out.println(myIbisId + " -> SECOND " + toDo.size() + " cubes");
        return results;
    }

    public static int generateAnotherLevel(Cube cube, CubeCache cache, ArrayList<Cube> toDo) {
        int n = toDo.size();
        int results = 0;
        int i;
        solutionInitial(cube, cache, toDo);
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
        int bound = 1;
        int result = 0;
        int i, j;
        /*ArrayList<Cube> work = workManager.getWork(true);
        Cube cube = work.get(0);*/
        //System.out.println("SolutionsServer");

        ArrayList<Cube> initialToDo;
        WriteMessage task;

        while(result == 0) {
            bound++;
            System.out.println(myIbisId + "-> BOUND: " + bound);
            initialCube.setBound(bound);
            initialToDo = new ArrayList<Cube>();
            result = generateFirstLevel(initialCube, cache, initialToDo);
            System.out.println(myIbisId + "-> SIZE1: " + initialToDo.size());
            if(result == 0) {
                System.out.println(myIbisId + "generateSecondLevel");
                result = generateSecondLevel(cache, initialToDo);
                if(result != 0) {
                    bound = 2;
                }
            } else {
                bound = 1;
            }
            if(result != 0) {
                continue;
            }

            System.out.println(myIbisId + "-> SIZE2: " + initialToDo.size());

            String s = "";
            for(i = 0; i < initialToDo.size(); i++) {
                s += (" " + initialToDo.get(i).getTwists());
            }
            System.out.println("TODO: " + s);

            for(j = 0; j < 3; j++) {
                int m = initialToDo.size() / nodes;
                int r = initialToDo.size() % nodes;

                if(j != 2) {
                    int startIndex = m * myIntIbisId;

                    for(i = 0; i < startIndex; i++) {
                        initialToDo.remove(0);
                    }

                    for(i = 0; i < m; i++) {
                        add(initialToDo.remove(0));
                    }

                    for(i = 0; i < (nodes - 1 - myIntIbisId) * m; i++) {
                        initialToDo.remove(0);
                    }

                    if(r != 0) {
                        for(i = 0; i < r; i++) {
                            generateAnotherLevel(initialToDo.remove(0), cache, initialToDo);
                        }
                    } else {
                        break;
                    }
                } else {
                    int[] cubes_per_proc = new int[nodes];
                    int[] displs = new int[nodes];
                    int avarage_cubes_per_proc = initialToDo.size() / nodes;
                    int rem = initialToDo.size() % nodes;
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
                    int mydisp = displs[myIntIbisId];
                    for(i = 0; i < displs[myIntIbisId]; i++) {
                        initialToDo.remove(0);
                    }
                    for(i = 0; i < cubes_per_proc[myIntIbisId]; i++) {
                        add(initialToDo.remove(0));
                    }
                }
            }



            System.out.println(myIbisId + "-> SIZE3: " + nodesOnTree);
            printTree();

            //Thread.sleep(1000);
            System.out.println(" " + bound);
            result = solutionsServer(resultsReceiver);

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

        int result = 0;
        boolean end = false;
        int i, j;
        int bound = 1;

        ArrayList<Cube> initialToDo = new ArrayList<Cube>();

        while(!end) {
            bound++;
            System.out.println(myIbisId + "-> BOUND: " + bound);
            initialCube.setBound(bound);
            initialToDo = new ArrayList<Cube>();
            result = generateFirstLevel(initialCube, cache, initialToDo);
            System.out.println(myIbisId + "-> SIZE1: " + initialToDo.size());
            if(result == 0) {
                result = generateSecondLevel(cache, initialToDo);
                System.out.println(myIbisId + "-> SIZE2: " + initialToDo.size());
                if(result != 0) {
                    end = true;
                    continue;
                }
            } else {
                end = true;
                continue;
            }
            if(result != 0) {
                continue;
            }

            for(j = 0; j < 3; j++) {
                int m = initialToDo.size() / nodes;
                int r = initialToDo.size() % nodes;


                int startIndex = m * myIntIbisId;

                for(i = 0; i < startIndex; i++) {
                    initialToDo.remove(0);
                }

                for(i = 0; i < m; i++) {
                    add(initialToDo.remove(0));
                }

                for(i = 0; i < (nodes - 1 - myIntIbisId) * m; i++) {
                    initialToDo.remove(0);
                }

                if(r != 0) {
                    for(i = 0; i < r; i++) {
                        generateAnotherLevel(initialToDo.remove(0), cache, initialToDo);
                    }
                } else {
                    break;
                }
            }

            System.out.println(myIbisId + "-> SIZE3: " + nodesOnTree);
            printTree();

            result = solutionsWorkers();
            System.out.println("Ibis[" + myIntIbisId + "] -> valuatedCubes: "  + valuatedCubes);

            //workManager.printSize();
            valuatedCubes = 0;
            //communicate my results
            WriteMessage resultMessage = resultsSender.newMessage();
            resultMessage.writeInt(result);
            resultMessage.finish();

            System.out.println("Ibis[" + myIntIbisId + "] -> Wait continue from server");
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
        int i = 0;
        for (IbisIdentifier joinedIbis : joinedIbises) {
            System.err.println("Ibis joined: " + joinedIbis);
            if(joinedIbis.equals(myIbisId)) {
                myIntIbisId = i;
            }
            i++;
        }

        initialCube = generateCube();
        cache = new CubeCache(initialCube.getSize());



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