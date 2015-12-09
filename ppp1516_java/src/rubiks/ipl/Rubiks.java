package rubiks.ipl;

import ibis.ipl.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.IOException;

import java.lang.Math;


/**
 * Parallel solver for rubik's cube puzzle.
 *
 * @author Stefano Sandonà
 *
 */
public class Rubiks {

    static PortType portType1toM = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_DATA, PortType.RECEIVE_EXPLICIT,
            PortType.CONNECTION_ONE_TO_MANY);

    static PortType portTypeMto1 = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_DATA, PortType.RECEIVE_EXPLICIT,
            PortType.CONNECTION_MANY_TO_ONE, PortType.RECEIVE_POLL);

    static IbisCapabilities ibisCapabilities = new IbisCapabilities(
        IbisCapabilities.ELECTIONS_STRICT, IbisCapabilities.MEMBERSHIP_TOTALLY_ORDERED);

    static int nodes = 1;                   //number of nodes on the Ibis pool
    static IbisIdentifier[] joinedIbises;   //Ibises that joned the pool
    static int myIntIbisId;                 //Ibis id of the current Ibis instance
    static Cube initialCube = null;         //Initial Rubik Cube
    static CubeCache cache = null;          //CubeCache

    static String[] arguments;              //arguments provided by the user to generate the cube
    static int resultOnFirstPart;           //variable useful for the work splitting phase

    static int levelOfResult;               //variable useful for the work splitting phase

    public static final boolean PRINT_SOLUTION = false;

    static ArrayList<Cube> toDo = new ArrayList<Cube>();    //List of work to do

    /**
     * Print the queue of work that remains to do
     */
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

    /**
     * Function to get the next job (Rubik's Cube) from the work queue
     * @return the cube to evaluate or null if the queue is empty
     */
    public static Cube getFromPool () {
        if(toDo.size() == 0) {
            return null;
        }
        return toDo.remove(0);
    }

    /**
     * Recursive function to find a solution for a given cube. Only searches to
     * the bound set in the cube object.
     *
     * @param cube
     *            cube to solve
     * @return the number of solutions found for the subtree rooted in cube
     */
    private static int solutions(Cube cube) {
        if (cube.isSolved()) {
            return 1;
        }

        if (cube.getTwists() >= cube.getBound()) {
            return 0;
        }

        // generate all possible cubes from this one by twisting it in
        // every possible way. Gets new objects from the cache
        Cube[] children = cube.generateChildren(cache);

        int result = 0;

        for (Cube child : children) {
            // recursion step
            int childSolutions = solutions(child);
            if (childSolutions > 0) {
                result += childSolutions;
                if (PRINT_SOLUTION) {
                    child.print(System.err);
                }
            }
            // put child object in cache
            cache.put(child);
        }

        return result;
    }

    /**
     * Function called at the begin of the computation to insert in the
     * initial queue of work the children of a given cube
     *
     * @param cube
     *            cube to solve
     * @param initialToDo
     *            initial queue of work
     * @return if the cube was solved or not
     */
    public static int generateAnotherLevel(Cube cube, ArrayList<Cube> initialToDo) {
        if (cube.isSolved()) {
            return 1;
        }
        // generate all possible cubes from this one by twisting it in
        // every possible way. Gets new objects from the cache
        Cube[] children = cube.generateChildren(cache);
        int i;
        //add childrens on the pool
        for(i = 0; i < children.length; i++) {
            initialToDo.add(children[i]);
        }
        return 0;
    }

    public static int levelUntilExpand() {
        boolean ok = false;
        int n = 6*(initialCube.getSize()-1);
        int z = 0;
        while(Math.pow(n, z) / nodes < 101 && Math.pow(n, z) % nodes != 0) {
            //System.out.println("Math.pow(n, z) -> "+Math.pow(n, z));
            z = z + 1;
        }
        return z;
    }

    /**
     * Function called by all the workers sert in the
     * initial queue of work the children of a given cube
     * @return the number of solutions found for the subtrees rooted in the
     *         cubes of the assigned work queue
     */
    public static int solutionsWorkers() throws Exception {
        ArrayList<Cube> actual;
        int result = 0;
        int i;
        Cube cube;
        boolean end = false;
        while((cube = getFromPool()) != null) {
            result += solutions(cube);

        }
        return result;
    }

    /**
     * Function called by the server Ibis instance to calculate its part of the solution and
     * collect results from other nodes
     * @param resultsReceiver
     *            port on wich read the results from other nodes
     * @return the cumulative results
     */
    public static int solutionsServer(ReceivePort resultsReceiver) throws Exception {
        int i;
        //calculate the results for the assigned part of the tree
        int result = solutionsWorkers();
        //wait results from other nodes
        for(i = 0; i < nodes - 1; i++) {
            ReadMessage r = resultsReceiver.receive();
            result += r.readInt();
            r.finish();
        }
        return result;
    }

    /**
     * Function called to generate the initial cube from the parameters passed by the user
     * @return the generated cube
     */
    public static Cube generateCube() {
        Cube cube = null;

        // default parameters of puzzle
        int size = 3;
        int twists = 11;
        int seed = 0;
        String fileName = null;

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

    /**
     * Function called at the begin to split the work as fairly as possible among Ibis instances.
     * @return if the solution of the cube was found during the splitting phase
     */
    public static boolean splitTheWork() {
        int z = levelUntilExpand();
        System.out.println("Level untile expand -> "+z);
        ArrayList<Cube> initialToDo = new ArrayList<Cube>();
        //add the initial cube to the initial work queue (root of the tree)
        initialToDo.add(initialCube);
        int result = 0;
        int i, j;

        boolean levelFound = false;
        boolean terminated = false;
        levelOfResult = -1;
        int n = initialCube.getSize();
        for(i = 0; i < z; i++) {
            resultOnFirstPart = 0;
            int levelCubes = (int)Math.pow(n, z);
            for(j = 0; j < levelCubes; j++) {
                resultOnFirstPart += generateAnotherLevel(initialToDo.remove(0), initialToDo);
            }
            levelOfResult = z;
            if(resultOnFirstPart != 0) {
                return true;
            }
        }


        /*find the first tree level with more nodes than ibis instances. Split the nodes fairly
        among the N ibis instances. If some nodes have left out (N is not a divisor of the
        number of nodes of this level), these are expanded to the next tree level, otherwise
        the splitting phase is terminated*/

        int m = initialToDo.size() / nodes;
        int r = initialToDo.size() % nodes;

        int startIndex = m * myIntIbisId;

        for(i = 0; i < startIndex; i++) {
            initialToDo.remove(0);
        }

        for(i = 0; i < m; i++) {
            toDo.add(initialToDo.remove(0));
        }

        for(i = 0; i < (nodes - 1 - myIntIbisId) * m; i++) {
            initialToDo.remove(0);
        }

        if(r != 0) {
            for(i = 0; i < r; i++) {
                generateAnotherLevel(initialToDo.remove(0), initialToDo);
            }
        } else {
            terminated = true;
        }



        /*if we have expanded some nodes, we try to split them among the N ibis instances.
        Until they are less than the number of ibis instances we generate another level of
        the tree from them, when they are enough, we split them as fairly as possible*/
        while(!terminated) {
            m = initialToDo.size() / nodes;
            r = initialToDo.size() % nodes;
            if(m == 0) {
                int s = initialToDo.size();
                for(i = 0; i < s; i++) {
                    generateAnotherLevel(initialToDo.remove(0), initialToDo);
                }
                continue;
            } else {
                terminated = true;
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
                    toDo.add(initialToDo.remove(0));
                }
            }
        }
        return false;
    }

    /**
     * Function called by the server Ibis instance to do its part of work
     * @param ibis
     *            local Ibis instance
     */
    private static void solveServer(Ibis ibis) throws Exception {
        int bound = 0;
        int result = 0;
        int i, j;
        //port for receive results from other Ibis instances
        ReceivePort resultsReceiver = ibis.createReceivePort(portTypeMto1, "results");
        resultsReceiver.enableConnections();

        //port for send to the other Ibis instances if the work is finish or not
        SendPort terminationSender = ibis.createSendPort(portType1toM);
        for (IbisIdentifier joinedIbis : joinedIbises) {
            if(joinedIbis.equals(ibis.identifier())) {
                continue;
            }
            terminationSender.connect(joinedIbis, "continue");
        }

        long start = System.currentTimeMillis();
        System.out.print("Bound now:");

        //continue to iterate until a solution is found
        while(result == 0) {
            bound++;
            initialCube.setBound(bound);
            if(splitTheWork()) {
                result = resultOnFirstPart;
                bound = levelOfResult;
                continue;
            }
            System.out.print(" " + bound);
            result = solutionsServer(resultsReceiver);
            //if the solution found for this bounds are zero, communicate to the ibis instances
            //that the next bound has to be evaluated
            if(result == 0) {
                WriteMessage task = terminationSender.newMessage();
                task.writeBoolean(false);
                task.finish();
            }
        }

        long end = System.currentTimeMillis();
        System.out.println("Solving cube possible in " + result + " ways of "
                           + bound + " steps");

        System.err.println("Solving cube took " + (end - start)
                           + " milliseconds");

        //say to all the ibis instances that the work is finished (if we have not found
        //a solution splitting the work at the begin, otherwise they already know that
        //the work is finished)
        if(resultOnFirstPart == 0) {
            WriteMessage task = terminationSender.newMessage();
            task.writeBoolean(true);
            task.finish();
        }

        terminationSender.close();
        Thread.sleep(2000); //wait for safety
        resultsReceiver.close();
    }


    public static void solveWorkers(Ibis ibis, IbisIdentifier server) throws Exception {

        int result = 0;
        boolean end = false;
        int i, j;
        int bound = 0;

        //port for receive if the next bound has to be evaluated or the work is finished
        ReceivePort terminationReceiver = ibis.createReceivePort(portType1toM, "continue");
        terminationReceiver.enableConnections();

        //port to send to the server the solutions find for the local assigned subtree
        SendPort resultsSender = ibis.createSendPort(portTypeMto1);
        resultsSender.connect(server, "results");

        while(!end) {
            bound++;
            initialCube.setBound(bound);
            if(splitTheWork()) {
                result = resultOnFirstPart;
                break;
            }
            result = solutionsWorkers();
            //communicate local results to the server
            WriteMessage resultMessage = resultsSender.newMessage();
            resultMessage.writeInt(result);
            resultMessage.finish();

            //check if the work is finished
            ReadMessage r = terminationReceiver.receive();
            end = r.readBoolean();
            r.finish();
        }

        resultsSender.close();
        Thread.sleep(2000); //wait for safety
        terminationReceiver.close();
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
     * Initial called function
     */
    private void run() throws Exception {
        // Create an ibis instance.
        Ibis ibis = IbisFactory.createIbis(ibisCapabilities, null, portTypeMto1, portType1toM);
        Thread.sleep(5000); //wait for safety that all the ibises join the pool

        // Elect a server
        IbisIdentifier server = ibis.registry().elect("Server");

        System.out.println("Server is " + server);

        //get the list of the ibises that joined the pool, in order to know the number of Ibis
        //instances involved in the calculation and the integer id of the local ibis on the pool
        joinedIbises = ibis.registry().joinedIbises();
        nodes = joinedIbises.length;
        int i = 0;
        for (IbisIdentifier joinedIbis : joinedIbises) {
            System.err.println("Ibis joined: " + joinedIbis);
            if(joinedIbis.equals(ibis.identifier())) {
                myIntIbisId = i;
            }
            i++;
        }

        //generate the initial cube and from this initialize the cache
        initialCube = generateCube();
        cache = new CubeCache(initialCube.getSize());

        // If I am the server, run solveServer, else run solveWorkers.
        if (server.equals(ibis.identifier())) {
            solveServer(ibis);
        } else {
            solveWorkers(ibis, server);
        }
        ibis.end();
    }

    /**
         * Main function.
         *
         * @param arguments
         *            list of arguments
         */
    public static void main(String[] argumentsForCube) {
        arguments = argumentsForCube;
        try {
            new Rubiks().run();
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }

}