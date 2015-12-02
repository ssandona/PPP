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
public class Rubiks implements RegistryEventHandler {

    /*static PortType portType1 = new PortType(PortType.COMMUNICATION_RELIABLE,
                                     PortType.SERIALIZATION_DATA, PortType.RECEIVE_EXPLICIT,
                                     PortType.CONNECTION_ONE_TO_ONE);*/

    static PortType portType2 = new PortType(PortType.COMMUNICATION_RELIABLE,
            PortType.SERIALIZATION_DATA, PortType.RECEIVE_EXPLICIT,
            PortType.CONNECTION_MANY_TO_ONE);




    static IbisCapabilities ibisCapabilities = new IbisCapabilities(
        IbisCapabilities.ELECTIONS_STRICT, IbisCapabilities.MEMBERSHIP_TOTALLY_ORDERED, IbisCapabilities.TERMINATION);

    /*static IbisCapabilities ibisCapabilities = new IbisCapabilities(
        IbisCapabilities.ELECTIONS_STRICT);*/

    static int counter = 0;
    static ArrayList<Cube> toDo;
    //static ArrayList<String> done;
    static int result = 0;
    static int nodes = 1;
    static ArrayList<Cube[]> machines;
    static IbisIdentifier[] joinedIbises;
    static IbisIdentifier myIbisId;
    static Integer[] cubes_per_proc;
    static Integer[] displs;
    static Ibis myIbis;
    static Cube cube = null;
    static int id;

    public void joined(IbisIdentifier joinedIbis) {
        System.err.println("Got event from registry: " + joinedIbis
                           + " joined pool");
    }

    public void died(IbisIdentifier corpse) {
        System.err.println("Got event from registry: " + corpse + " died!");
    }

    public void left(IbisIdentifier leftIbis) {
        System.err.println("Got event from registry: " + leftIbis + " left");
    }

    public void electionResult(String electionName, IbisIdentifier winner) {
        System.err.println("Got event from registry: " + winner
                           + " won election " + electionName);
    }

    public void gotSignal(String signal, IbisIdentifier source) {
        System.err.println("Got event from registry: signal \"" + signal
                           + "\" from " + source);
    }

    public void poolClosed() {
        System.err.println("Got event from registry: pool closed");
    }

    public void poolTerminated(IbisIdentifier source) {
        System.err.println("Got event from registry: pool terminated by "
                           + source);
    }

    public static final boolean PRINT_SOLUTION = false;

    /**
     * Recursive function to find a solution for a given cube. Only searches to
     * the bound set in the cube object.
     *
     * @param cube
     *            cube to solve
     * @param cache
     *            cache of cubes used for new cube objects
     * @return the number of solutions found
     */

    private static int solutions(Cube cube, CubeCache cache, String s) {
        /*if(counter <= 1) {
            System.out.println(s+"AAA: Solutions -> cache size:" + cache.getSize());
        }*/
        if (cube.isSolved()) {
            return 1;
        }

        if (cube.getTwists() >= cube.getBound()) {
            /*if(counter <= 1) {
                System.out.println(s+"AAA: Twist>=Bound");
            }*/
            return 0;
        }

        /*if(counter <= 1) {
                System.out.println(s+"AAA: GenerateChildren");
            }*/
        // generate all possible cubes from this one by twisting it in
        // every possible way. Gets new objects from the cache
        Cube[] children = cube.generateChildren(cache);

        int result = 0;

        for (Cube child : children) {
            /*if(counter <= 1) {
                System.out.println(s+"AAA: Child");
            }*/
            // recursion step
            int childSolutions = solutions(child, cache, s + " ");
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
     * Solves a Rubik's cube by iteratively searching for solutions with a
     * greater depth. This guarantees the optimal solution is found. Repeats all
     * work for the previous iteration each iteration though...
     *
     * @param cube
     *            the cube to solve
     */


    public static int solutionsServer(Ibis ibis, Cube cube, CubeCache cache) throws Exception {
        System.out.println("SolutionsServer");
        ReceivePort resultsReceiver = ibis.createReceivePort(portType2, "results");
        resultsReceiver.enableConnections();
        SendPort taskSender = ibis.createSendPort(portType2);
        if (cube.isSolved()) {
            return 1;
        }

        /*if(counter <= 1) {
                System.out.println(s+"AAA: GenerateChildren");
            }*/
        // generate all possible cubes from this one by twisting it in
        // every possible way. Gets new objects from the cache
        /*Cube[] children = cube.generateChildren(cache);

        System.out.println("ChildrenGenerated");
        //work distribution
        machines = new ArrayList<Cube[]>();
        int last_displs = 0;
        int i = -1; int j;
        for (IbisIdentifier joinedIbis : joinedIbises) {
            i++;
            if(joinedIbis.equals(myIbisId)) {
                toDo = new ArrayList<Cube>(Arrays.asList(Arrays.copyOfRange(children, last_displs, displs[i])));
                last_displs = displs[i];
                continue;
            }
            Cube[] cubes=new Cube[12];
            Cube[] others=Arrays.copyOfRange(children, last_displs, displs[i])
            for(j=0;j<12;j++){
            	if(j<others.length){
            		cubes[j]=others[j];
            	}
            	else{
            		cubes[j]=null;
            	}
            }
            machines.add(cubes);
            last_displs = displs[i];
        }
        System.out.println("MachinesNumber: " + machines.size());
        Thread.sleep(5000);
        System.out.println("SendTasks");
        i = 0;
        /*for (IbisIdentifier joinedIbis : joinedIbises) {
            if(joinedIbis.equals(myIbisId)) {
                continue;
            }
                System.out.println("SendTaskToMachine");
                taskSender.connect(joinedIbis, "" + joinedIbis);
                // create a reply message
                WriteMessage task = taskSender.newMessage();
                task.writeArray(machines.get(i));
                task.finish();
                System.out.println("Sent");
        }*/

        for (IbisIdentifier joinedIbis : joinedIbises) {
            if(joinedIbis.equals(myIbisId)) {
                continue;
            }
            System.out.println("SendTaskToMachine");
            taskSender.connect(joinedIbis, "" + joinedIbis);
            // create a reply message
            WriteMessage task = taskSender.newMessage();
            task.writeObject(cube);
            task.finish();
            System.out.println("Sent");
        }

        System.out.println("ComputeMyPart");
        Cube[] children = cube.generateChildren(cache);
        int last_displs;
        if(id != 0) {
            last_displs = displs[id - 1];
        } else {
            last_displs = 0;
        }
        toDo = new ArrayList<Cube>(Arrays.asList(Arrays.copyOfRange(children, last_displs, displs[id])));


        //compute my part
        int result = 0; int i;
        while(!toDo.isEmpty()) {
            result += solutions(toDo.remove(0), cache, "");
        }
        System.out.println("CollectResults of " + nodes + " nodes");
        //collect results from other nodes
        for(i = 0; i < nodes - 1; i++) {
            ReadMessage r = resultsReceiver.receive();
            result += r.readInt();
            r.finish();
        }

        resultsReceiver.close();
        taskSender.close();
        return result;
    }


    private static void solveServer(Ibis ibis) throws Exception {


        int bound = 0;
        int result = 0;
        Cube cube = toDo.remove(0);
        CubeCache cache = new CubeCache(cube.getSize());

        while (result == 0) {
            bound++;
            cube.setBound(bound);
            System.out.print(" " + bound);
            result = solutionsServer(ibis, cube, cache);
        }

        System.out.println();
        System.out.println("Solving cube possible in " + result + " ways of "
                           + bound + " steps");

    }

    public static void solveWorkers(Ibis ibis, IbisIdentifier server) throws Exception {
        ReceivePort taskReceiver = ibis.createReceivePort(portType2, "" + myIbisId);
        taskReceiver.enableConnections();
        SendPort sender = ibis.createSendPort(portType2);
        Thread.sleep(1000);
        sender.connect(server, "results");
        System.out.println("ConnectedToServerPort");
        boolean first = true;
        int i;

        Cube cube = null;
        CubeCache cache = null;
        while(!ibis.registry().hasTerminated()) {
            //System.out.print("Bound now:");
            if(toDo.isEmpty()) {
                System.out.println("EmptyWorkQueueWait");
                ReadMessage r = taskReceiver.receive();
                System.out.println("ReceivedMyWork");
                cube=(Cube)r.readObject();
                r.finish();
            }
            if(first) {
                System.out.println("First");
                cache = new CubeCache(cube.getSize());
                first = false;
            }
            int result = 0;
            Cube[] children = cube.generateChildren(cache);
            int last_displs;
            if(id != 0) {
                last_displs = displs[id - 1];
            } else {
                last_displs = 0;
            }
            toDo = new ArrayList<Cube>(Arrays.asList(Arrays.copyOfRange(children, last_displs, displs[id])));
            while(!toDo.isEmpty()) {
                result += solutions(toDo.remove(0), cache, "");
            }
            System.out.println("CalculatedResult: " + result);
            // create a message
            WriteMessage resultMessage = sender.newMessage();
            resultMessage.writeInt(result);
            resultMessage.finish();
        }
        taskReceiver.close();
        sender.close();
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
        System.out.println("done");
        // Create an ibis instance.
        Ibis ibis = IbisFactory.createIbis(ibisCapabilities, null, portType2);
        Thread.sleep(1000);
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
                id = i;
            }
            i++;
        }
        cubes_per_proc = new Integer[nodes];
        displs = new Integer[nodes];
        int avarage_cubes_per_proc = 12 / nodes;
        int rem = 12 % nodes;
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

        // If I am the server, run server, else run client.
        if (server.equals(ibis.identifier())) {
            toDo.add(cube);
            long start = System.currentTimeMillis();
            solveServer(ibis);
            long end = System.currentTimeMillis();

            // NOTE: this is printed to standard error! The rest of the output is
            // constant for each set of parameters. Printing this to standard error
            // makes the output of standard out comparable with "diff"
            System.err.println("Solving cube took " + (end - start)
                               + " milliseconds");
            //terminate all workers
            // terminate the pool
            System.out.println("Terminating pool");
            ibis.registry().terminate();
            // wait for this termination to propagate through the system
            ibis.registry().waitUntilTerminated();


        } else {
            solveWorkers(ibis, server);
        }

        // End ibis.
        ibis.registry().terminate();
        ibis.end();
    }

    public static void main(String[] arguments) {

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
