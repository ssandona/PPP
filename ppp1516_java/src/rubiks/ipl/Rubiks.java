package rubiks.ipl;

import ibis.ipl.*;
import java.util.ArrayList;

/**
 * Solver for rubik's cube puzzle.
 *
 * @author Niels Drost, Timo van Kessel
 *
 */
public class Rubiks {

    PortType portType = new PortType(PortType.COMMUNICATION_RELIABLE,
                                     PortType.SERIALIZATION_DATA, PortType.RECEIVE_EXPLICIT,
                                     PortType.CONNECTION_ONE_TO_ONE);


    IbisCapabilities ibisCapabilities = new IbisCapabilities(
        IbisCapabilities.ELECTIONS_STRICT, IbisCapabilities.MEMBERSHIP_TOTALLY_ORDERED);

    static int counter = 0;
    static ArrayList<Cube> toDo;
    static ArrayList<String> done;
    static int result = 0;
    static int nodes = 1;
    static ArrayList<ArrayList<Cube>> machines;
    static IbisIdentifier[] joinedIbises;
    static IbisIdentifier myId;
    static SendPort sender;

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
    private static int solutions(Cube cube, CubeCache cache, int twistSolution, String s) {
        /*if(counter <= 1) {
            System.out.println(s+"AAA: Solutions -> cache size:" + cache.getSize());
        }*/
        if (cube.isSolved()) {
            return 1;
        }

        if(cube.getTwist() >= twistSolution)
            return 0;


        // generate all possible cubes from this one by twisting it in
        // every possible way. Gets new objects from the cache
        Cube[] children = cube.generateChildren(cache);

        machines = new ArrayList<ArrayList<Cube>>();
        //

        for (Cube child : children) {
            switch(child.hash() % nodes) {
            case(myId): {
                toDo.add(child);
                break;
            }
            case(0):
            {machines[0].add(child); break;},
            case(1):
            {machines[1].add(child); break;},
            case(2):
            {machines[2].add(child); break;},
            case(3):
            {machines[3].add(child); break;},
            case(4):
            {machines[4].add(child); break;},
            case(5):
            {machines[5].add(child); break;},
            case(6):
            {machines[6].add(child); break;},
            case(7):
            {machines[7].add(child); break;},
            case(8):
            {machines[8].add(child); break;},
            case(9):
            {machines[9].add(child); break;},
            case(10):
            {machines[10].add(child); break;},
            case(11):
            {machines[11].add(child); break;},
            }

            int i = 0;
            for (IbisIdentifier joinedIbis : joinedIbises) {
                if(joinedIbis != myIbisId && !machines[i].isEmpty()) {
                    sender.connect(joinedIbis, "" + joinedIbis);
                    // create a reply message
                    WriteMessage reply = sender.newMessage();
                    reply.writeObject(machines[i]);
                    reply.finish();
                }
                i++;
            }

        }
        // put child object in cache
        cache.put(child);
        return 0;
    }

    // Create a send port for sending requests and connect.




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
private static void solve(Ibis ibis, boolean server) {
    // cache used for cube objects. Doing new Cube() for every move
    // overloads the garbage collector
    boolean end = false;
    CubeCache cache;
    //int bound = 0;

    // Create a receive port and enable connections.
    ReceivePort taskReceiver = ibis.createReceivePort(portType, "" + myId);
    taskReceiver.enableConnections();
    sender = ibis.createSendPort(portType);
    ReceivePort resultsReceiver;
    if(myId == server) {
        resultsReceiver = myIbis.createReceivePort(portType, "results");
        taskReceiver.enableConnections();
    }

    if(myId == server) {

    }

    //System.out.print("Bound now:");
    if(toDo.isEmpty()) {
        // Read the message.
        ReadMessage r = taskReceiver.receive();
        toDo = (ArrayList<Cube>)r.readObject()();
        r.finish();
        //System.out.println("Server received: " + s);

        if(toDo.get(0).getClass().equals(NullCube.class)) {
            end = true;
        } else {
            cache = new CubeCache(toDo.get(0).getSize());
        }
    } else {
        cache = new CubeCache(toDo.get(0).getSize());
    }

    Cube actual=toDo.get(0);
    int twistSolution = -1;
    while (result == 0) {
        if(done.indexOf(actual.hash()) == -1) {
            done.add(actual.hash());
            actual = toDo.remove(0);
            result = solutions(actual, cache, twistSolution, "");
            if(result != 0) {
                break;
            }
        }
        if(toDo.isEmpty()) {
            // Read the message.
            ReadMessage r = receiver.receive();
            toDo = (ArrayList<Cube>)r.readObject()();
            r.finish();
            //System.out.println("Server received: " + s);

            if(toDo.get(0).isSolved()) {
                twistSolution = toDo.get(0).getTwist();
                //send to server that we have received the last cube
            }
        }


        if(myId != server && result > 0) {

            for (IbisIdentifier joinedIbis : joinedIbises) {
                if(joinedIbis != myIbisId) {
                    sender.connect(joinedIbis, "" + joinedIbis);
                    // create a reply message
                    WriteMessage reply = sender.newMessage();
                    reply.writeObject(actual);
                    reply.finish();
                }
                i++;
            }
            sender.connect(server, "results");
            // create a reply message
            WriteMessage res = sender.newMessage();
            res.writeInt(actual.getTwist());
            reply.finish();
        }





        //send "end" to all

        //collect results from all and select best (until twist of the winner)

        System.out.println();
        System.out.println("Solving cube possible in " + result + " ways of "
                           + bound + " steps");

        //else send to beginner

        // Close receive port.
        receiver.close();
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
        // Create an ibis instance.
        Ibis ibis = IbisFactory.createIbis(ibisCapabilities, null, portType);
        myId = ibis.identifier();

        // Elect a server
        IbisIdentifier server = ibis.registry().elect("Server");

        System.out.println("Server is " + server);

        joinedIbises = ibis.registry().joinedIbises();
        nodes = joinedIbises.length;
        for (IbisIdentifier joinedIbis : joinedIbises) {
            System.err.println("Ibis joined: " + joinedIbis);
        }

        // If I am the server, run server, else run client.
        if (server.equals(ibis.identifier())) {
            long start = System.currentTimeMillis();
            solve(ibis, true);
            long end = System.currentTimeMillis();

            // NOTE: this is printed to standard error! The rest of the output is
            // constant for each set of parameters. Printing this to standard error
            // makes the output of standard out comparable with "diff"
            System.err.println("Solving cube took " + (end - start)
                               + " milliseconds");


        } else {
            solve(ibis, false);
        }

        // End ibis.
        ibis.end();
    }

    public static void main(String[] arguments) {

        Cube cube = null;

        // default parameters of puzzle
        int size = 3;
        int twists = 11;
        int seed = 0;
        String fileName = null;

        // number of threads used to solve puzzle
        // (not used in sequential version)

        //If beginner

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

        done = new ArrayList<String>();
        toDo = new ArrayList<Cube>();

        //if beginner
        toDo.add(cube);

        try {
            new Hello().run();
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }

    }

}
