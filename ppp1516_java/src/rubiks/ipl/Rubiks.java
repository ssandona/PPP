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

    

    public static void main(String[] arguments) {
        long i;
        Object lock = new Object();
        long sum=0;
        int cont=0;
        long start = System.currentTimeMillis();
        for(i=0;i<999999999;i++){
            sum+=i;
            cont++;
        }
        long end = System.currentTimeMillis();
        System.out.println("SUM -> "+sum+" count -> "+cont);
        System.err.println("Solving cube took " + (end - start)
                           + " milliseconds");
    }


}
