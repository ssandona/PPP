/*
    N-Body simulation code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

extern double   sqrt(double);
extern double   atan2(double, double);

#define GRAVITY     1.1
#define FRICTION    0.01
#define MAXBODIES  10000
#define DELTA_T     (0.025/5000)
#define BOUNCE      -0.9
#define SEED        27102015

typedef struct {
    double x[2];        /* Old and new X-axis coordinates */
    double y[2];        /* Old and new Y-axis coordinates */
    double xf;          /* force along X-axis */
    double yf;          /* force along Y-axis */
    double xv;          /* velocity along X-axis */
    double yv;          /* velocity along Y-axis */
    double mass;        /* Mass of the body */
    double radius;      /* width (derived from mass) */
} bodyType;

typedef struct {
    double xf;          /* force along X-axis */
    double yf;          /* force along Y-axis */
} forceType;

int globalStartB;
int globalStartC;
forceType *forces;      /*list of forces per body*/
forceType *new_forces;  /*ausiliar list of forces (MPI reduce)*/
int bodyCt;             /*number of bodies*/
int old = 0;            /* Flips between 0 and 1 */
bodyType *bodies;       /*list of bodies*/
int *displs;            /*list of the starting indexes of forces assigned per processor*/
int *forces_per_proc;   /*list of the number of forces assigned per processor*/
int myid;               /*MPI process ID*/
int printed = 0;
int numprocs;           /*number of MPI processes involeved in the computation*/
MPI_Op mpi_sum;
double totalNumberOfForcesComputed = 0;

/*  Macros to hide memory layout
*/
#define X(B)       bodies[B].x[old]
#define XN(B)      bodies[B].x[old^1]
#define Y(B)       bodies[B].y[old]
#define YN(B)      bodies[B].y[old^1]
#define XF(B)      forces[B].xf
#define YF(B)      forces[B].yf
#define XV(B)      bodies[B].xv
#define YV(B)      bodies[B].yv
#define R(B)       bodies[B].radius
#define M(B)       bodies[B].mass

/*  Dimensions of space (very finite, ain't it?)
*/
int     xdim = 0;
int     ydim = 0;


void
clear_forces(void) {
    int b;

    /* Clear force accumulation variables */
    for (b = 0; b < bodyCt; ++b) {
        YF(b) = (XF(b) = 0);
    }
}

void
compute_forces(void) {
    int b, c;
    int count = 0;
    /* Incrementally accumulate forces from each assigned body pair,
       skipping force of body on itself (c == b). The first loop is
       separated to avoid an additional if construct
    */
    b = globalStartB;
    for (c = globalStartC; c < bodyCt && count < forces_per_proc[myid]; ++c) {
        double dx = X(c) - X(b);
        double dy = Y(c) - Y(b);
        double angle = atan2(dy, dx);
        double dsqr = dx * dx + dy * dy;
        double mindist = R(b) + R(c);
        double mindsqr = mindist * mindist;
        double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
        double force = M(b) * M(c) * GRAVITY / forced;
        double xf = force * cos(angle);
        double yf = force * sin(angle);

        /* Slightly sneaky...
           force of b on c is negative of c on b;
        */
        XF(b) += xf;
        YF(b) += yf;
        XF(c) -= xf;
        YF(c) -= yf;

        count++;
        totalNumberOfForcesComputed++;
    }

    /*standard loop*/

    for (b = globalStartB + 1; b < bodyCt && count < forces_per_proc[myid]; ++b) {
        for (c = b + 1; c < bodyCt && count < forces_per_proc[myid]; ++c) {
            double dx = X(c) - X(b);
            double dy = Y(c) - Y(b);
            double angle = atan2(dy, dx);
            double dsqr = dx * dx + dy * dy;
            double mindist = R(b) + R(c);
            double mindsqr = mindist * mindist;
            double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
            double force = M(b) * M(c) * GRAVITY / forced;
            double xf = force * cos(angle);
            double yf = force * sin(angle);

            /* Slightly sneaky...
               force of b on c is negative of c on b;
            */
            XF(b) += xf;
            YF(b) += yf;
            XF(c) -= xf;
            YF(c) -= yf;

            count++;
            totalNumberOfForcesComputed++;
        }
    }
}



/**
     * Function called at the begin to calculate the
     * initial indexes for the assigned forces chunk
*/
void
calculateAssignedForces() {
    int b, c;
    int count = 0;
    for (b = 0; b < bodyCt; ++b) {
        for (c = b + 1; c < bodyCt; ++c) {
            if(count == displs[myid]) {
                globalStartB = b;
                globalStartC = c;
                b = bodyCt;
            }
            count++;

        }
    }
}

void
compute_velocities(void) {
    int b;
    for (b = 0; b < bodyCt; ++b) {
        double xv = XV(b);
        double yv = YV(b);
        double force = sqrt(xv * xv + yv * yv) * FRICTION;
        double angle = atan2(yv, xv);
        double xf = XF(b) - (force * cos(angle));
        double yf = YF(b) - (force * sin(angle));

        XV(b) += (xf / M(b)) * DELTA_T;
        YV(b) += (yf / M(b)) * DELTA_T;
    }
}

void
compute_positions(void) {
    int b;
    for (b = 0; b < bodyCt; ++b) {
        double xn = X(b) + (XV(b) * DELTA_T);
        double yn = Y(b) + (YV(b) * DELTA_T);

        /* Bounce of image "walls" */
        if (xn < 0) {
            xn = 0;
            XV(b) = -XV(b);
        } else if (xn >= xdim) {
            xn = xdim - 1;
            XV(b) = -XV(b);
        }
        if (yn < 0) {
            yn = 0;
            YV(b) = -YV(b);
        } else if (yn >= ydim) {
            yn = ydim - 1;
            YV(b) = -YV(b);
        }

        /* Update position */
        XN(b) = xn;
        YN(b) = yn;
    }
}


/*  Graphic output stuff...
*/

#include <fcntl.h>
#include <sys/mman.h>

int     fsize;
unsigned char   *map;
unsigned char   *image;


unsigned char *
map_P6(char *filename,
       int *xdim,
       int *ydim) {
    /* The following is a fast and sloppy way to
       map a color raw PPM (P6) image file
    */
    int fd;
    unsigned char *p;
    int maxval;

    /* First, open the file... */
    if ((fd = open(filename, O_RDWR)) < 0) {
        return((unsigned char *) 0);
    }

    /* Read size and map the whole file... */
    fsize = lseek(fd, ((off_t) 0), SEEK_END);
    map = ((unsigned char *)
           mmap(0,      /* Put it anywhere */
                fsize,  /* Map the whole file */
                (PROT_READ | PROT_WRITE),   /* Read/write */
                MAP_SHARED, /* Not just for me */
                fd,     /* The file */
                0));    /* Right from the start */
    if (map == ((unsigned char *) - 1)) {
        close(fd);
        return((unsigned char *) 0);
    }

    /* File should now be mapped; read magic value */
    p = map;
    if (*(p++) != 'P') goto ppm_exit;
    switch (*(p++)) {
    case '6':
        break;
    default:
        goto ppm_exit;
    }

#define Eat_Space \
    while ((*p == ' ') || \
           (*p == '\t') || \
           (*p == '\n') || \
           (*p == '\r') || \
           (*p == '#')) { \
        if (*p == '#') while (*(++p) != '\n') ; \
        ++p; \
    }

    Eat_Space;      /* Eat white space and comments */

#define Get_Number(n) \
    { \
        int charval = *p; \
 \
        if ((charval < '0') || (charval > '9')) goto ppm_exit; \
 \
        n = (charval - '0'); \
        charval = *(++p); \
        while ((charval >= '0') && (charval <= '9')) { \
            n *= 10; \
            n += (charval - '0'); \
            charval = *(++p); \
        } \
    }

    Get_Number(*xdim);  /* Get image width */

    Eat_Space;      /* Eat white space and comments */
    Get_Number(*ydim);  /* Get image width */

    Eat_Space;      /* Eat white space and comments */
    Get_Number(maxval); /* Get image max value */

    /* Should be 8-bit binary after one whitespace char... */
    if (maxval > 255) {
ppm_exit:
        close(fd);
        munmap(map, fsize);
        return((unsigned char *) 0);
    }
    if ((*p != ' ') &&
            (*p != '\t') &&
            (*p != '\n') &&
            (*p != '\r')) goto ppm_exit;

    /* Here we are... next byte begins the 24-bit data */
    return(p + 1);

    /* Notice that we never clean-up after this:

       close(fd);
       munmap(map, fsize);

       However, this is relatively harmless;
       they will go away when this process dies.
    */
}

#undef  Eat_Space
#undef  Get_Number

static inline void
color(int x, int y, int b) {
    unsigned char *p = image + (3 * (x + (y * xdim)));
    int tint = ((0xfff * (b + 1)) / (bodyCt + 2));

    p[0] = (tint & 0xf) << 4;
    p[1] = (tint & 0xf0);
    p[2] = (tint & 0xf00) >> 4;
}

static inline void
black(int x, int y) {
    unsigned char *p = image + (3 * (x + (y * xdim)));

    p[2] = (p[1] = (p[0] = 0));
}

void
display(void) {
    double i, j;
    int b;

    /* For each pixel */
    for (j = 0; j < ydim; ++j) {
        for (i = 0; i < xdim; ++i) {
            /* Find the first body covering here */
            for (b = 0; b < bodyCt; ++b) {
                double dy = Y(b) - j;
                double dx = X(b) - i;
                double d = sqrt(dx * dx + dy * dy);

                if (d <= R(b) + 0.5) {
                    /* This is it */
                    color(i, j, b);
                    goto colored;
                }
            }

            /* No object -- empty space */
            black(i, j);

colored:
            ;
        }
    }
}

void
print(void) {
    int b;
    for (b = 0; b < bodyCt; ++b) {
        printf("%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n", X(b), Y(b), XF(b), YF(b), XV(b), YV(b));
    }
}

/*void
print_forces(void) {
    int b;
    for (b = 0; b < bodyCt; ++b) {
        printf("%10.3f %10.3f\n", XF(b), YF(b));
    }
}*/

/**
     * Function called for the MPI reduce operation
     *
     * @param in
     *            array of forces to sum
     * @param inout
     *            array of forces to which sum the 'in' array
     * @param dtype
     *            datatype
*/
void sumForces(forceType *in, forceType *inout, int *len, MPI_Datatype *dtype) {
    int i;
    for (i = 0; i < *len; ++i) {
        inout->xf += in->xf;
        inout->yf += in->yf;
        in++;
        inout++;
    }
}


/*  Main program...
*/

int
main(int argc, char **argv) {
    unsigned int lastup = 0;
    unsigned int secsup;
    int b;
    int steps;
    double rtime;
    struct timeval start;
    struct timeval end;
    int i;
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];    //processors involved on the computation


    if (argc != 5) {
        fprintf(stderr,
                "Usage: %s num_bodies secs_per_update ppm_output_file steps\n",
                argv[0]);
        exit(1);
    }

    if ((bodyCt = atol(argv[1])) > MAXBODIES ) {
        fprintf(stderr, "Using only %d bodies...\n", MAXBODIES);
        bodyCt = MAXBODIES;
    } else if (bodyCt < 2) {
        fprintf(stderr, "Using two bodies...\n");
        bodyCt = 2;
    }

    bodies = malloc(sizeof(bodyType) * bodyCt);
    forces = malloc(sizeof(forceType) * bodyCt);

    //forces initialization
    for(i = 0; i < bodyCt; i++) {
        XF(i) = 0;
        YF(i) = 0;
    }

    secsup = atoi(argv[2]);
    image = map_P6(argv[3], &xdim, &ydim);
    steps = atoi(argv[4]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);

    fprintf(stderr, "Process %d on %s\n", myid, processor_name);

    fprintf(stderr, "Running N-body with %i bodies and %i steps\n", bodyCt, steps);

    /* Initialize simulation data */
    if(myid == 0) {
        srand(SEED);
        for (b = 0; b < bodyCt; ++b) {
            X(b) = (rand() % xdim);
            Y(b) = (rand() % ydim);
            R(b) = 1 + ((b * b + 1.0) * sqrt(1.0 * ((xdim * xdim) + (ydim * ydim)))) /
                   (25.0 * (bodyCt * bodyCt + 1.0));
            M(b) = R(b) * R(b) * R(b);
            XV(b) = ((rand() % 20000) - 10000) / 2000.0;
            YV(b) = ((rand() % 20000) - 10000) / 2000.0;
        }
    }

    //calculate the number of total forces to compute*/
    int forceCt = 0;
    for(i = 0; i < bodyCt; i++) {
        forceCt += i;
    }

    forces_per_proc = malloc(sizeof(int) * numprocs);
    displs = malloc(sizeof(int) * numprocs);
    int rem = forceCt % numprocs; // elements remaining after division among processes
    fprintf(stderr, "Total Forces => %d\n", forceCt);

    /*custom MPI dataType for the Forces distribution*/
    const int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype mpi_force_type;
    MPI_Aint     offsets[2];

    offsets[0] = offsetof(forceType, xf);
    offsets[1] = offsetof(forceType, yf);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_force_type);
    MPI_Type_commit(&mpi_force_type);


    /*custom MPI dataType for the Bodies distribution*/
    const int nitems2 = 6;
    int blocklengths2[6] = {2, 2, 1, 1, 1, 1};
    MPI_Datatype types2[6] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,};
    MPI_Datatype mpi_body_type;
    MPI_Aint     offsets2[6];

    offsets2[0] = offsetof(bodyType, x);
    offsets2[1] = offsetof(bodyType, y);
    offsets2[2] = offsetof(bodyType, xv);
    offsets2[3] = offsetof(bodyType, yv);
    offsets2[4] = offsetof(bodyType, mass);
    offsets2[5] = offsetof(bodyType, radius);

    MPI_Type_create_struct(nitems2, blocklengths2, offsets2, types2, &mpi_body_type);
    MPI_Type_commit(&mpi_body_type);

    /*custom MPI reduce operation*/
    MPI_Op_create((MPI_User_function *) sumForces, 1, &mpi_sum);


    /*calculate the forces to assign to each process and the displacements*/
    int avarage_forces_per_proc = forceCt / numprocs;
    int sum = 0;
    for (i = 0; i < numprocs; i++) {
        forces_per_proc[i] = avarage_forces_per_proc;
        if (rem > 0) {
            forces_per_proc[i]++;
            rem--;
        }
        displs[i] = sum;
        sum += forces_per_proc[i];
    }

    for (i = 0; i < numprocs; i++) {
        fprintf(stderr, "[%d] Forces -> %d, displ -> %d \n", myid, forces_per_proc[i], displs[i]);
    }


    calculateAssignedForces();

    fprintf(stderr, "B -> %d, C -> %d \n", globalStartB, globalStartC);

    /*broadcast the bodies to all the processes*/
    MPI_Bcast(bodies, bodyCt, mpi_body_type, 0, MPI_COMM_WORLD);

    int cont;

    if(gettimeofday(&start, 0) != 0) {
        fprintf(stderr, "could not do timing\n");
        exit(1);
    }

    while (steps--) {
        cont = 0;
        clear_forces();
        compute_forces();

        /*Reduce all the forces calculated by each process*/
        new_forces = malloc(sizeof(forceType) * bodyCt);
        MPI_Allreduce(forces, new_forces, bodyCt, mpi_force_type, mpi_sum, MPI_COMM_WORLD);
        free(forces);
        forces = new_forces;

        compute_velocities();
        compute_positions();
        /*if(0 == myid) {
            print_forces();
            printf("------step %d-----\n",steps);
        }*/

        old ^= 1;

    }
    if(0 == myid) {
        if(gettimeofday(&end, 0) != 0) {
            fprintf(stderr, "could not do timing\n");
            exit(1);
        }
        rtime = (end.tv_sec + (end.tv_usec / 1000000.0)) -
                (start.tv_sec + (start.tv_usec / 1000000.0));


    }

    if(0 == myid) {
        print();
        fprintf(stderr, "fine\n");
        fprintf(stderr, "N-body took %10.3f seconds\n", rtime);
    }

    fprintf(stderr, "Process %d compute %d forces\n, assigned %d bodies", myid, totalNumberOfForcesComputed, forces_per_proc[myid]);

    MPI_Finalize();


    free(forces_per_proc);
    free(displs);
    //free(bodies);
    free(forces);
    //free(new_forces);

    return 0;
}
