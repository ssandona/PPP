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
    //double x[2];        /* Old and new X-axis coordinates */
    //double y[2];        /* Old and new Y-axis coordinates */
    // double xf;          /* force along X-axis */
    // double yf;          /* force along Y-axis */
    double xv;          /* velocity along X-axis */
    double yv;          /* velocity along Y-axis */
    double mass;        /* Mass of the body */
    double radius;      /* width (derived from mass) */
} bodyType;

typedef struct {
    double x[2];        /* Old and new X-axis coordinates */
    double y[2];        /* Old and new Y-axis coordinates */
} bodyPositionType;

typedef struct {
    double xf;          /* force along X-axis */
    double yf;          /* force along Y-axis */
} forceType;

int globalStartB;
int globalStartC;
int globalStopB;
int globalStopC;
bodyType bodies[MAXBODIES];
bodyPositionType positions[MAXBODIES];
forceType *forces;
forceType *new_forces;
forceType *new_forces2;
int bodyCt;
int old = 0;    /* Flips between 0 and 1 */
bodyType *new_bodies;
bodyPositionType *new_positions;
bodyType *new_bodies2;
bodyType *rec_bodies;
bodyPositionType *rec_positions;
int *displs;
int *displs2;
int *bodies_per_proc;
int *forces_per_proc;
int myid;
int printed = 0;
int numprocs;
MPI_Op mpi_sum;
int totalNumberOfForcesComputed = 0;

/*  Macros to hide memory layout
*/
#define X(B)        positions[B].x[old]
#define XN(B)       positions[B].x[old^1]
#define Y(B)        positions[B].y[old]
#define YN(B)       positions[B].y[old^1]
#define XF(B)       forces[B].xf
#define YF(B)       forces[B].yf
#define XV(B)       bodies[B].xv
#define YV(B)       bodies[B].yv
#define R(B)        bodies[B].radius
#define M(B)        bodies[B].mass

/*  Macros to hide memory layout
*/
#define _X(B)       new_positions[B].x[old]
#define _XN(B)      new_positions[B].x[old^1]
#define _Y(B)       new_positions[B].y[old]
#define _YN(B)      new_positions[B].y[old^1]
#define _XF(B)      new_forces[B].xf
#define _YF(B)      new_forces[B].yf
#define _XV(B)      new_bodies[B].xv
#define _YV(B)      new_bodies[B].yv
#define _R(B)       new_bodies[B].radius
#define _M(B)       new_bodies[B].mass

/*  Dimensions of space (very finite, ain't it?)
*/
int     xdim = 0;
int     ydim = 0;


void
clear_forces(void) {
    int b;

    /* Clear force accumulation variables */
    for (b = 0; b < bodyCt; ++b) {
        _YF(b) = (_XF(b) = 0);
    }
}

void
compute_forces(void) {
    int b, c;
    int count = 0;
    /* Incrementally accumulate forces from each body pair,
       skipping force of body on itself (c == b)
    */
    b = globalStartB;
    for (c = globalStartC; c < bodyCt && count<forces_per_proc[myid]; ++c) {
        double dx = _X(c) - _X(b);
        double dy = _Y(c) - _Y(b);
        double angle = atan2(dy, dx);
        double dsqr = dx * dx + dy * dy;
        double mindist = _R(b) + _R(c);
        double mindsqr = mindist * mindist;
        double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
        double force = _M(b) * _M(c) * GRAVITY / forced;
        double xf = force * cos(angle);
        double yf = force * sin(angle);

        /* Slightly sneaky...
           force of b on c is negative of c on b;
        */
        _XF(b) += xf;
        _YF(b) += yf;
        _XF(c) -= xf;
        _YF(c) -= yf;

        count++;
        totalNumberOfForcesComputed++;
    }

    for (b = globalStartB + 1; b < bodyCt && count<forces_per_proc[myid]; ++b) {
        for (c = b + 1; c < bodyCt && count<forces_per_proc[myid]; ++c) {
            double dx = _X(c) - _X(b);
            double dy = _Y(c) - _Y(b);
            double angle = atan2(dy, dx);
            double dsqr = dx * dx + dy * dy;
            double mindist = _R(b) + _R(c);
            double mindsqr = mindist * mindist;
            double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
            double force = _M(b) * _M(c) * GRAVITY / forced;
            double xf = force * cos(angle);
            double yf = force * sin(angle);

            /* Slightly sneaky...
               force of b on c is negative of c on b;
            */
            _XF(b) += xf;
            _YF(b) += yf;
            _XF(c) -= xf;
            _YF(c) -= yf;

            count++;
            totalNumberOfForcesComputed++;
        }
    }
}




void
calculateAssignedForces() {
    int b, c;
    int count = 0;


    /* Incrementally accumulate forces from each body pair,
       skipping force of body on itself (c == b)
    */
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

    for (b = displs2[myid]; b < displs2[myid] + bodies_per_proc[myid]; ++b) {
        //for (b = displs[myid]; b < displs[myid] + bodies_per_proc[myid]; ++b) {
        double xv = _XV(b);
        double yv = _YV(b);
        double force = sqrt(xv * xv + yv * yv) * FRICTION;
        double angle = atan2(yv, xv);
        double xf = _XF(b) - (force * cos(angle));
        double yf = _YF(b) - (force * sin(angle));

        _XV(b) += (xf / _M(b)) * DELTA_T;
        _YV(b) += (yf / _M(b)) * DELTA_T;
    }
}

void
compute_positions(void) {
    int b;
    for (b = displs2[myid]; b < displs2[myid] + bodies_per_proc[myid]; ++b) {
        //for (b = displs[myid]; b < displs[myid] + bodies_per_proc[myid]; ++b) {
        double xn = _X(b) + (_XV(b) * DELTA_T);
        double yn = _Y(b) + (_YV(b) * DELTA_T);

        /* Bounce of image "walls" */
        if (xn < 0) {
            xn = 0;
            _XV(b) = -_XV(b);
        } else if (xn >= xdim) {
            xn = xdim - 1;
            _XV(b) = -_XV(b);
        }
        if (yn < 0) {
            yn = 0;
            _YV(b) = -_YV(b);
        } else if (yn >= ydim) {
            yn = ydim - 1;
            _YV(b) = -_YV(b);
        }

        /* Update position */
        _XN(b) = xn;
        _YN(b) = yn;
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
        printf("%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n", _X(b), _Y(b), _XF(b), _YF(b), _XV(b), _YV(b));
    }
}

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
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    /* MPI_Init(&argc, &argv);
     MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
     MPI_Comm_rank(MPI_COMM_WORLD, &myid);
     MPI_Get_processor_name(processor_name, &namelen);

     fprintf(stderr, "Process %d on %s\n", myid, processor_name);*/

    if (argc != 5) {
        fprintf(stderr,
                "Usage: %s num_bodies secs_per_update ppm_output_file steps\n",
                argv[0]);
        exit(1);
    }
    /*fprintf(stderr, "0 => %s\n", argv[0]);
    fprintf(stderr, "1 => %s\n", argv[1]);
    fprintf(stderr, "2 => %s\n", argv[2]);
    fprintf(stderr, "3 => %s\n", argv[3]);
    fprintf(stderr, "4 => %s\n", argv[4]);*/
    if ((bodyCt = atol(argv[1])) > MAXBODIES ) {
        fprintf(stderr, "Using only %d bodies...\n", MAXBODIES);
        bodyCt = MAXBODIES;
    } else if (bodyCt < 2) {
        fprintf(stderr, "Using two bodies...\n");
        bodyCt = 2;
    }

    forces = malloc(sizeof(forceType) * bodyCt);
    for(i = 0; i < bodyCt; i++) {
        forces[i].xf = 0;
        forces[i].yf = 0;
    }
    /*if(bodyCt > numprocs) {
        bodyCt = numprocs;
    }*/
    new_bodies = malloc(sizeof(bodyType) * bodyCt);
    new_positions = malloc(sizeof(bodyPositionType) * bodyCt);
    bodies_per_proc = malloc(sizeof(int) * numprocs);

    secsup = atoi(argv[2]);
    image = map_P6(argv[3], &xdim, &ydim);
    steps = atoi(argv[4]);

    fprintf(stderr, "Running N-body with %i bodies and %i steps\n", bodyCt, steps);

    /* Initialize simulation data */
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
    //fprintf(stderr, "a\n");




    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);

    fprintf(stderr, "Process %d on %s\n", myid, processor_name);



    int forceCt = 0;
    for(i = 0; i < bodyCt; i++) {
        forceCt += i;
    }
    forces_per_proc = malloc(sizeof(int) * numprocs);
    displs = malloc(sizeof(int) * numprocs);
    displs2 = malloc(sizeof(int) * numprocs);
    int rem = forceCt % numprocs; // elements remaining after division among processes
    //fprintf(stderr, "rem => %d\n", rem);
    fprintf(stderr, "Total Forces => %d\n", forceCt);

    //fprintf(stderr, "b\n");
    const int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype mpi_force_type;
    MPI_Aint     offsets[2];

    offsets[0] = offsetof(forceType, xf);
    offsets[1] = offsetof(forceType, yf);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_force_type);
    MPI_Type_commit(&mpi_force_type);


    const int nitems2 = 2;
    int blocklengths2[2] = {2, 2};
    MPI_Datatype types2[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype mpi_position_type;
    MPI_Aint     offsets2[2];

    offsets2[0] = offsetof(bodyPositionType, x);
    offsets2[1] = offsetof(bodyPositionType, y);

    MPI_Type_create_struct(nitems2, blocklengths2, offsets2, types2, &mpi_position_type);
    MPI_Type_commit(&mpi_position_type);

    const int nitems3 = 4;
    int blocklengths3[4] = {1, 1, 1, 1};
    MPI_Datatype types3[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,};
    MPI_Datatype mpi_body_type;
    MPI_Aint     offsets3[4];

    offsets3[0] = offsetof(bodyType, xv);
    offsets3[1] = offsetof(bodyType, yv);
    offsets3[2] = offsetof(bodyType, mass);
    offsets3[3] = offsetof(bodyType, radius);

    MPI_Type_create_struct(nitems3, blocklengths3, offsets3, types3, &mpi_body_type);
    MPI_Type_commit(&mpi_body_type);




    //fprintf(stderr, "c\n");

    MPI_Op_create((MPI_User_function *) sumForces, 1, &mpi_sum);

    int avarage_forces_per_proc = forceCt / numprocs;
    //fprintf(stderr, "avarage => %d\n", avarage_bodies_per_proc);
    int sum = 0;

    //fprintf(stderr, "d\n");
    // calculate send counts and displacements
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

    int avarage_bodies_per_proc = bodyCt / numprocs;
    //fprintf(stderr, "avarage => %d\n", avarage_bodies_per_proc);
    sum = 0;

    //fprintf(stderr, "d\n");
    // calculate send counts and displacements
    for (i = 0; i < numprocs; i++) {
        bodies_per_proc[i] = avarage_bodies_per_proc;
        if (rem > 0) {
            bodies_per_proc[i]++;
            rem--;
        }
        displs2[i] = sum;
        sum += bodies_per_proc[i];
    }

    int bufSize = bodyCt % numprocs == 0 ? bodyCt / numprocs : (bodyCt / numprocs + 1);

    rec_positions = malloc(sizeof(bodyPositionType) * bufSize);
    rec_bodies = malloc(sizeof(bodyType) * bufSize);



    calculateAssignedForces();

    fprintf(stderr, "B -> %d, C -> %d \n", globalStartB, globalStartC);

    //MPI_Bcast(new_bodies, bodyCt, mpi_body_type, 0, MPI_COMM_WORLD);
    MPI_Scatterv(bodies, bodies_per_proc, displs2, mpi_body_type, rec_bodies, bufSize, mpi_body_type, 0, MPI_COMM_WORLD);
    

    MPI_Scatterv(positions, bodies_per_proc, displs2, mpi_position_type, rec_positions, bufSize, mpi_position_type, 0, MPI_COMM_WORLD);

    int cont;

    new_forces = malloc(sizeof(forceType) * bodyCt);
    for(i = 0; i < bodyCt; i++) {
        new_forces[i].xf = 0;
        new_forces[i].yf = 0;
    }

    new_bodies = malloc(sizeof(bodyType) * bodyCt);
    new_positions = malloc(sizeof(bodyPositionType) * bodyCt);

    MPI_Allgatherv(rec_bodies, bodies_per_proc[myid], mpi_body_type, new_bodies, bodies_per_proc, displs2, mpi_body_type, MPI_COMM_WORLD);
    
    MPI_Allgatherv(rec_positions, bodies_per_proc[myid], mpi_position_type, new_positions, bodies_per_proc, displs2, mpi_position_type, MPI_COMM_WORLD);


    if(gettimeofday(&start, 0) != 0) {
        fprintf(stderr, "could not do timing\n");
        exit(1);
    }




    /*fprintf(stderr, "\n");

    fprintf(stderr, "InitialForces2 -> ");
    for (i = 0; i < bodyCt; i++) {
        fprintf(stderr, "[%10.3f,%10.3f] ", _XF(i), _YF(i));
    }
    fprintf(stderr, "\n");*/

    while (steps--) {
        cont = 0;
        clear_forces();

        compute_forces();

        /*fprintf(stderr, "CalculatedForces -> ");
        for (i = 0; i < bodyCt; i++) {
            fprintf(stderr, "[%10.3f,%10.3f] ", _XF(i), _YF(i));
        }
        fprintf(stderr, "\n");*/

        new_forces2 = malloc(sizeof(forceType) * bodyCt);
        MPI_Allreduce(new_forces, new_forces2, bodyCt, mpi_force_type, mpi_sum, MPI_COMM_WORLD);
        free(new_forces);
        new_forces = new_forces2;

        /*fprintf(stderr, "TOTALForces -> ");
        for (i = 0; i < bodyCt; i++) {
            fprintf(stderr, "[%10.3f,%10.3f] ", _XF(i), _YF(i));
        }
        fprintf(stderr, "\n");*/

        compute_velocities();
        compute_positions();
        rec_positions = new_positions + displs2[myid];
        new_positions = malloc(sizeof(bodyPositionType) * bodyCt);
        //MPI_Allgatherv(rec_bodies, bodies_per_proc[myid], mpi_body_type, new_bodies, bodies_per_proc, displs, mpi_body_type, MPI_COMM_WORLD);
        MPI_Allgatherv(rec_positions, bodies_per_proc[myid], mpi_position_type, new_positions, bodies_per_proc, displs2, mpi_position_type, MPI_COMM_WORLD);


        old ^= 1;
        rec_positions = new_positions + displs2[myid];

    }

    
    if(0 == myid) {
        if(gettimeofday(&end, 0) != 0) {
            fprintf(stderr, "could not do timing\n");
            exit(1);
        }
        rtime = (end.tv_sec + (end.tv_usec / 1000000.0)) -
                (start.tv_sec + (start.tv_usec / 1000000.0));


    }

    fprintf(stderr, "FINEEE 1\n");
    new_positions = malloc(sizeof(bodyPositionType) * bodyCt);
    fprintf(stderr, "FINEEE 2\n");
    MPI_Gatherv(rec_positions, bodies_per_proc[myid], mpi_position_type, new_positions, bodies_per_proc, displs2, mpi_position_type, 0, MPI_COMM_WORLD);
    fprintf(stderr, "FINEEE 3\n");
    rec_bodies = new_bodies + displs2[myid];
    fprintf(stderr, "FINEEE 4\n");
    new_bodies = malloc(sizeof(bodyType) * bodyCt);
    fprintf(stderr, "FINEEE 5\n");
    MPI_Gatherv(rec_bodies, bodies_per_proc[myid], mpi_body_type, new_bodies, bodies_per_proc, displs2, mpi_body_type, 0, MPI_COMM_WORLD);
    fprintf(stderr, "FINEEE 6\n");

    if(0 == myid) {
        print();
        fprintf(stderr, "fine\n");
        fprintf(stderr, "N-body took %10.3f seconds\n", rtime);
    }

    fprintf(stderr, "Process %d compute %d forces\n, assigned %d bodies", myid, totalNumberOfForcesComputed, forces_per_proc[myid]);





    MPI_Finalize();


    free(forces_per_proc);
    free(displs);
    free(displs2);
    free(new_bodies);
    free(rec_bodies);


    return 0;
}
