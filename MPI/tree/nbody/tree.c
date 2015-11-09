#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

extern double	sqrt(double);
extern double	atan2(double, double);

#define GRAVITY		1.1
#define FRICTION	0.01
#define MAXBODIES	1024
#define DELTA_T		(0.025/5000)
#define	BOUNCE		-0.9
#define	SEED		27102015

typedef struct {
    double x[2];		/* Old and new X-axis coordinates */
    double y[2];		/* Old and new Y-axis coordinates */
    double xf;			/* force along X-axis */
    double yf;			/* force along Y-axis */
    double xv;			/* velocity along X-axis */
    double yv;			/* velocity along Y-axis */
    double mass;		/* Mass of the body */
    double radius;		/* width (derived from mass) */
} bodyType;

typedef struct node {
    int value;
    double mass;
    double massX;
    double massY;
    int cont;
    struct node *childs[4];
    /*Node* childA=NULL;
    Node* childB=NULL;
    Node* childC=NULL;
    Node* childD=NULL;*/
    double minX;
    double maxX;
    double minY;
    double maxY;
} *Node;

typedef struct {
    Node root;
} QuadTree;

bodyType bodies[MAXBODIES];
int	bodyCt;
int	old = 0;	/* Flips between 0 and 1 */
QuadTree q;

#define	X(B)		bodies[B].x[old]
#define	XN(B)		bodies[B].x[old^1]
#define	Y(B)		bodies[B].y[old]
#define	YN(B)		bodies[B].y[old^1]
#define	XF(B)		bodies[B].xf
#define	YF(B)		bodies[B].yf
#define	XV(B)		bodies[B].xv
#define	YV(B)		bodies[B].yv
#define	R(B)		bodies[B].radius
#define	M(B)		bodies[B].mass

int		xdim = 0;
int		ydim = 0;

static void initializeNode(Node parent, struct node *childs[]) {
    int i;
    double sumX = (parent->minX + parent->maxX);
    double sumY = (parent->minY + parent->maxY);
    for(i = 0; i < 4; i++) {
        switch(i) {
        case 0: {
            childs[i]->minX = parent->minX;
            childs[i]->maxX = sumX / 2;
            childs[i]->minY = sumY / 2;
            childs[i]->maxY = parent->maxY;
            break;
        };
        case 1: {
            childs[i]->minX = parent->minX;
            childs[i]->maxX = sumX / 2;
            childs[i]->minY = parent->minY;
            childs[i]->maxY = sumY / 2;
            break;
        };
        case 2: {
            childs[i]->minX = sumX / 2;
            childs[i]->maxX = parent->maxX;
            childs[i]->minY = parent->minY;
            childs[i]->maxY = sumY / 2;
            break;
        };
        case 3: {
            childs[i]->minX = sumX / 2;
            childs[i]->maxX = parent->maxX;
            childs[i]->minY = sumY / 2;
            childs[i]->maxY = parent->maxY;
            break;
        };
        default: {
            printf("ciao");
            return;
        }
        }
    }
    return;
}

static int getParent(int body, struct node *childs[]) {
    if(X(body) < childs[0]->maxX && X(body) >= childs[0]->minX && Y(body) <= childs[0]->maxY && Y(body) >= childs[0]->minY) {
        return 0;
    }
    if(X(body) < childs[1]->maxX && X(body) >= childs[1]->minX && Y(body) < childs[1]->maxY && Y(body) >= childs[1]->minY) {
        return 1;
    }
    if(X(body) <= childs[2]->maxX && X(body) >= childs[2]->minX && Y(body) < childs[2]->maxY && Y(body) >= childs[2]->minY) {
        return 2;
    }
    if(X(body) <= childs[3]->maxX && X(body) >= childs[3]->minX && Y(body) <= childs[3]->maxY && Y(body) >= childs[3]->minY) {
        return 3;
    }
    printf("\nbody: %d -> dennis\n", body);
    return -1;
}

void printBounds(struct node *childs[]) {
    int i;
    for(i = 0; i < 4; i++) {
        printf("\nminX: %10.3f, maxX: %10.3f, minY: %10.3f, maxY: %10.3f", childs[i]->minX, childs[i]->maxX, childs[i]->minY, childs[i]->maxY);

    }
}

static void QuadInsert(int i, Node r) {
    if(r->cont == 0) {
        r->value = i;
        r->cont = 1;
        return;
    } else if(r->cont == 1) {
        int j;
        for(j = 0; j < 4; j++) {
            r->childs[j] = malloc(sizeof(struct node));
            r->childs[j]->cont = 0;
        }
        initializeNode(r, r->childs);
        printBounds(r->childs);
        int quadrant = getParent(r->value, r->childs);
        printf("\nr->value, position: (%10.3f, %10.3f) =>quadrant -> %d", X(r->value), Y(r->value), quadrant);
        QuadInsert(r->value, r->childs[quadrant]);
        quadrant = getParent(i, r->childs);
        printf("\nposition: (%10.3f, %10.3f) =>quadrant -> %d", X(i), Y(i), quadrant);
        QuadInsert(i, r->childs[quadrant]);
        r->cont++;
        r->value = 0;
        return;
    } else if(r->cont > 1) {
        printBounds(r->childs);
        int quadrant = getParent(i, r->childs);
        printf("\nposition: (%10.3f, %10.3f) =>quadrant -> %d", X(i), Y(i), quadrant);
        QuadInsert(i, r->childs[quadrant]);
        r->cont++;
        return;
    }
    printf("\n 8 \n");
}

static void QuadTreeBuild() {
    printf("1");
    Node r = malloc(sizeof(struct node));
    r->cont = 0;
    printf("2");
    r->minX = 0;
    r->maxX = xdim;
    r->minY = 0;
    r->maxY = ydim;
    printf("3");
    q.root = r;
    int i;
    printf("4");
    for(i = 0; i < bodyCt; i++) {
        printf("\ni-> %d", i);
        QuadInsert(i, r);
    }
}


void calculateMassCenter(Node n) {
	if(n->cont==0)
		return;
    if(n->cont == 1) {
        n->massX = X(n->value);
        n->massY = Y(n->value);
    } else {
        double numX = 0;
        double numY = 0;
        int i;
        for(i = 0; i < 4; i++) {
            calculateMassCenter(n->childs[i]);
        }
        for(i = 0; i < 4; i++) {
            if(n->childs[i]->cont == 1) {
                numX += (n->childs[i]->massX * M(n->childs[i]->value));
                numY += (n->childs[i]->massY * M(n->childs[i]->value));
            }
            if(n->childs[i]->cont>1) {
                numX += (n->childs[i]->massX * n->childs[i]->value);
                numY += (n->childs[i]->massY * n->childs[i]->value);
            }
        }
        n->massX = numX / n->value;
        n->massY = numY / n->value;
    }
}

double calculateMass(Node n) {
    int i;
    if(n->cont == 0)
        return 0;
    if(n->cont == 1) {
        return M(n->value);
    } else {
        printf("\n -------------initial value -> %10.3f ----------", n->value);
        for(i = 0; i < 4; i++) {
            double m = calculateMass(n->childs[i]);
            //printf("\nmass of child number %d: %10.3f",i,m);
            printf("MM: %10.3f", m);
            n->value += m;
        }
        printf("\nTotal mass: %10.3f", n->value);
        return n->value;
    }
}

void printTree(Node n, int level) {
    int i;
    if(n->cont == 0) {
        for(i = 0; i < level; i++)
            printf("\t");
        printf("[minX:%d, maxX:%d, minY:%d, maxY:%d] VALUE-> 0\n", (int)n->minX, (int)n->maxX, (int)n->minY, (int)n->maxY);
    }
    if(n->cont == 1) {
        for(i = 0; i < level; i++)
            printf("\t");
        printf("leaf: %d, pos: (%f,%f)\n", (int)M(n-> value), X(n->value), Y(n->value));
        return;
    }
    if(n->cont > 1) {
        for(i = 0; i < level; i++)
            printf("\t");
        printf("[minX:%d, maxX:%d, minY:%d, maxY:%d] - MC:(%d,%d) - VALUE-> %d\n", (int)n->minX, (int)n->maxX, (int)n->minY, (int)n->maxY, (int)n->massX,(int)n->massY,(int)n->value);
        for(i = 0; i < 4; i++) {
            printTree(n->childs[i], level + 1);
        }
    }

}






/*	Graphic output stuff...
*/

#include <fcntl.h>
#include <sys/mman.h>

int		fsize;
unsigned char	*map;
unsigned char	*image;


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
           mmap(0,		/* Put it anywhere */
                fsize,	/* Map the whole file */
                (PROT_READ | PROT_WRITE),	/* Read/write */
                MAP_SHARED,	/* Not just for me */
                fd,		/* The file */
                0));	/* Right from the start */
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

#define	Eat_Space \
	while ((*p == ' ') || \
	       (*p == '\t') || \
	       (*p == '\n') || \
	       (*p == '\r') || \
	       (*p == '#')) { \
		if (*p == '#') while (*(++p) != '\n') ; \
		++p; \
	}

    Eat_Space;		/* Eat white space and comments */

#define	Get_Number(n) \
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

    Get_Number(*xdim);	/* Get image width */

    Eat_Space;		/* Eat white space and comments */
    Get_Number(*ydim);	/* Get image width */

    Eat_Space;		/* Eat white space and comments */
    Get_Number(maxval);	/* Get image max value */

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

#undef	Eat_Space
#undef	Get_Number

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





int main(int argc, char **argv) {
    unsigned int lastup = 0;
    unsigned int secsup;
    int b;
    int steps;
    double rtime;
    struct timeval start;
    struct timeval end;
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

    secsup = atoi(argv[2]);
    image = map_P6(argv[3], &xdim, &ydim);
    steps = atoi(argv[4]);

    fprintf(stderr, "Running N-body with %i bodies and %i steps\n", bodyCt, steps);

    srand(SEED);
    for (b = 0; b < bodyCt; ++b) {
        X(b) = (rand() % xdim);
        Y(b) = (rand() % ydim);
        R(b) = ((b * b + 1.0) * sqrt(1.0 * ((xdim * xdim) + (ydim * ydim)))) /
               (25.0 * (bodyCt * bodyCt + 1.0));
        M(b) = R(b) * R(b) * R(b);
        XV(b) = ((rand() % 20000) - 10000) / 2000.0;
        YV(b) = ((rand() % 20000) - 10000) / 2000.0;
    }

    QuadTreeBuild();
    double m = calculateMass(q.root);
    printf("\nspace mass: %10.3f", m);
    int i;
    int m2 = 0;
    for(i = 0; i < bodyCt; i++) {
        printf("\nbody: %d, mass: %d, pos: (%d,%d)", i, (int)M(i), (int)X(i), (int)Y(i));
        m2 += M(i);
    }
    if(m == m2) {
        printf("OK");
    } else {
        printf("PETTINO");
    }


    printf("\nmass_first_node: %d", (int)q.root->value);
    printf("\nq.root->%d", q.root);
    printf("\nmass_first_node: %d", (int)q.root->value);

    printf("\nmass: %10.3f", m);
    printf("\nmass_calc: %10.3f", m2);
    printf("\nmass_first_node: %d", (int)q.root->value);
    for(i = 0; i < 4; i++) {
        printf("\n-%d", (int)(q.root->childs[i])->value);
    }
    printf("\n");
    printf("\nxDim: %d, yDim: %d\n", xdim, ydim);
    calculateMassCenter(q.root);
    printTree(q.root, 0);
    

}