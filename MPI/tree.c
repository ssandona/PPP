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

typedef struct {
    int value;
    double mass;
    double massX;
    double massY;
    int cont;
    Node *childs[4];
    /*Node* childA=NULL;
    Node* childB=NULL;
    Node* childC=NULL;
    Node* childD=NULL;*/
    double minX;
    double maxX;
    double minY;
    double maxY;
} Node;

typedef struct {
    Node *root;
} QuadTree;

bodyType bodies[MAXBODIES];
int	bodyCt;
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

static void QuadTreeBuild() {
    Node *r = malloc(sizeof(Node));
    r.cont=0;
    r.minX = 0;
    r.maxX = xdim;
    r.minY = 0;
    r.maxY = ydim;
    q.root = r;
    int i;
    for(i = 0; i < bodyCt; i++)
        QuadInsert(i, r);
}

static void initializeNode(Node *parent, Node *childs) {
    int i;
    double sumX = (parent.minX + parent.maxX);
    double sumY = (parent.minY + parent.maxY);
    for(i = 0; i < 4; i++) {
        switch(i) {
        case 0: {
            child.minX = parent.minY;
            child.maxX = sumX / 2;
            child.minY = sumY / 2;
            child.maxY = parent.maxY;
            break;
        };
        case 1: {
            child.minX = parent.minX;
            child.maxX = sumX / 2;
            child.minY = parent.minY;
            child.maxY = sumY / 2;
            break;
        };
        case 2: {
            child.minX = sumX / 2;
            child.maxX = parent.maxX;
            child.minY = parent.minY;
            child.maxY = sumY / 2;
            break;
        };
        case 3: {
            child.minX = sumX / 2;
            child.maxX = parent.maxX;
            child.minY = sumY / 2;
            child.maxY = parent.maxY;
            break;
        };
        default:
            return 0;
        }
    }
    return 1;
}

static int getParent(int body, Node *childs) {
    int i;
    if(X(body) < childs[0].maxX && X(body) >= childs[0].minX && Y(body) <= childs[0].maxY && Y(body) >= childs[0].minY){
    	return 0;
    }
    if(X(body) < childs[1].maxX && X(body) >= childs[1].minX && Y(body) < childs[1].maxY && Y(body) >= childs[1].minY){
    	return 1;
    }
    if(X(body) <= childs[2].maxX && X(body) >= childs[2].minX && Y(body) < childs[2].maxY && Y(body) >= childs[2].minY){
    	return 2;
    }
    if(X(body) <= childs[3].maxX && X(body) >= childs[3].minX && Y(body) <= childs[3].maxY && Y(body) >= childs[3].minY){
    	return 3;
    }
    return -1;
}

static void QuadInsert(int i, Node *r) {
	int quadrant;
    if(r.cont == 0) {
        r.value = i;
        r.cont = 1;
    } else if(r.cont == 1) {
    	int j;
        for(j = 0; j< 4; j++){
            r.childs[j] = malloc(sizeof(Node));
            r.childs[j].cont=0;
        }
        initializeNode(r, r.childs);
        int quadrant=getParent(r.value,r.childs);
        QuadInsert(r.value,quadrant);
        quadrant=getParent(i,r.childs);
        QuadInsert(r.value, quadrant);
        r.cont++;
    }
    else if(r.cont>1){
    	int quadrant=getParent(i,r.childs);
        QuadInsert(r.value, quadrant);
        r.cont++;
    }
}

int main() {
    int b;
    int steps;

    /* Get Parameters */
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
    /* Initialize simulation data */
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

}