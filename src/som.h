/* 
   Unsupervised correlation learning network using self-organizing maps

	Each sensor projects onto a SOM network which will encode 
	the sensory afferent into neural activity. Depending on the 
	global network connectivity (1, 2, 3, ..., N sensory vars)
	each SOM connects to other SOMs associated with other sensory
	variables.

  	Simple scenario with 2 variables, representing sensory data. 
	
	SOM specific functionality.
	Definitions.
	
*/

#include <stdio.h>
#include <stdlib.h> 
#include <unistd.h>
#include <math.h>

/* data source */
enum{
        SENSOR_DATA = 0,
        ARTIFICIAL_DATA
};

/* learning and cross modal params */
enum{
        FIXED_PARAMS = 0,
        ADAPTIVE_PARAMS
};

/* correlation type */
enum{
        ALGEBRAIC = 0,
        TEMPORAL,
        NONLINEAR,
        DELAY,
        ALGTEMP
};

/* SOM neuron */
typedef struct{
	short xpos;	// x position in the SOM lattice 
	short ypos;	// y position in the SOM lattice
	double* W;	// sensory projections synaptic weights
	double** H;	// cross modal synaptic weights
	double As;	// sensory elicitied activity
	double Ax;	// cross modal elicited activity
	double At;	// total activity
}neuron;

/* SOM network */
typedef struct{
	/* structure */
	short id;	// id of the network
	short size; 	// size of the network
	short insize;	// input vector size
	neuron** neurons;	// the neurons lattice
}som;

/* build a SOM network given input params */
som* cln_create_som(short, short, short, double, double);
/* destroy a SOM network */
void cln_destroy_som(som*);
/* display a SOM network details */
void cln_display_som(som*);

