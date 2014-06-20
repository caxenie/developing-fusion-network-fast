/* 
   Unsupervised correlation learning network using self-organizing maps

	Each sensor projects onto a SOM network which will encode 
	the sensory afferent into neural activity. Depending on the 
	global network connectivity (1, 2, 3, ..., N sensory vars)
	each SOM connects to other SOMs associated with other sensory
	variables.

  	Simple scenario with 2 variables, representing sensory data. 
	
	SOM specific functionality
*/

#include <stdio.h>
#include <stdlib.h> 
#include <unistd.h>
#include <float.h>
#include "tools.h"

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

/* cross modal learning rule */
enum
{
	NONE = 0,
	HEBBIAN,
	COVARIANCE
};

/* simulation params */
typedef struct{
        short paramsupdate;     // som dynamics params update type
        double* alpha;          // sensory projections learning rate
        double* sigma;          // neighborhood size
        double* gamma;          // cross-modal impact factor
        double* xi;             // inhibitory component factor
        double* kappa;          // cross-modal Hebbian learning rate
        double lambda;          // temporal coef for learning rates
        short datasrc;          // data source
        int simepochs;          // simulation epochs    
        int cur_epoch;          // current training epoch 
	short learn_rule;	// type of learning rule for cross-modal interaction
}simopts;

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
	short xsize; 	// size of the network x dimension
	short ysize;	// size of the network y dimension
	short insize;	// input vector size
	neuron** neurons;	// the neurons lattice
	simopts* params;// parameters for simulation for som
}som;

/* build a SOM network given input params */
som* cln_create_som(short nid, short nszx, short nszy, short insz, double inmin, double inmax);
/* destroy a SOM network */
void cln_destroy_som(som* som);
/* display the SOM network details */
void cln_display_som(som* som);
/* find the sensory elicited winner neuron */
neuron* cln_find_sensory_bmu(som* som, double* ind);
/* find the cross-modal elicited winner neuron */
neuron* cln_find_xmodal_bmu(som* soms, som* somt);
/* compute forward activation - sensory afferents elicited activation */
void cln_compute_sensory_activation(som* s, neuron* bmu);
/* compute indirect activation - cross-modal elicited activation */
void cln_compute_xmodal_activation(som* s, neuron* bmu);
/* comute the joint activation as a weighted sum of direct and indirect activations */
void cln_compute_joint_activation(som* s);
/* adapt the sensory projecton weights */
void cln_compute_sensory_weights(som* s, double* in_vector);
/* adapt the cross-modal hebbian links */
void cln_compute_xmodal_weights(som* s, som* d);
