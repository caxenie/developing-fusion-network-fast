/* 
   Unsupervised correlation learning network using self-organizing maps

        Each sensor projects onto a SOM network which will encode 
        the sensory afferent into neural activity. Depending on the 
        global network connectivity (1, 2, 3, ..., N sensory vars)
        each SOM connects to other SOMs associated with other sensory
        variables.

        Simple scenario with 2 variables, representing sensory data. 
        
        Data input readout and preprocessing utils.
	Definitions.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "simulation.h"

/* input dataset representation */
typedef struct{
	int size;	 // size of a training vector
	int len;	 // number of training vectors
	double** data;	 // actual data num_vectors x len_vector
}indataset;

/* output dataset representation */
typedef struct{
	simopts* sopts;		// simulation params
	indataset* idata;	// input data
	som* somnet;		// som network
}outdataset;

/* read the data from input file or generate it depending on params */
indataset* cln_create_input_dataset(short, short, int, int, char*);
/* create the output dataset struct */
outdataset* cln_create_output_dataset(simopts*, indataset*, som*);
/* dump output dataset to file */
int cln_dump_output_dataset(outdataset*);

