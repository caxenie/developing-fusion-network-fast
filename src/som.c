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

#include "som.h"

/* build a SOM network given input params */
som* cln_create_som(short ni, short nsz, short insz, double inmin, double inmax)
{
	/* init */	
	printf("cln_create_som: Creating SOM%d with %dx%d neurons ...\n", ni, nsz, nsz);
	som* network = (som*)calloc(1, sizeof(som));
	network->id = ni;
	network->size = nsz;
	network->insize = insz;
	network->neurons = (neuron**)calloc(network->size, sizeof(neuron*));
	for(int idx = 0; idx < network->size; idx++){
		network->neurons[idx] = (neuron*)calloc(network->size, sizeof(neuron));
	}	
	for(int idx = 0; idx < network->size; idx++){
		for(int jdx = 0; jdx < network->size; jdx++){
			network->neurons[idx][jdx].xpos = idx;
			network->neurons[idx][jdx].ypos = jdx;
			network->neurons[idx][jdx].As = 0.0f;
			network->neurons[idx][jdx].Ax = 0.0f;
			network->neurons[idx][jdx].At = 0.0f;
			network->neurons[idx][jdx].W = (double*)calloc(insz, sizeof(double));
			network->neurons[idx][jdx].H = (double**)calloc(network->size, sizeof(double*));
			for (int tdx = 0; tdx < network->size; tdx++){
				network->neurons[idx][jdx].H[tdx] = (double*)calloc(network->size, sizeof(double));
			}
		}
	}
	for(int idx = 0; idx < network->size; idx++){
		for(int jdx = 0; jdx< network->size; jdx++){
			for(int in_idx = 0; in_idx < network->insize; in_idx++){
				/* randomly init sensory weights between min and max of input data to speed up convergence */
				network->neurons[idx][jdx].W[in_idx] = inmin + ((double)rand()/(double)RAND_MAX)*(inmax - inmin);
			}
			for(int tdx = 0; tdx < network->size; tdx++){
				for(int vdx = 0; vdx < network->size; vdx++){
					/* randomly init cross-modal weights with small numbers */
					network->neurons[idx][jdx].H[tdx][vdx] = (double)rand()/(double)RAND_MAX;
				}
			}
		}
	}
	printf("cln_create_som: SOM%d was created and initialized.\n", network->id);
	return network;
}

/* destroy a SOM network */
void cln_destroy_som(som* som)
{
	/* deallocate resources */
	for(int idx = 0;idx < som->size; idx++){
		for(int jdx = 0; jdx < som->size; jdx++){
	 		free(som->neurons[idx][jdx].W);
			for(int tdx = 0; tdx < som->size; tdx++){
				free(som->neurons[idx][jdx].H[tdx]);
			}
			free(som->neurons[idx][jdx].H);
		}
	}
	printf("cln_destory_som: Freed SOM%d allocated resources.\n", som->id);
}

/* display the SOM network details */
void cln_display_som(som* som)
{	
	printf("cln_display_som: SOM%d structure \n", som->id);
	printf("\n SIZE: %d \t INSIZE: %d \n", som->size, som->insize);
	printf("\n SYNAPTIC WEIGHTS - SENSORY AFFERENTS \n \n");
	for(int idx = 0; idx < som->size; idx++){
		for(int jdx = 0; jdx<som->size; jdx++){
				for(int in_idx = 0; in_idx<som->insize; in_idx++){
					printf(" %lf ", som->neurons[idx][jdx].W[in_idx]);
				}
				printf("\t");
		}
		printf("\n \n");
	}
	printf("\n SYNAPTIC WEIGHTS - CROSS MODAL LINKS \n \n");
	for(int idx = 0; idx < som->size; idx++){
		for(int in_idx = 0; in_idx<som->size; in_idx++){
			for(int jdx = 0; jdx<som->size; jdx++){
				for(int in_jdx = 0; in_jdx < som->size; in_jdx++){
					printf(" %lf ", som->neurons[idx][jdx].H[in_idx][in_jdx]);
				}
				printf("\t");
			}
			printf("\n");
		}
		printf("\n");
	}	
	printf("\n SENOSORY EVOKED NEURAL ACTIVATION\n \n");
	for(int idx = 0; idx < som->size; idx++){
		for(int jdx = 0; jdx < som->size; jdx++){
				printf(" %lf ", som->neurons[idx][jdx].As);
		}
		printf("\n");
	}
	printf("\n CROSS SENSORY EVOKED NEURAL ACTIVATION\n \n");
	for(int idx = 0; idx < som->size; idx++){
		for(int jdx = 0; jdx < som->size; jdx++){
				printf(" %lf ", som->neurons[idx][jdx].Ax);
		}
		printf("\n");
	}	
	printf("\n TOTAL NEURAL ACTIVATION\n \n");
	for(int idx = 0; idx < som->size; idx++){
		for(int jdx = 0; jdx < som->size; jdx++){
				printf(" %lf ", som->neurons[idx][jdx].At);
		}
		printf("\n");
	}
	printf("\n");
}

