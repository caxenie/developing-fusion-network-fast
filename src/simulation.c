/* 
   Unsupervised correlation learning network using self-organizing maps

        Each sensor projects onto a SOM network which will encode 
        the sensory afferent into neural activity. Depending on the 
        global network connectivity (1, 2, 3, ..., N sensory vars)
        each SOM connects to other SOMs associated with other sensory
        variables.

        Simple scenario with 2 variables, representing sensory data. 
        
        Simulation options and parameters for runtime.
*/

#include "simulation.h"

/* init simulation params */
simopts* cln_setup_simulation(short nnets, som** nets, short ut, double ai, double si, double gi, double xii, double ki, short src, int epochs)
{
	simopts* so = (simopts*)calloc(1, sizeof(simopts));
	so->nsom = nnets;
	so->somnets = nets;
	so->paramsupdate = ut;
	so->simepochs = epochs;
	so->datasrc = src;	
	so->lambda = epochs/log(si);
	so->alpha = (double**)calloc(epochs, sizeof(double*));
	so->sigma = (double**)calloc(epochs, sizeof(double*));
	so->gamma = (double**)calloc(epochs, sizeof(double*));
	so->xi = (double**)calloc(epochs, sizeof(double*));
	so->kappa = (double**)calloc(epochs, sizeof(double*));
	for(int idx = 0; idx< so->simepochs; idx++){
		so->alpha[idx] = (double*)calloc(((so->somnets[1])->insize)*epochs, sizeof(double));
		so->sigma[idx] = (double*)calloc(((so->somnets[1])->insize)*epochs, sizeof(double));
		so->gamma[idx] = (double*)calloc(((so->somnets[1])->insize)*epochs, sizeof(double));
		so->xi[idx] = (double*)calloc(((so->somnets[1])->insize)*epochs, sizeof(double));
		so->kappa[idx] = (double*)calloc(((so->somnets[1])->insize)*epochs, sizeof(double));
	}
	for(int idx = 0; idx< so->simepochs; idx++){
		for(int jdx = 0; jdx < ((so->somnets[1])->insize)*epochs; jdx++){
			so->alpha[idx][jdx] = ai, so->sigma[idx][jdx] = si, so->gamma[idx][jdx] = gi, so->xi[idx][jdx] = xii, so->kappa[idx][jdx] = ki;
		}
	}
	return so;
}

/* return simulation parameters after runtime */
simopts* cln_get_simulation_params(simopts* in)
{
	simopts* simout = (simopts*)calloc(1, sizeof(simopts));
	simout = in;
	return simout;	
}


