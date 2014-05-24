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
	printf("cln_setup_simulation: Initializing simulation params ...\n");
	simopts* so = (simopts*)calloc(1, sizeof(simopts));
	so->nsom = nnets;
	so->somnets = nets;
	so->paramsupdate = ut;
	so->simepochs = epochs;
	so->datasrc = src;	
	so->lambda = epochs/log(si);
	so->alpha = (double*)calloc(epochs, sizeof(double));
	so->sigma = (double*)calloc(epochs, sizeof(double));
	so->gamma = (double*)calloc(epochs, sizeof(double));
	so->xi = (double*)calloc(epochs, sizeof(double));
	so->kappa = (double*)calloc(epochs, sizeof(double));
	for(int idx = 0; idx< so->simepochs; idx++){
			so->alpha[idx] = ai, so->sigma[idx] = si, so->gamma[idx] = gi, so->xi[idx] = xii, so->kappa[idx] = ki;
	}
	for(int idx = 0; idx< so->simepochs; idx++){
                printf("epoch [%d] %lf %lf %lf %lf %lf\n", idx, so->alpha[idx], so->sigma[idx], so->gamma[idx], so->xi[idx], so->kappa[idx]);
        }

	printf("cln_setup_simulation: Simulation parameters are set.\n");
	return so;
}

/* return simulation parameters after runtime */
simopts* cln_get_simulation_params(simopts* in)
{
	printf("cln_get_simulation_params: Getting current value of simulation params...\n");
	simopts* simout = (simopts*)calloc(1, sizeof(simopts));
	simout = in;
	printf("cln_get_simulation_params: Simulation params after runtime are saved.\n");
	return simout;	
}

/* set the current parameters in the simulation struct */
void cln_set_simulation_params(simopts* so, int iter, double ai, double si, double gi, double xii, double ki)
{
	so->alpha[iter] = ai, so->sigma[iter] = si, so->gamma[iter] = gi, so->xi[iter] = xii, so->kappa[iter] = ki;
        printf("cln_set_params: epoch [%d]  %lf %lf %lf %lf %lf\n", iter, so->alpha[iter], so->sigma[iter], so->gamma[iter], so->xi[iter], so->kappa[iter]);
}

