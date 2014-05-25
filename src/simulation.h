/* 
   Unsupervised correlation learning network using self-organizing maps

        Each sensor projects onto a SOM network which will encode 
        the sensory afferent into neural activity. Depending on the 
        global network connectivity (1, 2, 3, ..., N sensory vars)
        each SOM connects to other SOMs associated with other sensory
        variables.

        Simple scenario with 2 variables, representing sensory data. 
        
        Simulation options and parameters for runtime.
	Definitions.
*/

#include "som.h"

/* simulation params */
typedef struct{
	short nsom;		// number of som in the net
	som** somnets;		// array of soms
	short paramsupdate;	// som dynamics params update type
	double* alpha;		// sensory projections learning rate
	double* sigma;		// neighborhood size
	double* gamma;		// cross-modal impact factor
	double* xi;		// inhibitory component factor
	double* kappa;		// cross-modal Hebbian learning rate
	double lambda;		// temporal coef for learning rates
	short datasrc;		// data source
	int simepochs;		// simulation epochs	
}simopts;

/* init simulation params */
simopts* cln_setup_simulation(short, som**, short, double, double, double, double, double, short, int);
/* return simulation parameters after runtime */
simopts* cln_get_simulation_params(simopts*);
/* set the current parameters in the simulation struct */
void cln_set_simulation_params(simopts*, int, double, double, double, double, double);

