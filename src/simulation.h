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

#include "som.h"

/* init simulation params */
simopts* cln_setup_simulation(short ut, double ai, double si, double gi, double xii, double ki, short src, int epochs, short learning_type);
/* return simulation parameters after runtime */
simopts* cln_get_simulation_params(simopts* in);
/* set the current parameters in the simulation struct */
void cln_set_simulation_params(simopts*so, int iter, double ai, double si, double gi, double xii, double ki);

