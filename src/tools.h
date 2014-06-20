/* 
   Unsupervised correlation learning network using self-organizing maps

        Each sensor projects onto a SOM network which will encode 
        the sensory afferent into neural activity. Depending on the 
        global network connectivity (1, 2, 3, ..., N sensory vars)
        each SOM connects to other SOMs associated with other sensory
        variables.

        Simple scenario with 2 variables, representing sensory data. 
        
	Tools to use in simulating networks dynamics. Definition.      
*/

#include <math.h>

/* compute the norm of 2 vectors */
double cln_compute_norm(double* v1, double* v2, int sz);

