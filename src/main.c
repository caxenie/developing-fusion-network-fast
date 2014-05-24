/*
	Unsupervised correlation learning network using self-organizing maps

        Each sensor projects onto a SOM network which will encode 
        the sensory afferent into neural activity. Depending on the 
        global network connectivity (1, 2, 3, ..., N sensory vars)
        each SOM connects to other SOMs associated with other sensory
        variables.

        Simple scenario with 2 variables, representing sensory data. 
        
        Correlation network entry point.
*/

#include "data.h"

/* simulation parms */
#define NET_SIZE	2 			// number of soms in the network
#define MAX_EPOCHS	200			// number of epochs
#define INPUT_SIZE	2			// input vector size
#define NUM_IN_VEC	10			// number of input vectors
#define SENSOR_DATASET	"robot_data_jras_paper" // sensory data file
#define DATA_SOURCE	ARTIFICIAL_DATA		// network input source
/* network params */
#define NET_SOM_SIZE	10			// net soms sizes

/* dynamics params */
#define ALPHA0		0.1f 			// learning rate init
#define SIGMA0		NET_SOM_SIZE/2  	// neightborhood radius init
#define GAMMA0		0.1f			// cross-modal influence factor
#define XI0		0.01f			// inhibitory component for co-activation init
#define KAPPA0		0.3f			// Hebbian learning rate init
#define TAU		500			// time constant for learning adaptation
#define LAMDA		MAX_EPOCHS/log(SIGMA0)  // time constant for radius adaptation

/* entry point */
int main(int argc, char** argv)
{
	/* local sim params */
	int net_iter = 0;
	
	/* create the SOM nets of the net */
	som** som_array = (som**)calloc(NET_SIZE, sizeof(som*));
	som* som1 = cln_create_som(1, NET_SOM_SIZE, INPUT_SIZE, 2, 12);
	som* som2 = cln_create_som(2, NET_SOM_SIZE, INPUT_SIZE, 8, 48);
	som_array[1] = som1;
	som_array[2] = som2;
	/* prepare simulation params */
	simopts* simulation = cln_setup_simulation(NET_SIZE, som_array, ADAPTIVE_PARAMS, ALPHA0, SIGMA0, GAMMA0, XI0, KAPPA0, DATA_SOURCE, MAX_EPOCHS);
	/* build the input datasets */
	indataset* ind1 = cln_create_input_dataset(som1->id, DATA_SOURCE, INPUT_SIZE, NUM_IN_VEC, SENSOR_DATASET);
	indataset* ind2 = cln_create_input_dataset(som2->id, DATA_SOURCE, INPUT_SIZE, NUM_IN_VEC, SENSOR_DATASET);
	/* -------------------------------------------------------------------------------------------------------------------------------------------*/
	/* loop the network */
	while(1){
		if(net_iter++<=MAX_EPOCHS){
			/* adaptation parameters check */
			if(simulation->paramsupdate==ADAPTIVE_PARAMS){
				/* compute the adaptation params */
			 	 cln_set_simulation_params(simulation, net_iter, 
							  ALPHA0*exp(-(double)net_iter/TAU),
							  SIGMA0*exp(-(double)net_iter/(double)LAMDA),
							  GAMMA0*exp((double)net_iter/TAU),
							  XI0*exp((double)net_iter/TAU),
							  KAPPA0*exp((double)net_iter/TAU));
			}	
			else{		 	 
				cln_set_simulation_params(simulation, net_iter, 
							  ALPHA0,
							  SIGMA0*exp(-(double)net_iter/(double)LAMDA),
							  GAMMA0,
							  XI0,
							  KAPPA0);
			}
		}
		else{
			printf("cln_main: Finalized training phase.\n");
			break;
		}
	}
	/* -------------------------------------------------------------------------------------------------------------------------------------------*/
	/* get post simulation data */
	simopts* sim_par_final = cln_get_simulation_params(simulation);
	/* build the output datasets */
	outdataset* outd1 = cln_create_output_dataset(sim_par_final, ind1, som_array[1]);
	outdataset* outd2 = cln_create_output_dataset(sim_par_final, ind2, som_array[2]);
	/* dump data file */
	cln_dump_output_dataset(outd1);
	cln_dump_output_dataset(outd2);

	/* free up resources */
	cln_destroy_som(som1);
	cln_destroy_som(som2);

	return EXIT_SUCCESS;
}
