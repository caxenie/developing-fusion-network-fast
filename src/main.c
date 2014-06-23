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
#define NET_SOM_SIZEX	10			// soms size on X axis
#define NET_SOM_SIZEY	10			// soms size on Y axis

/* dynamics params */
#define ALPHA0		0.1f 					// learning rate init
#define SIGMA0		MAX(NET_SOM_SIZEX,NET_SOM_SIZEY)/2  	// neightborhood radius init
#define GAMMA0		0.1f					// cross-modal influence factor
#define XI0		0.01f					// inhibitory component for co-activation init
#define KAPPA0		0.3f					// Hebbian learning rate init
#define TAU		500					// time constant for learning adaptation
#define LAMDA		MAX_EPOCHS/log(SIGMA0)  		// time constant for radius adaptation
#define XMOD_LEARNING	HEBBIAN

/* entry point */
int main(int argc, char** argv)
{
	/* local sim params */
	int net_iter = 0;
	/* debug ASCII encoded files - remove in production code */	
	char* debug_som1 = NULL;
	char* debug_som2 = NULL;
	/* create the SOM nets of the net */
	som* som1 = cln_create_som(1, NET_SOM_SIZEX, NET_SOM_SIZEY, INPUT_SIZE, 2, 12);
	som* som2 = cln_create_som(2, NET_SOM_SIZEX, NET_SOM_SIZEY, INPUT_SIZE, 8, 48);
	/* prepare simulation params */
	simopts* simulation = cln_setup_simulation(ADAPTIVE_PARAMS, 
						   ALPHA0, SIGMA0, GAMMA0, XI0, KAPPA0, 
						   DATA_SOURCE, 
						   MAX_EPOCHS, 
						   XMOD_LEARNING);
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
			/* present a vector from the dataset to the network - plastic afferent sensory projections weights */
			for(int data_iter = 0; data_iter<NUM_IN_VEC; data_iter++){	
				/* get the current value of the simulation params */
				som1->params = cln_get_simulation_params(simulation);
				som2->params = cln_get_simulation_params(simulation);
				/* present data to som nets, find bmus and compute sensory elicited activity */
				cln_compute_sensory_activation(som1, cln_find_sensory_bmu(som1, ind1->data[data_iter]));
   			        cln_compute_sensory_activation(som2, cln_find_sensory_bmu(som2, ind2->data[data_iter]));
   			        /* compute the cross modal activation by cross propagating the sensory elicited activity */
				cln_compute_xmodal_activation(som1, cln_find_xmodal_bmu(som2, som1));
				cln_compute_xmodal_activation(som2, cln_find_xmodal_bmu(som1, som2));	
				/* compute the total activation in each som */
				cln_compute_joint_activation(som1);
				cln_compute_joint_activation(som2);
				/* update the sensory projection weights */
			        cln_compute_sensory_weights(som1, ind1->data[data_iter]);		
				cln_compute_sensory_weights(som2, ind2->data[data_iter]);
				/* update the cross-modal hebbian links */
				cln_compute_xmodal_weights(som1, som2);
				cln_compute_xmodal_weights(som2, som1);
		}				
		}
		else{
			printf("cln_main: Finalized training phase.\n");
			break;
		}
	}
	/* -------------------------------------------------------------------------------------------------*/
	/* get post simulation data */
	simopts* sim_par_final = cln_get_simulation_params(simulation);	
	/* display som data */
	cln_display_som(som1);
	cln_display_som(som2);
	/* build the output datasets */
	outdataset* outd1 = cln_create_output_dataset(sim_par_final, ind1, som1);
	outdataset* outd2 = cln_create_output_dataset(sim_par_final, ind2, som2);
	/* dump data file */
	debug_som1 = cln_dump_output_dataset(outd1);
	debug_som2 = cln_dump_output_dataset(outd2);
	/* dump the debug data file - ASCII encoded data */
	cln_read_output_dataset(debug_som1);
	cln_read_output_dataset(debug_som2);
	/* free up resources */
	cln_destroy_som(som1);
	cln_destroy_som(som2);

	return EXIT_SUCCESS;
}
