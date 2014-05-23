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

#define NET_SIZE	  2 	// number of soms in the network
#define MAX_EPOCHS	100	// number of epochs
#define INPUT_SIZE	2	// input vector size
#define NUM_IN_VEC	100	// number of input vectors
#define SENSOR_DATA	"robot_data_jras_paper"

int main()
{
	/* create the SOM nets of the net */
	som** som_array = (som**)calloc(NET_SIZE, sizeof(som*));
	som* som1 = cln_create_som(1, 2, 2, 2, 12);
	som* som2 = cln_create_som(2, 2, 2, 8, 48);
	som_array[1] = som1;
	som_array[2] = som2;
	/* prepare simulation params */
	simopts* simulation = cln_setup_simulation(NET_SIZE, som_array, FIXED_PARAMS,0.1f, som1->size/2+1, 0.1f, 0.01f, 0.35f,(short)SENSOR_DATA, MAX_EPOCHS);
	/* build the input dataset */
	indataset* ind = cln_create_input_dataset((short)SENSOR_DATA, INPUT_SIZE, NUM_IN_VEC, SENSOR_DATA);
	/* loop the network */

	/* get post simulation data */
	simopts* sim_par_final = cln_get_simulation_params(simulation);
	
	/* build the output dataset */
	outdataset* outd = cln_create_output_dataset(sim_par_final, ind, som_array);
	/* dump data file */
	FILE *runtime = cln_dump_output_dataset(outd);
	
	/* free up resources */
	cln_destroy_som(som1);
	cln_destroy_som(som2);

	return EXIT_SUCCESS;
}
