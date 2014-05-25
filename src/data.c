/* 
   Unsupervised correlation learning network using self-organizing maps

        Each sensor projects onto a SOM network which will encode 
        the sensory afferent into neural activity. Depending on the 
        global network connectivity (1, 2, 3, ..., N sensory vars)
        each SOM connects to other SOMs associated with other sensory
        variables.

        Simple scenario with 2 variables, representing sensory data. 
        
        Data input readout and preprocessing utils.
	Implementations.
*/

#include "data.h"

/* read the data from input file or generate it depending on params */
indataset* cln_create_input_dataset(short netid, short data_src, int vsize, int nv, char* data_file)
{
	/* init */
	FILE *fin, *fout; 
	int lcnt = 0, lidx = 1, c;
	printf("cln_create_input_dataset: Creating input dataset from %s data...\n",(data_src==SENSOR_DATA)?"sensory":"artificial");
        /* build the dataset */
	indataset* dset = (indataset*)calloc(1, sizeof(indataset));
	dset->size = vsize;
	dset->len = nv;
	dset->data = (double**)calloc(dset->len, sizeof(double*));
	for(int idx = 0; idx < dset->len; idx++)
        	dset->data[idx] = (double*)calloc(dset->size, sizeof(double));
	/* check data source */
	switch(data_src){
		case ARTIFICIAL_DATA:
			/* algebraic correlation */
			for(int idx = 0; idx < dset->len; idx++){
				for(int jdx=0; jdx<dset->size; jdx++){
					/* overlay some noise from [1,2] uniformly distributed */
					dset->data[idx][jdx] = 4.5f*netid + (1.0f + (2.0f-1.0f)*((double)rand()/(double)RAND_MAX));
				}
			}
			/* temporal and algebraic correlation */
			/* generate a sine wave */
			for(int idx = 0; idx < dset->len; idx++){
				for(int jdx=0; jdx<dset->size; jdx++){
					/* sine with f=25Hz, Ts = 25ms and amplitude given by som id */
					dset->data[idx][jdx] = netid*sin(2*M_PI*50*(jdx/25));
				}
			}					
		break;
		case SENSOR_DATA:
      			printf("cln_create_input_dataset: Reading sensor data from file...\n");
			/* access the raw data file */  
			fin = fopen(data_file, "r");
		        while((c=fgetc(fin))!=EOF) if(c=='\n') lcnt++;
			printf("cln_create_input_dataset: Read %d sensory data samples.\n", lcnt);
	        	for(int idx = 0; idx<dset->len; idx++){
                		for(int jdx = 0; jdx < dset->size; jdx++){
					// TODO add parsing routine and post-processing
		                        dset->data[idx][jdx] = 0.0;
                		}
        		}
			printf("cln_create_input_dataset: Closing sensory data file.\n");
			fclose(fin);		
		break;
	}
	printf("cln_create_input_dataset: Created %s data input dataset.\n", data_src==ARTIFICIAL_DATA ? "artificial" : "sensory");
	return dset;
}

/* create the output dataset struct */
outdataset* cln_create_output_dataset(simopts* so, indataset* ind, som* net)
{
	printf("cln_create_output_dataset: Creating output dataset for som %d ...\n", net->id);
	outdataset* outd = (outdataset*)calloc(1, sizeof(outdataset));
	outd->sopts = so;
	outd->idata = ind;
	outd->somnet = net;
	printf("cln_create_output_dataset: Created som %d output dataset.\n", net->id);	
	return outd;
}

/* dump the runtime data in a file for later processing */
int cln_dump_output_dataset(outdataset* ods)
{	
	printf("cln_dump_output_dataset: Dumping output dataset to disk...\n");
	time_t rawt; time(&rawt);
	struct tm* tinfo = localtime(&rawt);
	FILE* fout; char nfout[400];
	char simparams[200];	

	strftime(nfout, 100, "%Y-%m-%d__%H:%M:%S", tinfo);
	strcat(nfout, "_cln_runtime_data_som_");
	sprintf(simparams, "%d_%d_epochs_%s_srcdata_%s_params_adaptation", 
			   ods->somnet->id,
			   ods->sopts->simepochs, 
		           ods->sopts->datasrc==ARTIFICIAL_DATA ? "artificial" : "sensory", 
			   ods->sopts->paramsupdate==FIXED_PARAMS ? "fixed" : "adaptive");
	strcat(nfout, simparams);

	if((fout = fopen(nfout, "wb"))==NULL){
		printf("cln_dump_output_dataset: Cannot create output file.\n");
		return -1;
	}
	fwrite(ods, sizeof(outdataset), 1, fout);
			
	printf("cln_dump_output_dataset: Dumped to disk: %s\n", nfout);
	return EXIT_SUCCESS;
}
