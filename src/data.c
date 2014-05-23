/* 
   Unsupervised correlation learning network using self-organizing maps

        Each sensor projects onto a SOM network which will encode 
        the sensory afferent into neural activity. Depending on the 
        global network connectivity (1, 2, 3, ..., N sensory vars)
        each SOM connects to other SOMs associated with other sensory
        variables.

        Simple scenario with 2 variables, representing sensory data. 
        
        Data input readout and preprocessing utils.
*/

#include "data.h"

/* read the data from input file or generate it depending on params */
indataset* cln_create_input_dataset(short data_src, int vsize, int nv, char* data_file)
{
	/* init */
	FILE *fin, *fout; 
	int lcnt = 0, lidx = 1, c;
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

		break;
		case SENSOR_DATA:
      			/* access the raw data file */  
			fin = fopen(data_file, "r");
		        while((c=fgetc(fin))!=EOF) if(c=='\n') lcnt++;
	        	for(int idx = 0; idx<dset->len; idx++){
                		for(int jdx = 0; jdx < dset->size; jdx++){
		                        dset->data[idx][jdx] =1;
                		}
        		}		
		break;
	}
	return dset;
}

/* create the output dataset struct */
outdataset* cln_create_output_dataset(simopts* so, indataset* ind, som** nets)
{
	outdataset* outd = (outdataset*)calloc(1, sizeof(outdataset));
	outd->sopts = so;
	outd->idata = ind;
	outd->somnets = nets;

	return outd;
}

/* dump the runtime data in a file for later processing */
FILE* cln_dump_output_dataset(outdataset* ods)
{	
	time_t rawt; time(&rawt);
	struct tm* tinfo = localtime(&rawt);
	FILE* fout; char nfout[400];
	char simparams[200];	

	strftime(nfout, 100, "%Y-%m-%d__%H:%M:%S", tinfo);
	strcat(nfout, "_cln_runtime_data_");
	sprintf(simparams, "%d_epochs_%d_srcdata_%d_params_adaptation", ods->sopts->simepochs, ods->sopts->datasrc, ods->sopts->paramsupdate);
	strcat(nfout, simparams);

	if((fout = fopen(nfout, "w+"))==NULL){
		printf("cln_dump_output_dataset: Cannot create output file.\n");
		return NULL;
	}
		
		
	return fout;
}
