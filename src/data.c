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
indataset* cln_create_input_dataset(short netid, short data_src, int vsize, int nv, char* data_file)
{
	/* init */
	FILE *fin, *fout; 
	int lcnt = 0, lidx = 1, c;
	printf("cln_create_input_dataset: Creating input dataset...\n");
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
			/*
			for(int idx = 0; idx < dset->len; idx++){
				for(int jdx=0; jdx<dset->size; jdx++){
					// sine with f=25Hz, Ts = 25ms and amplitude given by som id 
					dset->data[idx][jdx] = netid*sin(2*M_PI*50*(jdx/25));
				}
			}
			*/					
		break;
		case SENSOR_DATA:
      			printf("cln_create_input_dataset: Reading sensor data from file...\n");
			/* access the raw data file */  
			fin = fopen(data_file, "r");
		        while((c=fgetc(fin))!=EOF) if(c=='\n') lcnt++;
			printf("cln_create_input_dataset: Read %d sensory data samples.\n", lcnt);
	        	for(int idx = 0; idx<dset->len; idx++){
                		for(int jdx = 0; jdx < dset->size; jdx++){
		                        dset->data[idx][jdx] = 1;
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
char* cln_dump_output_dataset(outdataset* ods)
{	
	printf("cln_dump_output_dataset: Dumping output dataset to disk...\n");
	time_t rawt; time(&rawt);
	struct tm* tinfo = localtime(&rawt);
	FILE* fout; 
	char* nfout = (char*)calloc(400, sizeof(char));
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
		return NULL;
	}
	fwrite(ods, sizeof(outdataset), 1, fout);
	printf("cln_dump_output_dataset: Dumped to disk: %s\n", nfout);
	fclose(fout);
	return nfout;
}

/* read output dataset for debugging purposes */
int cln_read_output_dataset(char *dumped_file)
{	
	printf("cln_read_output_dataset: Reading the dumped output dataset - debug ...\n");
	char* debug_file = (char*)calloc(strlen(dumped_file)+9, sizeof(char));
	debug_file = strcat(debug_file, dumped_file); 
	debug_file = strcat(debug_file, "-DEBUG");
	FILE* fin = fopen(dumped_file, "rb");
	FILE* fout = fopen(debug_file, "w");
	printf("cln_read_output_dataset: Debug file is %s \n", debug_file);
	outdataset* od = (outdataset*)calloc(1, sizeof(outdataset));
	if(!fin){
		printf("cln_read_output_dataset: Cannot open runtime data file !\n");
		return -1;
	}
	if(!fout){
		printf("cln_read_output_dataset: Cannot open debug output data file !\n");
		return -1;
	}
	/* find the end of the file and rewind to first position */
	fseek(fin, sizeof(od), SEEK_END);
	rewind(fin);
	fread(od, sizeof(outdataset), 1, fin);
	/* display contents of the struct read from the output file 
		contents of the output dataset struct:
		        od->sopts  ... simulation options
		        od->idata  ... input data
		        od->somnet ... network data
	*/
	fprintf(fout, "---- NETWORK SIMULATION PARAMETERS ----\n");
	fprintf(fout, "Initial values for params\n");	
	fprintf(fout, "update_type: %d \n sim_epochs: %d \n data_src: %d \n lambda: %lf \n alpha: %lf \n sigma: %lf \n gamma: %lf \n xi: %lf \n kappa: %lf \n cur_epoch: %d \n learn_rule: %d \n", od->sopts->paramsupdate, od->sopts->simepochs, od->sopts->datasrc, od->sopts->lambda, od->sopts->alpha[0], od->sopts->sigma[0], od->sopts->gamma[0], od->sopts->xi[0], od->sopts->kappa[0], od->sopts->cur_epoch, od->sopts->learn_rule);
	fprintf(fout, "Development process values for params\n");
	fprintf(fout, "update_type \t sim_epochs \t data_src \t lambda \t\t alpha \t\t sigma \t\t gamma \t\t xi \t\t kappa \t\t cur_epoch \t\t learn_rule \n");
	for(int idx = 0; idx<od->sopts->simepochs; idx++){
		fprintf(fout, "%d \t\t %d \t\t %d \t\t %lf \t\t %lf \t\t %lf \t\t %lf \t\t %lf \t\t %lf \t\t %d \t\t %d \n", od->sopts->paramsupdate, od->sopts->simepochs, od->sopts->datasrc, od->sopts->lambda, od->sopts->alpha[idx], od->sopts->sigma[idx], od->sopts->gamma[idx], od->sopts->xi[idx], od->sopts->kappa[idx], od->sopts->cur_epoch, od->sopts->learn_rule);
	}
	/* close file and return succesfully */	
	fclose(fin);
	fclose(fout);
	return EXIT_SUCCESS;
}

