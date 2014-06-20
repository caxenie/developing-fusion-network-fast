/* 
   Unsupervised correlation learning network using self-organizing maps

	Each sensor projects onto a SOM network which will encode 
	the sensory afferent into neural activity. Depending on the 
	global network connectivity (1, 2, 3, ..., N sensory vars)
	each SOM connects to other SOMs associated with other sensory
	variables.

  	Simple scenario with 2 variables, representing sensory data. 
	
	SOM specific functionality
*/

#include "som.h"

/* build a SOM network given input params */
som* cln_create_som(short ni, short nszx, short nszy, short insz, double inmin, double inmax)
{
	/* init */	
	printf("cln_create_som: Creating SOM%d with %dx%d neurons ...\n", ni, nszx, nszy);
	som* network = (som*)calloc(1, sizeof(som));
	network->id = ni;
	network->xsize = nszx;
	network->ysize = nszy;
	network->insize = insz;
	network->neurons = (neuron**)calloc(network->xsize, sizeof(neuron*));
	for(int idx = 0; idx < network->xsize; idx++){
		network->neurons[idx] = (neuron*)calloc(network->ysize, sizeof(neuron));
	}	
	for(int idx = 0; idx < network->xsize; idx++){
		for(int jdx = 0; jdx < network->ysize; jdx++){
			network->neurons[idx][jdx].xpos = idx;
			network->neurons[idx][jdx].ypos = jdx;
			network->neurons[idx][jdx].As = 0.0f;
			network->neurons[idx][jdx].Ax = 0.0f;
			network->neurons[idx][jdx].At = 0.0f;
			network->neurons[idx][jdx].W = (double*)calloc(insz, sizeof(double));
			network->neurons[idx][jdx].H = (double**)calloc(network->xsize, sizeof(double*));
			for (int tdx = 0; tdx < network->xsize; tdx++){
				network->neurons[idx][jdx].H[tdx] = (double*)calloc(network->ysize, sizeof(double));
			}
		}
	}
	for(int idx = 0; idx < network->xsize; idx++){
		for(int jdx = 0; jdx< network->ysize; jdx++){
			for(int in_idx = 0; in_idx < network->insize; in_idx++){
				/* randomly init sensory weights between min and max of input data to speed up convergence */
				network->neurons[idx][jdx].W[in_idx] = inmin + ((double)rand()/(double)RAND_MAX)*(inmax - inmin);
			}
			for(int tdx = 0; tdx < network->xsize; tdx++){
				for(int vdx = 0; vdx < network->ysize; vdx++){
					/* randomly init cross-modal weights with small numbers */
					network->neurons[idx][jdx].H[tdx][vdx] = (double)rand()/(double)RAND_MAX;
				}
			}
		}
	}
	printf("cln_create_som: SOM%d was created and initialized.\n", network->id);
	return network;
}

/* destroy a SOM network */
void cln_destroy_som(som* som)
{
	/* deallocate resources */
	for(int idx = 0;idx < som->xsize; idx++){
		for(int jdx = 0; jdx < som->ysize; jdx++){
	 		free(som->neurons[idx][jdx].W);
			for(int tdx = 0; tdx < som->ysize; tdx++){
				free(som->neurons[idx][jdx].H[tdx]);
			}
			free(som->neurons[idx][jdx].H);
		}
	}
	printf("cln_destory_som: Freed SOM%d allocated resources.\n", som->id);
}

/* display the SOM network details */
void cln_display_som(som* som)
{	
	printf("cln_display_som: SOM%d structure \n", som->id);
	printf("\n SIZE: %dx%d \t INSIZE: %d \n", som->xsize, som->ysize, som->insize);
	printf("\n SYNAPTIC WEIGHTS - SENSORY AFFERENTS \n \n");
	for(int idx = 0; idx < som->xsize; idx++){
		for(int jdx = 0; jdx<som->ysize; jdx++){
				for(int in_idx = 0; in_idx<som->insize; in_idx++){
					printf(" %lf ", som->neurons[idx][jdx].W[in_idx]);
				}
				printf("\t");
		}
		printf("\n \n");
	}
	printf("\n SYNAPTIC WEIGHTS - CROSS MODAL LINKS \n \n");
	for(int idx = 0; idx < som->ysize; idx++){
		for(int in_idx = 0; in_idx<som->xsize; in_idx++){
			for(int jdx = 0; jdx<som->ysize; jdx++){
				for(int in_jdx = 0; in_jdx < som->xsize; in_jdx++){
					printf(" %lf ", som->neurons[idx][jdx].H[in_idx][in_jdx]);
				}
				printf("\t");
			}
			printf("\n");
		}
		printf("\n");
	}	
	printf("\n SENOSORY EVOKED NEURAL ACTIVATION\n \n");
	for(int idx = 0; idx < som->xsize; idx++){
		for(int jdx = 0; jdx < som->ysize; jdx++){
				printf(" %lf ", som->neurons[idx][jdx].As);
		}
		printf("\n");
	}
	printf("\n CROSS SENSORY EVOKED NEURAL ACTIVATION\n \n");
	for(int idx = 0; idx < som->xsize; idx++){
		for(int jdx = 0; jdx < som->ysize; jdx++){
				printf(" %lf ", som->neurons[idx][jdx].Ax);
		}
		printf("\n");
	}	
	printf("\n TOTAL NEURAL ACTIVATION\n \n");
	for(int idx = 0; idx < som->xsize; idx++){
		for(int jdx = 0; jdx < som->ysize; jdx++){
				printf(" %lf ", som->neurons[idx][jdx].At);
		}
		printf("\n");
	}
	printf("\n");
}

/* find the sensory elicited winner neuron */
neuron* cln_find_sensory_bmu(som* som, double* vin)
{
	/* the winner neuron after projecting the sensory data */
	neuron* bmu = (neuron*)calloc(1, sizeof(neuron));
	/* max quantization error */
	double max_qe = DBL_MAX;
	double cur_qe = 0.0f;
	/* find the bmu in the som */
	for(int idx=0; idx<som->xsize; idx++){
		for (int jdx = 0;jdx<som->ysize;jdx++){
			/* the winner is the neuron which minimizes the Euclidian distance to the input */
			if((cur_qe = cln_compute_norm(vin, som->neurons[idx][jdx].W, som->insize))<max_qe){
				max_qe = cur_qe;
				bmu->xpos = idx;
				bmu->ypos = jdx;
			}
		}
	}
	return bmu;
}

/* find the cross-modal elicited winner neuron */
neuron* cln_find_xmodal_bmu(som* src_som, som* dest_som)
{
	/* cross modal projection winner neurnon */
	neuron* bmu = (neuron*)calloc(1,sizeof(neuron));
	/* global cross modal activation from source to target network */
	double** xmod_act = (double**) calloc(dest_som->xsize, sizeof(double*));
	for(int i=0; i<dest_som->xsize; i++)
		xmod_act[i] = (double*)calloc(dest_som->ysize, sizeof(double));
	/* maximally activated neuron activation */
	double max_xmod_act = 0.0f;
	/* compute activation from source network to target network and check winner (maximizes the activation) */
	for(int idx = 0; idx<dest_som->xsize; idx++){
		for(int jdx=0;jdx<dest_som->ysize; jdx++){
			for (int tdx = 0; tdx<src_som->xsize; tdx++){
				for(int sdx = 0; sdx<src_som->ysize; sdx++){
					xmod_act[idx][jdx]+=src_som->neurons[tdx][sdx].As*src_som->neurons[tdx][sdx].H[idx][jdx];
				}
			}
		}
	}
	/* find the neuron which is closer to the maximum cross activation */
	for (int ridx = 0;ridx<dest_som->xsize; ridx ++){
		for (int cidx = 0; cidx<dest_som->ysize; cidx++){
			if(xmod_act[ridx][cidx]>max_xmod_act){
				max_xmod_act = xmod_act[ridx][cidx];
				bmu->xpos = ridx;
				bmu->ypos = cidx;
			}
		}
	}
	return bmu;
}

/* compute forward activation - sensory afferents elicited activation */
void cln_compute_sensory_activation(som* s, neuron* bmu)
{	
	double win_val[2] = {bmu->xpos, bmu->ypos};
	double cur_val[2] = {0,0};
	for (int idx=0; idx<s->xsize; idx++){
		for(int jdx =0; jdx<s->ysize;jdx++){
			cur_val[0] = idx; cur_val[1] = jdx;
			s->neurons[idx][jdx].As = exp(-pow(cln_compute_norm(win_val, cur_val, 2), 2)/(2*pow(s->params->sigma[s->params->cur_epoch], 2)));
		}
	}
}

/* compute indirect activation - cross-modal elicited activation */
void cln_compute_xmodal_activation(som* s, neuron* bmu)
{
	double win_val[2] = {bmu->xpos, bmu->ypos};
	double cur_val[2] = {0,0};
	for (int idx=0; idx<s->xsize; idx++){
		for(int jdx =0; jdx<s->ysize;jdx++){
			cur_val[0] = idx; cur_val[1] = jdx;
			s->neurons[idx][jdx].Ax = exp(-pow(cln_compute_norm(win_val, cur_val, 2), 2)/(2*pow(s->params->sigma[s->params->cur_epoch], 2)));
		}
	}
}

/* comute the joint activation as a weighted sum of direct and indirect activations */
void cln_compute_joint_activation(som*s)
{
	for (int idx=0; idx<s->xsize; idx++){
		for(int jdx =0; jdx<s->ysize;jdx++){
			s->neurons[idx][jdx].At = (1.0 - s->params->gamma[s->params->cur_epoch])*s->neurons[idx][jdx].As + 
							(s->params->gamma[s->params->cur_epoch])*s->neurons[idx][jdx].Ax;
		}
	}
}

/* adapt the sensory projecton weights */
void cln_compute_sensory_weights(som* s, double* inp)
{
	 for (int idx=0; idx<s->xsize; idx++){
                for(int jdx =0; jdx<s->ysize;jdx++){
			for(int widx = 0; widx<s->insize; widx++)
	                        s->neurons[idx][jdx].W[widx] += s->params->alpha[s->params->cur_epoch]*s->neurons[idx][jdx].At*(inp[idx]-s->neurons[idx][jdx].W[widx]) - 
								s->params->xi[s->params->cur_epoch]*(s->neurons[idx][jdx].As - s->neurons[idx][jdx].At)*(inp[idx]-s->neurons[idx][jdx].W[widx]);
                }
        }
}

/* adapt the cross-modal hebbian links */
void cln_compute_xmodal_weights(som* s, som* d)
{
	 double meanAts = 0.0f, meanAtd = 0.0f;
	 for (int idx=0; idx<s->xsize; idx++){
                for(int jdx =0; jdx<s->ysize;jdx++){
			for(int hidx = 0; hidx<s->xsize; hidx++){
				for(int hjdx = 0; hjdx<s->ysize; hjdx++){
					switch(s->params->learn_rule){
       						case(NONE):
							s->neurons[hidx][hjdx].H[idx][jdx] = 0.0f;
							d->neurons[hidx][hjdx].H[idx][jdx] = 0.0f;			
                				break;
                				case (HEBBIAN):
							s->neurons[hidx][hjdx].H[idx][jdx] = s->params->kappa[s->params->cur_epoch]*s->neurons[hidx][hjdx].At*d->neurons[idx][jdx].At;
                                                        d->neurons[hidx][hjdx].H[idx][jdx] = s->params->kappa[s->params->cur_epoch]*d->neurons[hidx][hjdx].At*s->neurons[idx][jdx].At;
                				break;
        				        case(COVARIANCE):
							/* compute the mean activation at the source and destination map */
							for (int midx=0; midx<s->xsize; midx++){
						                for(int mjdx =0; mjdx<s->ysize;mjdx++){							
									meanAts += s->neurons[midx][mjdx].At;
									meanAtd += d->neurons[midx][mjdx].At;
								}
							}
							meanAts /= s->xsize*s->ysize;
							meanAtd /= d->xsize*d->ysize;
							s->neurons[hidx][hjdx].H[idx][jdx] = s->params->kappa[s->params->cur_epoch]*(s->neurons[hidx][hjdx].At - meanAts)*
																    (d->neurons[idx][jdx].At - meanAtd);
							d->neurons[hidx][hjdx].H[idx][jdx] = s->params->kappa[s->params->cur_epoch]*(d->neurons[hidx][hjdx].At - meanAtd)*
																    (s->neurons[idx][jdx].At - meanAts);

                				break;		
         				}
				}
			}
		}
	}
	/* normalize weights */
	/* fin min and max in the weight matrices */
	double minHs = DBL_MAX, minHd = DBL_MAX;
	double maxHs = 0.0f, maxHd = 0.0f;
	for (int idx = 0; idx<s->xsize; idx++){
		for(int jdx=0; jdx<s->ysize; jdx++){
			for (int hidx = 0; hidx<s->xsize; hidx++){
				for(int hjdx = 0; hjdx<s->ysize; hjdx++){
					if(s->neurons[idx][jdx].H[hidx][hjdx] < minHs)
						minHs = s->neurons[idx][jdx].H[hidx][hjdx];
					if(s->neurons[idx][jdx].H[hidx][hjdx] > maxHs)
						maxHs = s->neurons[idx][jdx].H[hidx][hjdx];
					if(d->neurons[idx][jdx].H[hidx][hjdx] < minHd)
						minHd = d->neurons[idx][jdx].H[hidx][hjdx];
					if(d->neurons[idx][jdx].H[hidx][hjdx] > maxHd)
						maxHd = d->neurons[idx][jdx].H[hidx][hjdx];	
				}
			}
		}
	}
	/* apply normalization */
	for (int idx = 0; idx<s->xsize; idx++){
                for(int jdx=0; jdx<s->ysize; jdx++){
                        for (int hidx = 0; hidx<s->xsize; hidx++){
                                for(int hjdx = 0; hjdx<s->ysize; hjdx++){
					s->neurons[idx][jdx].H[hidx][hjdx] = (s->neurons[idx][jdx].H[hidx][hjdx] - minHs)/(maxHs - minHs);
					d->neurons[idx][jdx].H[hidx][hjdx] = (d->neurons[idx][jdx].H[hidx][hjdx] - minHd)/(maxHd - minHd);
				}
			}
		}
	}
}

