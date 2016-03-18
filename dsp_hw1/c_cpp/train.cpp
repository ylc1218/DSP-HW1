#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "hmm.h"
#define MAX_TIME 55
HMM hmm;

/*
typedef struct{
   char *model_name;
   int state_num;                   //number of state
   int observ_num;                  //number of observation
   double initial[MAX_STATE];           //initial prob.
   double transition[MAX_STATE][MAX_STATE]; //transition prob.
   double observation[MAX_OBSERV][MAX_STATE];   //observation prob.
} HMM;
*/

void trainHmm(char* seq_model){
    FILE *fp = open_or_die(seq_model, "r");

    char seq[MAX_TIME];
    int state_num = hmm.state_num;
    double gamma[MAX_TIME][state_num];
    double epsilon[state_num][state_num];

    memset(gamma, 0, sizeof(gamma));
    memset(epsilon, 0, sizeof(epsilon));

    int N=0, T;
    while(fscanf(fp, "%s", seq)>0){
        T = strlen(seq);
        N++;

        //calculate alpha
        double alpha[T][state_num];
        for(int s=0;s<state_num;s++){ //initial alpha
            alpha[0][s] = hmm.initial[s]*hmm.observation[seq[0]-'A'][s];
        }

        for(int t=1;t<T;t++){ //forward: for each time slice
            int obsv = seq[t]-'A'; //observation at time t
            for(int s=0;s<state_num;s++){ //for each current state
                alpha[t][s] = 0;
                for(int ps=0;ps<state_num;ps++){ //for each previous state
                    alpha[t][s] += alpha[t-1][ps]*hmm.transition[ps][s];
                }
                alpha[t][s] *= hmm.observation[obsv][s];
            }
        }

        //calculate beta
        double beta[T][state_num];
        for(int s=0;s<state_num;s++) beta[T-1][s] = 1; //initial beta

        for(int t=T-2;t>=0;t--){//backward: for each time slice
            int obsv = seq[t+1]-'A'; //observation at time t+1
            for(int s=0;s<state_num;s++){ //for each current state
                beta[t][s] = 0;
                for(int ns=0;ns<state_num;ns++) // for each next state
                    beta[t][s] += hmm.transition[s][ns]*hmm.observation[obsv][ns]*beta[t+1][ns];
            }
        }

        //calculate gamma and epsilon
        for(int t=0;t<T;t++){
            double sum = 0;
            int obsv = seq[t]-'A';
            for(int s=0;s<state_num;s++) sum += alpha[t][s]*beta[t][s];
            //printf("%d:sum = %lf\n", t, sum);
            for(int s=0;s<state_num;s++){
                //calculate gamma
                gamma[t][s] += alpha[t][s]*beta[t][s]/sum;
                   
                //calculate epsilon
                if(t==T-1) continue;
                for(int ns=0;ns<state_num;ns++)
                    epsilon[s][ns] += alpha[t][s]*hmm.transition[s][ns]*hmm.observation[obsv][ns]*beta[t+1][ns]/sum;
            }
        }
    }
    fclose(fp);


    for(int s=0;s<state_num;s++){
        //update pi
        hmm.initial[s] = gamma[0][s]/N;

        //update transition(A)
        double gamma_sum = 0;
        for(int t=0;t<T-1;t++) gamma_sum += gamma[t][s];
        for(int ns=0;ns<state_num;ns++)
            hmm.transition[s][ns] = epsilon[s][ns]/gamma_sum;
    }

    //update observation(B)
    for(int obsv=0;obsv<hmm.observ_num;obsv++){
        for(int s=0;s<state_num;s++){
            double gamma_sum = 0, obsv_gamma_sum = 0;
            for(int t=0;t<T;t++){
                gamma_sum += gamma[t][s];
            }
        }
    }
}

int main(int argc, char** argv){
    if(argc!=5){
        printf("Error: Need 4 arguments (#iteration, model_init.txt, seq_model.txt, model.txt).\n");
        return 0;
    }
    
    //set up arguments
    int itr = atoi(argv[1]);
    char* init_txt = argv[2];
    char* seq_model = argv[3];
    char* out_txt = argv[4];

    loadHMM(&hmm, init_txt);

    for(int i=0;i<itr;i++) trainHmm(seq_model);

    FILE *fp_out = open_or_die(out_txt, "w");
    dumpHMM(fp_out, &hmm );
    fclose(fp_out);

    return 0;
}