#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "hmm.h"
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
    int state_num = hmm.state_num, obsv_num = hmm.observ_num;
    double gamma[MAX_TIME][state_num], obsv_gamma[obsv_num][state_num];
    double epsilon[state_num][state_num];

    memset(gamma, 0, sizeof(gamma));
    memset(obsv_gamma, 0, sizeof(obsv_gamma));
    memset(epsilon, 0, sizeof(epsilon));

    int N=0, T;
    while(fgets(seq, MAX_TIME, fp)>0){
        T = strlen(seq)-1;
        N++;
        for(int t=0;t<T;t++) seq[t] -= 'A';
            
        //calculate alpha
        double alpha[T][state_num];
        for(int s=0;s<state_num;s++){ //initial alpha
            alpha[0][s] = hmm.initial[s]*hmm.observation[seq[0]][s];
            //printf("alpha[%d][%d] = %g\n",  0, s, alpha[0][s]);
        }

        for(int t=1;t<T;t++){ //forward: for each time slice
            for(int s=0;s<state_num;s++){ //for each current state
                alpha[t][s] = 0;
                for(int ps=0;ps<state_num;ps++){ //for each previous state
                    alpha[t][s] += alpha[t-1][ps]*hmm.transition[ps][s];
                }
                alpha[t][s] *= hmm.observation[seq[t]][s];
                //printf("alpha[%d][%d] = %g\n", t, s, alpha[t][s]);
            }
        }

        //calculate beta
        double beta[T][state_num];
        for(int s=0;s<state_num;s++){
            beta[T-1][s] = 1; //initial beta
            //printf("beta[%d][%d] = %g\n", T-1, s, beta[T-1][s]);
        }

        for(int t=T-2;t>=0;t--){//backward: for each time slice
            for(int s=0;s<state_num;s++){ //for each current state
                beta[t][s] = 0;
                for(int ns=0;ns<state_num;ns++) // for each next state
                    beta[t][s] += hmm.transition[s][ns]*hmm.observation[seq[t+1]][ns]*beta[t+1][ns];
                //printf("beta[%d][%d] = %g\n", t, s, beta[t][s]);
            }
        }

        //calculate gamma and epsilon
        for(int t=0;t<T;t++){
            double sumArr[state_num], sum = 0;
            for(int s=0;s<state_num;s++){
                sumArr[s] = alpha[t][s]*beta[t][s];
                sum += sumArr[s];
            }

            for(int s=0;s<state_num;s++){
                //calculate gamma
                gamma[t][s] += sumArr[s]/sum;
                //printf("gamma[%d][%d][%d] = %g\n", N-1, t, s, sumArr[s]/sum);
                obsv_gamma[seq[t]][s] += sumArr[s]/sum;
                   
                //calculate epsilon
                if(t==T-1) continue;
                for(int ns=0;ns<state_num;ns++)
                    epsilon[s][ns] += (alpha[t][s]*hmm.transition[s][ns]*hmm.observation[seq[t+1]][ns]*beta[t+1][ns])/sum;
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
    for(int obsv=0;obsv<obsv_num;obsv++){
        for(int s=0;s<state_num;s++){
            double gamma_sum = 0;
            for(int t=0;t<T;t++) gamma_sum += gamma[t][s];
            hmm.observation[obsv][s] = obsv_gamma[obsv][s]/gamma_sum;
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
    dumpHMM(fp_out, &hmm);
    fclose(fp_out);

    return 0;
}