#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "hmm.h"
#define MAX_HMM_NUM 10
using namespace std;

HMM hmms[MAX_HMM_NUM];
char name[MAX_HMM_NUM][30];
int HMM_NUM;

double HMMProb(int hid, char*seq){
    int T = strlen(seq);
    HMM hmm = hmms[hid];
    int state_num = hmm.state_num;
    double delta[T][state_num];

    for(int s=0;s<state_num;s++) //initial delta
        delta[0][s] = hmm.initial[s]*hmm.observation[seq[0]-'A'][s];

    for(int t=1;t<T;t++){ //forward: for each time slice
        int obsv = seq[t]-'A'; //observation at time t
        for(int s=0;s<state_num;s++){ //for each current state
            delta[t][s] = 0;
            for(int ps=0;ps<state_num;ps++) //for each previous state
                delta[t][s] = max(delta[t][s], delta[t-1][ps]*hmm.transition[ps][s]);
            
            delta[t][s] *= hmm.observation[obsv][s];
        }
    }

    double prob = 0;
    for(int s=0;s<state_num;s++) //find max prob
        prob = max(prob, delta[T-1][s]);

    return prob;
}

void testHMM(char *test_txt, char *result_txt){
    FILE *fp_test = open_or_die(test_txt, "r");
    FILE *fp_result = open_or_die(result_txt, "w");

    char seq[MAX_TIME];
    int N=0;
    while(fscanf(fp_test, "%s", seq)>0){
        double maxProb = 0;
        int maxId = 0;
        N++;
        for(int hid=0;hid<HMM_NUM;hid++){
            double prob = HMMProb(hid, seq);
            if(prob > maxProb){
                maxProb = prob;
                maxId = hid;
            }
            //fprintf(fp_result, "%d %g\n", hid, prob);
        }
        //fprintf(fp_result, "model_0%d.txt\t%g\n", maxId+1, maxProb);
        fprintf(fp_result, "%s\t%g\n", name[maxId], maxProb);
    }
    fclose(fp_test);
    fclose(fp_result);
}

int main(int argc, char** argv){
    if(argc!=4){
        fprintf(stderr, "Error: Need 3 arguments (modellist.txt  testing_data.txt  result.txt).\n");
        return 0;
    }    

    //set up arguments
    char* list_txt = argv[1];
    char* test_txt = argv[2];
    char* result_txt = argv[3];

    HMM_NUM = load_models(list_txt, hmms, MAX_HMM_NUM);
    
    FILE *fp_list = open_or_die(list_txt, "r");
    int i = 0;
    while(fscanf(fp_list, "%s", name[i])>0) i++;

    //for(int i=0;i<HMM_NUM;i++) dumpHMM(stderr, &hmms[i]);
    
    testHMM(test_txt, result_txt);

    return 0;
}