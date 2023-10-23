#include "HMM.h"
#include "DLLMacros.h"

CDLL_EXPORT void HMM_Viterbi(int numsamples, int numstates, const double* priors, const double* transition, const double* emissionProb, int* output_sequence)
{
	HMM hmm(numstates);

	hmm.Viterbi(numsamples, priors, transition, emissionProb, output_sequence);
}

// 
// priors: [numstates]
// transition: [numstates,numstates]
// 
CDLL_EXPORT void HMM_ForwardBackward(int numsamples, int numstates, const double* priors, const double* transition, const double* logEmissionProb, double* logposterior, double* loga,double* logb)
{
	HMM hmm(numstates);

	hmm.ForwardBackward(numsamples, priors, transition, logEmissionProb, logposterior, loga, logb);
}

//CDLL_EXPORT void HMM_PosteriorTransistionProb(int numsamples, int numstates, const double* priors, const double* transition, const double* emissionProb, )

