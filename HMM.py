# -*- coding: utf-8 -*-

import ctypes as ct
import os
import math
import numpy as np
import numpy.ctypeslib as ctl
import matplotlib.pyplot as plt
import sys
import time
import os
import json

from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import rv_histogram
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter1d

class HMMLib:
    def __init__(self, debugMode=False):
        if ct.sizeof(ct.c_voidp) == 4:
            raise RuntimeError(f"The SMLM library can only be used with 64-bit python.")

        if debugMode:
            dllpath = "/hmmlib/x64/Debug/hmmlib.dll"
        else:
            dllpath = "/hmmlib/x64/Release/hmmlib.dll"

        thispath = os.path.dirname(os.path.abspath(__file__))
        abs_dllpath = os.path.abspath(thispath + dllpath)

        #print(abs_dllpath)
        dll = ct.CDLL(abs_dllpath)
        self.lib = dll

        i=ct.c_int32
        fa=ctl.ndpointer(np.float64, flags="aligned, c_contiguous")
        ia=ctl.ndpointer(np.int32, flags="aligned, c_contiguous")

        #void HMM_Viterbi(int numsamples, int numstates, const float* priors,
        # const float* transition, const float* emissionProb, int* output_sequence)
        self._HMM_Viterbi = dll.HMM_Viterbi
        self._HMM_Viterbi.argtypes = [ i,i,fa,fa,fa,ia ]
        self._HMM_Viterbi.restype = None
        #void HMM_ForwardBackward(int numsamples, int numstates, const float* priors,
        # const float* transition, const float* emissionProb, float* posterior)
        self._HMM_ForwardBackward = dll.HMM_ForwardBackward
        self._HMM_ForwardBackward.argtypes = [i,i,fa,fa,fa,fa,fa,fa]
        self._HMM_ForwardBackward.restype = None

    def forwardBackward(self, priors, transitionMatrix, logSampleProb):
        logSampleProb = np.ascontiguousarray(logSampleProb,dtype=np.float64)
        priors = np.ascontiguousarray(priors,dtype=np.float64)
        numstates = logSampleProb.shape[1]
        numsamples = logSampleProb.shape[0]
        tr = np.ascontiguousarray(transitionMatrix,dtype=np.float64)
        logposterior = np.zeros((numsamples,numstates),dtype=np.float64)
        loga = np.zeros((numsamples,numstates),dtype=np.float64)
        logb = np.zeros((numsamples,numstates),dtype=np.float64)
        self._HMM_ForwardBackward(numsamples,numstates,priors,tr,logSampleProb,logposterior,loga,logb)
        return logposterior, loga, logb

    def viterbi(self, priors, transitionMatrix, logSampleProb):
        logSampleProb = np.ascontiguousarray(logSampleProb,dtype=np.float64)
        priors = np.ascontiguousarray(priors,dtype=np.float64)
        tr = np.ascontiguousarray(transitionMatrix,dtype=np.float64)
        numstates = logSampleProb.shape[1]
        numsamples = logSampleProb.shape[0]
        output = np.zeros(numsamples, dtype=np.int32)
        self._HMM_Viterbi(numsamples,numstates,priors,tr,logSampleProb,output)
        return output

    def close(self):
        if self.lib is not None:
            # Free DLL so we can overwrite the file when we recompile
            ct.windll.kernel32.FreeLibrary.argtypes = [ct.wintypes.HMODULE]
            ct.windll.kernel32.FreeLibrary(self.lib._handle)
            self.lib = None


    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

class HMM:
    def __init__(self, numstates, debugMode=False):
        self.lib = HMMLib(debugMode)
        self.tr = np.ones(shape=(numstates,numstates)) / numstates
        self.priors = np.ones(numstates)/numstates

    def sample_hidden(self,numsamples):
        """
        Generate a random sequence of hidden states using the transition matrix and priors
        """
        state = np.random.choice(len(self.priors), 1, p=self.priors)[0]
        states = np.zeros(numsamples, dtype=int)
        for k in range(numsamples):
            states[k] = state
            state = np.random.choice(len(self.priors), 1, p=self.tr[state])[0]
        return states

    def sample(self, numsamples, generateEmission):
        """
        Returns a tuple (samples, hidden_states) generated using the transition matrix and priors
        """
        true_states = self.sample_hidden(numsamples)
        samples = []
        for k in range(numsamples):
            samples.append(generateEmission(true_states[k], k))
        return np.array(samples), true_states

    def viterbi(self, logSampleProb):
        """
        Compute the most likely hidden states for the given samples
        """
        return self.lib.viterbi(self.priors,self.tr,logSampleProb)

    def getGaussianEmissionDistributions(self, means, sigma):
        k = len(self.tr)
        distr = [norm(loc=means[i],scale=sigma[i]) for i in range(k)]
        return distr

    def computePosteriorWithGaussianEmissions(self, emissions, means, sigma):
        """
        Compute the posterior probability ( p(z_i|x) ), the prob of being in state z given all the data x.
        Returns matrix with shape [numsamples, numstates]
        """

        means = np.array(means)
        logSampleProb = norm.logpdf(emissions[:,None] - means[None], scale=sigma)
        return self.computePosterior(logSampleProb)

    def computeEmissionLogProb(self, emissions, distr, minProb=1e-80):
        """
        Compute a matrix of [numsamples, numstates] containing
        the probability for each state to emit the sample.
        Distr is a list of numstates scipy.stats probability distributions.
        """
        k = len(self.tr)
        emissionProb = np.zeros((len(emissions), k))

        prev = np.seterr(divide='ignore')['divide'] # log(0) will complain even though PDF can be zero
        for i in range(k):
            emissionProb[:,i] = distr[i].logpdf(emissions)
        np.seterr(divide=prev)
        return np.maximum(emissionProb,np.log(minProb))

    def getEmissionDistributions(self, trace, posterior, bins=200, hist_smoothing=0.01, plot=False):
        """
        Return random distributions from the emissions. scipy.stats.rv_histogram() is used
        """
        k = len(self.tr)
        distr = []
        if plot: plt.figure()
        for i in range(k):
            hist, binpos = np.histogram(trace, bins=bins, weights=posterior[:,i])
            smhist = gaussian_filter1d(hist, hist_smoothing/(binpos[1]-binpos[0]))
            distr.append(rv_histogram([smhist, binpos]))
            if plot:
                bincenter = 0.5*(binpos[:-1]+binpos[1:])
                plt.plot(bincenter,hist,label=f"State {i}")

        if plot:
            plt.legend()
            plt.title('Histogram per state')

        return distr

    def computePosterior(self, logSampleProb):
        """
        Compute the posterior probability ( p(z_i|x) ), the prob of being in state z given all the data x.
        Returns matrix with shape [numsamples, numstates]
        """
        logposterior, loga, logb = self.lib.forwardBackward(self.priors, self.tr, logSampleProb)
        return np.exp(logposterior)

    @staticmethod
    def _broadcast_over_axes(numdims, axes):
        if np.isscalar(axes): axes=[axes]
        """
        Make an indexing tuple to get all elements but insert None at the places defined by axes
        Example: broadcast_over_axes(4, (1,2)) returns [:,None,None,:]
        """
        # sorry about this horror
        return tuple([np.s_[:] if not a in axes else None for a in np.arange(numdims)])

    @staticmethod
    def logsum(logp,axes):
        """
        Sum log-probabilities while keeping floating-point precision in mind.
        """
        bcast = HMM._broadcast_over_axes(len(logp.shape),axes)
        b = np.max(logp,axes);
        return b + np.log (np.sum(np.exp(logp-b[bcast]), axes));

    @staticmethod
    def lognormalize(z,axes):
        """
        Normalize the log-probabilities over the given axes.
        This means that sum(exp(z),axes) will be a bunch of ones
        """
        bcast = HMM._broadcast_over_axes(len(z.shape),axes)
        return z - HMM.logsum(z,axes)[bcast]

    def computePosteriorTransitionProb(self, logSampleProb):
        """
        Run one iteration of baum-welch, computing the transition matrix from the posterior p(z | x)
        """
        logposterior, loga, logb = self.lib.forwardBackward(self.priors, self.tr, logSampleProb)

        # something about 32-bit FPU math went wrong so recompute it
        #logposterior = self.lognormalize(loga+logb, 1)

        #t0 = time.time()

        # probability of being in state i at t, and j at t+1
        # alpha(t) * beta(t+1) * p(state i -> state j) * emissionProb(t+1 in state j)
        logTr = np.log(self.tr)

        # xi is a joint probability, we need to normalize over 2 axes.
        logxi = loga[:-1,:,None]+logb[1:,None,:]+logTr[None,:,:]+logSampleProb[1:,None,:]
        logxi = HMM.lognormalize(logxi,(1,2))

        #t1 = time.time()
        #print(f"Normalizing took {(t1-t0)*1000:.1f} ms. Samples/s: {len(sampleProb)/(t1-t0):.1f}")

        #tr_post = np.exp(self.logsum(logxi, 0))
        xi = np.exp(logxi)
        psum = np.sum(np.exp(logposterior),0)
        #print(f"Sum of posterior: {psum}")
        tr_post = np.sum(xi,0) / psum[:,None]

        return tr_post


def test_baum_welch(nruns=10, nsamples=20000, iterations=10):

    results = []

    for r in range(nruns):
        # Generate 2 random transition probabilities that are pretty close to zero
        a,b = np.exp(-np.random.uniform(1,5,size=2))
        true_tr = np.array([[1-a,a], [b, 1-b]])

        hmm = HMM(2)
        hmm.priors = [0.5, 0.5]
        hmm.tr = true_tr

        sigmaNoise=0.2
        x, z = hmm.sample(nsamples, lambda s, k: s + np.random.normal(scale=sigmaNoise))

        if r == 0:
            plt.figure()
            plt.plot(x[:1000], label="Samples")
            plt.plot(z[:1000], label="True")
            plt.legend()
            plt.show()

        states, X = np.meshgrid(np.arange(len(hmm.priors)), x)

        # emissionProb is a matrix holding the probabilities for each sample for each hidden state
        emissionProb = norm.pdf(X-states,scale=sigmaNoise)

        # Now assume we know nothing and re-estimate the transition matrix
        hmm.tr = [[0.5, 0.5], [0.5, 0.5]]

        for i in range(iterations):
            est_tr = hmm.computePosteriorTransitionProb(emissionProb)
            hmm.tr = est_tr

        print( f"True: {true_tr}. Estim: {est_tr}")

        results.append([true_tr, est_tr])

    results = np.array(results)

    errors = np.abs(results[:,0]-results[:,1])

    print(np.std(errors,0))
    return results


if __name__ == '__main__':
    # test_baum_welch(10)

    hmm = HMM(2)

    true_tr = np.array([[0.97, 0.03], [0.1, 0.9]])
    hmm.tr = true_tr
    hmm.priors = [0.2, 0.8]
    numsamples= 500

    sigmaNoise=1
    t0 = time.time()
    x, z = hmm.sample(numsamples, lambda s, k: s + np.random.normal(scale=sigmaNoise))
    t1 = time.time()
    print(f"Sampling took {(t1-t0)*1000:.1f} ms. Samples/s: {len(x)/(t1-t0):.1f}")

    distr = hmm.getGaussianEmissionDistributions([0,1], [sigmaNoise,sigmaNoise])
    emissionLogProb = hmm.computeEmissionLogProb(x, distr)

    reps=100
    t0 = time.time()
    for k in range(reps):
        z_viterbi = hmm.viterbi(emissionLogProb)
    t1 = time.time()
    # print(f"Viterbi took {(t1-t0)*1000:.1f} ms. Samples/s: {reps*numsamples/(t1-t0):.1f}")

    t0 = time.time()
    for k in range(reps):
        posterior = hmm.computePosterior(emissionLogProb)
    t1 = time.time()
    # print(f"Forward backward took {(t1-t0)*1000:.1f} ms. Samples/s: {reps*numsamples/(t1-t0):.1f}")

    plt.figure()
    # plt.plot(z, label="True")
    # plt.plot(x, label="Samples",linewidth=0.5)
    plt.plot(posterior[:,1], label="p(z=1)", linewidth=3)
    plt.plot(z_viterbi, '--',label="Viterbi",  linewidth=3)
    plt.legend()
    plt.show()
    est_tr = hmm.computePosteriorTransitionProb(emissionLogProb)
    print(est_tr)

########################################################################################################################
def plot_trace_sections(original, smoothed, viterbi, windowlen, dir_, tr_name, fig=None, save=False):
    total = len(original)

    numsections = (total+windowlen-1) // windowlen
    if fig is None:
        fig4, ax4 = plt.subplots(numsections, 1, sharex=True)#, figsize=(10, numsections*2), dpi=400)
    else:
        fig4 = fig
        ax4 = fig4.subplots(numsections, 1, sharex=True)
    for k in range(numsections):
        start = k*windowlen
        end = np.minimum( (k+1)*windowlen, total)
        t = np.arange(start,end )
        ax4[k].plot(original[t], c='k', label='original', linewidth=0.3, alpha=0.4)
        ax4[k].plot(smoothed[t], c='k', label='smoothed', linewidth=0.8, alpha=0.6)
        ax4[k].plot(viterbi[t], c='r', label='max-likelihood', linewidth=1.2, alpha=0.7)
    ax4[0].set_title('HMM fits')
    ax4[0].legend(loc='upper right')
    if save:
        fig4.savefig(f'{dir_}hmm_fit_{tr_name}.png')

def viterbi_transition_matrix(ml_trace, numstates):
    """
    Generate a transition matrix by counting the max-likelihood (viterbi) states switches
    """

    counts = np.zeros((numstates,numstates),dtype=np.int32)

    transition_events = 0
    for i in range(numstates):
        for j in range(numstates):
            counts[i,j] = np.sum((ml_trace[:-1] == i) & (ml_trace[1:] == j))
            if j != i:
                transition_events += counts[i,j]

    print(f"Max-Likelihood transition matrix based on {transition_events} transition events")
    transition_matrix = counts / np.sum(counts, 1)[:,None]
    return transition_matrix



def run_hmm(data, tr_name, fig, tr_means, dir_=None, st_prob=(0.45,0.1,0.45),
            sigma=0.05, numstates=2, freq=20, frames=40000, save=False, sequence=0):
### freq: Gaussian filter frequency
### numstates: number of states for HMM
### st_prob: probability of states

    tr = data
    if save: ## saving the tr_means in a text file
        fname = dir_ + 'state_means.txt'
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                dict_ = json.load(f)
            if tr_name not in dict_.keys():
                dict_[tr_name] = tr_means.tolist()
                with open(fname, 'w') as f:
                    json.dump(dict_, f)
        else:
            dict_ = {tr_name : tr_means.tolist()}
            with open(fname, 'w') as f:
                json.dump(dict_, f)

    smoothed = gaussian_filter1d(tr, freq)

    if sequence==0:
        fig1 = fig
        ax1 = fig1.add_subplot()
    else:
        fig1, ax1 = plt.subplots()
    ### plot histogram of smoothed data
    ax1.hist(smoothed, bins=400)
    ax1.set_title(f'Histogram for smoothed traces {tr_name}')
    if save:
        fig1.savefig(f'{dir_}smoothed_hist_{tr_name}.png')


    # how many iterations do we update both emission prob?
    emission_prob_iterations = 1
    # how many iterations do we update transition matrix?
    baum_welch_iterations = 0

    # transition matrices per trace
    trace_tr = []
    initial_transition_prob = 1e-9

    hmm = HMM(numstates,debugMode=False)
    hmm.priors = st_prob
    hmm.tr = np.ones((numstates,numstates)) * initial_transition_prob
    hmm.tr[np.diag_indices(numstates)] = 1-initial_transition_prob

    # Get an initial estimate of posterior prob. dist. using normal distributions
    distr = hmm.getGaussianEmissionDistributions(tr_means, sigma*np.ones(numstates))
    emissionLogProb = hmm.computeEmissionLogProb(tr, distr)
    posterior = hmm.computePosterior(emissionLogProb)

    for j in range(np.max([baum_welch_iterations,emission_prob_iterations])):
        if j<baum_welch_iterations:
            emissionLogProb = hmm.computeEmissionLogProb(tr, distr)
            hmm.tr = hmm.computePosteriorTransitionProb(emissionLogProb)

        if j<emission_prob_iterations:
            distr = hmm.getEmissionDistributions(tr, posterior, bins=400,
                                                 hist_smoothing=0.005, plot=False)
            updated_trace_means = np.array([distr[j].mean() for j in range(numstates)])
            print(f"Updated state mean position: {updated_trace_means}")

            if sequence == 2:
                fig2 = fig
                ax2 = fig2.add_subplot()
            else:
                fig2, ax2 = plt.subplots()

            for k in range(numstates):
                x = np.linspace(0.0, 0.4, 400)
                ax2.plot(x, distr[k].pdf(x), label=f"State {k}")
            ax2.set_title(f'Probability density per state - iteration {j}')

            if save:
                fig2.savefig(f'{dir_}statepdf_{tr_name}')

            emissionLogProb = hmm.computeEmissionLogProb(tr, distr)

    ml_trace = hmm.viterbi(emissionLogProb)

    ml_transition_matrix = viterbi_transition_matrix(ml_trace, numstates)
    trace_tr.append(ml_transition_matrix)
    print(f"Transition matrix for trace {tr_name}: {ml_transition_matrix}")

    if sequence==1:
        plot_trace_sections(tr, smoothed, tr_means[ml_trace], frames, dir_, tr_name, fig=fig, save=save)
    if save and sequence != 1:
        plot_trace_sections(tr, smoothed, tr_means[ml_trace], frames, dir_, tr_name, save=save)

    # m = trace_means[i]
    # plt.figure()
    # t = np.arange(0,10000)
    # plt.plot(t,traces[i][t],label='measured')
    # plt.plot(t,trace_means[i][ml_trace][t], label='true state')
    # #plt.plot(t,m[0]+(m[1]-m[0])*posterior[t,1], label='p(z=0)')
    # plt.plot(t,smoothed[i][t],label='smoothed')
    # plt.title(f'Max-Likelihood fit - trace {i}')
    # plt.legend()
    # plt.savefig(f'traces/mlfit{i}.png')

    if sequence==3:
        fig3 = fig
        ax3 = fig3.add_subplot()
    else:
        fig3, ax3 = plt.subplots()

    ax3.hist([smoothed[ml_trace == j] for j in range(numstates)],
             bins=30, label=[f'state {j}' for j in range(numstates)])
    ax3.legend()
    ax3.set_title('Per-state histogram of smoothed-positions')

    if save:
        fig3.savefig(f'{dir_}hist_{tr_name}.png')
        print('Figures are saved in: ..', dir_)

    plt.show()
    plt.close('all')




# if __name__ == '__main__':
#     path_ = 'traces/'
#     trace = np.genfromtxt(r'traces\trace_1.txt')
#     trace_means = np.array([0.15,0.2,0.26])
#
#     run_hmm(trace, 1, fig=None, tr_means=trace_means, dir_=path_,
#             sigma=0.05, numstates=3, freq=20, frames=40000, save=True, sequence=-1)
