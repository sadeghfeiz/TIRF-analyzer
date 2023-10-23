#pragma once
#include <vector>


class HMM {
public:
	int numstates;

	HMM(int numstates) : numstates(numstates) {	 }


	// assuming sizeof(int)=sizeof(double)

	// logTransition[i * N + j] = probability of going from i to j
	void Viterbi(int numsamples, const double* priors, const double* transition, const double* emissionLogProb, int* output_sequence)
	{
		std::vector<int> temp_choice(numsamples * numstates);
		std::vector<double> temp_logmu(numsamples * numstates);
		std::vector<double> logTransition = Log(transition, numstates * numstates);

		//	logmu(1, :) = log(prior) + logemission(samples(1, :), 1:m);
		auto logmu = [&](int smp, int state) -> double& { return temp_logmu[numstates*smp + state]; };
		auto choice = [&](int smp, int state) -> int& { return temp_choice[numstates*smp + state]; };

		for (int z = 0; z < numstates; z++)
			logmu(0, z) = log(priors[z]) + emissionLogProb[0*numstates+z];

		for (int k = 1; k < numsamples; k++) {
			for (int z = 0; z < numstates; z++) {
				double a = logTransition[0 * numstates + z] + logmu(k - 1, 0);
				int maxelem = 0;
				for (int j = 1; j < numstates; j++) {
					double b = logTransition[j * numstates + z] + logmu(k - 1, j);
					if (b > a) {
						maxelem = j; a = b;
					}
				}
				choice(k, z) = maxelem;
				logmu(k, z) = a + emissionLogProb[k*numstates+ z];
			}
		}

		int laststate = 0;
		for (int i = 1; i < numstates; i++)
			if (logmu(numsamples - 1, laststate) < logmu(numsamples - 1, i)) laststate = i;
		output_sequence[numsamples - 1] = laststate;

		for (int k = numsamples - 2; k >= 0; k--) {
			laststate = choice(k + 1, laststate);
			output_sequence[k] = laststate;
		}
	}


	static double logsum(double *f, int n)
	{
		double b = f[0];
		for (int i = 1; i < n; i++) {
			double t = f[i];
			if (t > b) b = t;
		}
		if (isinf(b))
			return -INFINITY;
		double sum = 0.0f;
		for (int i = 0; i < n; i++)
			sum += exp(f[i] - b);
		return b + log(sum);
	}

	static void lognormalize(double *v, int n)
	{
		double s = logsum(v, n);
		for (int i = 0; i < n; i++)
			v[i] -= s;
	}

	static std::vector<double> Log(const double* d, int size)
	{
		std::vector<double> r(size);
		for (int i = 0; i < size; i++)
			r[i] = log(d[i]);
		return r;
	}

	void ForwardBackward(int numsamples, const double* priors, const double* transition, const double* logEmissionProb, double* logposterior, double* loga_, double* logb_)
	{
		std::vector<double> logTransition = Log(transition, numstates * numstates);
		std::vector<double> temp2(numstates);

		auto loga = [&](int smp, int state) -> double& { return loga_[numstates*smp + state]; };
		auto logb = [&](int smp, int state) -> double& { return logb_[numstates*smp + state]; };

		// a(k, l) is the alpha(k) for the value of z=l
		// alpha(k, l) = p(x(1:k), z(k) | model)

		// Forward algorithm:
		// Goal: compute p(z(k), x(1:k))
		for (int z = 0; z < numstates; z++)
			loga(0, z) = logf(priors[z]) + logEmissionProb[0*numstates+ z];

		for (int k = 1; k < numsamples; k++)
		{
			for (int z = 0; z < numstates; z++) {
				for (int i = 0; i < numstates; i++)
					temp2[i] = loga(k - 1, i) + logTransition[i*numstates + z];
				loga(k, z) = logEmissionProb[k*numstates+ z] + logsum(temp2.data(), numstates);
			}
		}

		for (int z = 0; z < numstates; z++)
			logb(numsamples - 1, z) = 0;

		for (int k = numsamples - 2; k >= 0; k--) {
			for (int z = 0; z < numstates; z++) {
				for (int i = 0; i < numstates; i++)
					temp2[i] = logb(k + 1, i) + logEmissionProb[ (k + 1) * numstates + i ] + logTransition[z*numstates + i];
				logb(k, z) = logsum(temp2.data(), numstates);
			}
		}

		for (int k = 0; k < numsamples;k++) {
			for (int z = 0; z < numstates; z++)
				temp2[z] = loga(k, z) + logb(k, z);
			lognormalize(temp2.data(),numstates);

			for (int i = 0; i < numstates; i++)
				logposterior[k * numstates + i] = temp2[i];
		}
	}



};


