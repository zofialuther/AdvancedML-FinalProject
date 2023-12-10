# Run NF model on all combinations of base and target distributions
# This will require the normflows package

from test_dist import run
from new_targets import Spiral, SwissRoll, KleinBottle, MobiusStrip
from new_bases import BetaDistribution, GammaDistribution, ExponentialDistribution, PoissonDistribution, StudentTDistribution, CauchyDistribution, SawtoothDistribution
import normflows as nf
#import torch
#torch.autograd.set_detect_anomaly(True)

D = 2

# 2D dists
base_distributions = [("Diagonal Gaussian", nf.distributions.DiagGaussian(D)), ("Uniform", nf.distributions.base.Uniform(D)), ("Uniform-Gaussian", nf.distributions.UniformGaussian(D, [0])), ("GMM", nf.distributions.GaussianMixture(2, D)), ("Beta", BetaDistribution(D)), ("Gamma", GammaDistribution(D)), ("Exponential", ExponentialDistribution(D)), ("Poisson", PoissonDistribution(D)), ("Student's t", StudentTDistribution(D)), ("Cauchy", CauchyDistribution(D)), ("Sawtooth", SawtoothDistribution(D))]
target_distributions = [("Two Modes", nf.distributions.TwoModes(D, 0.1)), ("Two Moons", nf.distributions.TwoMoons()), ("Circular GMM", nf.distributions.CircularGaussianMixture()), ("Two Ring Mixture", nf.distributions.RingMixture()), ("Spiral", Spiral(3))]

#base_distributions = [("Diagonal Gaussian", nf.distributions.DiagGaussian(D))]
#target_distributions = [("Spiral", Spiral(3))]

#target_distributions = [("Swiss Roll", SwissRoll()), ("Klein Bottle", KleinBottle()), ("Mobius Strip", MobiusStrip())]

#base_distributions = [("Beta", BetaDistribution(D)), ("Gamma", GammaDistribution(D)), ("Exponential", ExponentialDistribution(D)), ("Poisson", PoissonDistribution(D)), ("Student's t", StudentTDistribution(D)), ("Cauchy", CauchyDistribution(D)), ("Sawtooth", SawtoothDistribution(D, 4))]

out = ["2D:"]
error = []

for base in base_distributions:
    for target in target_distributions:

        try:
            out.append("Loss for " + base[0] + " approximating " + target[0] + ": " + str(run(base[1], target[1])[-1]))
            print(out[-1])
        except Exception as e:
            error.append(e)

err = [str(e) for e in error]
print("\n----------------\n".join(err))
print("\n".join(out))
out.append("3D:")

D = 3
base_distributions = [("Diagonal Gaussian", nf.distributions.DiagGaussian(D)), ("Uniform", nf.distributions.Uniform(D)), ("Uniform-Gaussian", nf.distributions.UniformGaussian(D, [0])), ("GMM", nf.distributions.GaussianMixture(2, D)), ("Beta", BetaDistribution(D)), ("Gamma", GammaDistribution(D)), ("Exponential", ExponentialDistribution(D)), ("Poisson", PoissonDistribution(D)), ("Student's t", StudentTDistribution(D)), ("Cauchy", CauchyDistribution(D)), ("Sawtooth", SawtoothDistribution(D))]
target_distributions = [("Swiss Roll", SwissRoll()), ("Klein Bottle", KleinBottle()), ("Mobius Strip", MobiusStrip())]

for base in base_distributions:
    for target in target_distributions:

        try:
            out.append("Loss for " + base[0] + " approximating " + target[0] + ": " + str(run(base[1], target[1], D)[-1]))
            print(out[-1])
        except Exception as e:
            error.append(e)

err = [str(e) for e in error]
print("\n----------------\n".join(err))
print("\n".join(out))