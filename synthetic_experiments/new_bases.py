# Our implementations for new base distributions


from normflows.distributions import BaseDistribution
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Beta, Gamma, Exponential, Poisson, StudentT, Cauchy

# Beta, Gamma, Exponential, Poisson, Student’s t, Cauchy, and Sawtooth base distributions


class BetaDistribution(BaseDistribution):
    """
    Beta distribution for flow-based models
    """

    def __init__(self, shape, alpha=1.0, beta=1.0, trainable=True):
        """Constructor

        Args:
          shape: Shape of the distribution (assumes each dimension is independent)
          alpha: Initial alpha value for Beta distribution
          beta: Initial beta value for Beta distribution
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        self.shape = self._to_tuple(shape)
        self.n_dim = len(self.shape)
        self.d = np.prod(self.shape)

        self.alpha = nn.Parameter(torch.full(self.shape, float(alpha))) if trainable else torch.full(self.shape, float(alpha))
        self.beta = nn.Parameter(torch.full(self.shape, float(beta))) if trainable else torch.full(self.shape, float(beta))

    def _to_tuple(self, shape):
        if isinstance(shape, int):
            return (shape,)
        elif isinstance(shape, list):
            return tuple(shape)
        return shape

    def forward(self, num_samples=1, context=None):
        samples = torch.empty(num_samples, *self.shape)
        log_prob = torch.empty(num_samples, *self.shape)
        for i in range(self.shape[0]):
            dist = Beta(self.alpha[i], self.beta[i])
            samples[:, i] = dist.sample((num_samples,))
            log_prob[:, i] = dist.log_prob(samples[:, i])
        return samples, torch.sum(log_prob, dim=1)

    def log_prob(self, z, context=None):
        # Clamp the input values to the support of the Beta distribution
        z_clamped = torch.clamp(z, 0.0, 1.0)

        log_prob = torch.zeros_like(z)
        for i in range(self.shape[0]):
            dist = Beta(self.alpha[i], self.beta[i])
            log_prob[:, i] = dist.log_prob(z_clamped[:, i])

        # Penalize values outside the [0, 1] range
        outside_support_penalty = torch.where(
            (z < 0) | (z > 1),
            torch.full_like(z, -float('inf')),
            torch.zeros_like(z)
        )

        return torch.sum(log_prob, dim=1) + torch.sum(outside_support_penalty, dim=1)

    def sample(self, num_samples=1, context=None):
        samples, _ = self.forward(num_samples, context)
        return samples





class GammaDistribution(BaseDistribution):
    """
    Gamma distribution for flow-based models
    """

    def __init__(self, shape, alpha=1.0, beta=1.0, trainable=True):
        """Constructor

        Args:
          shape: Shape of the distribution (assumes each dimension is independent)
          alpha: Initial alpha value for Gamma distribution
          beta: Initial beta value for Gamma distribution
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        self.shape = self._to_tuple(shape)
        self.n_dim = len(self.shape)
        self.d = np.prod(self.shape)

        self.alpha = nn.Parameter(torch.full(self.shape, float(alpha))) if trainable else torch.full(self.shape, float(alpha))
        self.beta = nn.Parameter(torch.full(self.shape, float(beta))) if trainable else torch.full(self.shape, float(beta))

    def _to_tuple(self, shape):
        if isinstance(shape, int):
            return (shape,)
        elif isinstance(shape, list):
            return tuple(shape)
        return shape

    def forward(self, num_samples=1, context=None):
        samples = torch.empty(num_samples, *self.shape)
        log_prob = torch.empty(num_samples, *self.shape)
        for i in range(self.shape[0]):
            dist = Gamma(self.alpha[i], self.beta[i])
            samples[:, i] = dist.sample((num_samples,))
            log_prob[:, i] = dist.log_prob(samples[:, i])
        return samples, torch.sum(log_prob, dim=1)

    def log_prob(self, z, context=None):
        # Clamp the input values to be positive
        z_clamped = torch.clamp(z, min=0.0)

        log_prob = torch.zeros_like(z)
        for i in range(self.shape[0]):
            dist = Gamma(self.alpha[i], self.beta[i])
            log_prob[:, i] = dist.log_prob(z_clamped[:, i])

        # Penalize values outside the positive range
        outside_support_penalty = torch.where(
            z <= 0,
            torch.full_like(z, -float('inf')),
            torch.zeros_like(z)
        )

        return torch.sum(log_prob, dim=1) + torch.sum(outside_support_penalty, dim=1)

    def sample(self, num_samples=1, context=None):
        samples, _ = self.forward(num_samples, context)
        return samples



class ExponentialDistribution(BaseDistribution):
    """
    Exponential distribution for flow-based models
    """

    def __init__(self, shape, beta=1.0, trainable=True):
        """Constructor

        Args:
          shape: Shape of the distribution (assumes each dimension is independent)
          beta: Initial beta value for Exponential distribution
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        self.shape = self._to_tuple(shape)
        self.n_dim = len(self.shape)
        self.d = np.prod(self.shape)

        self.beta = nn.Parameter(torch.full(self.shape, float(beta))) if trainable else torch.full(self.shape, float(beta))

    def _to_tuple(self, shape):
        if isinstance(shape, int):
            return (shape,)
        elif isinstance(shape, list):
            return tuple(shape)
        return shape

    def forward(self, num_samples=1, context=None):
        # Pre-allocate tensors for samples and log probabilities
        samples = torch.empty(num_samples, *self.shape)
        log_prob = torch.empty(num_samples, *self.shape)

        for i in range(self.shape[0]):
            dist = Exponential(self.beta[i])
            samples_i = dist.sample((num_samples,))
            log_prob_i = dist.log_prob(samples_i)

            # Assign values to pre-allocated tensors
            samples[:, i] = samples_i
            log_prob[:, i] = log_prob_i

        return samples, torch.sum(log_prob, dim=1)

    def log_prob(self, z, context=None):
        z_clamped = torch.clamp(z, min=0.0)

        # Manually compute log probabilities for the Exponential distribution
        # log_prob = -beta * z_clamped - log(beta)
        log_prob = -self.beta.unsqueeze(0) * z_clamped - torch.log(self.beta).unsqueeze(0)

        # Apply penalty for values outside the support
        outside_support_penalty = torch.where(
            z <= 0,
            torch.tensor(-float('inf')).expand_as(z),
            torch.tensor(0.0).expand_as(z)
        )

        return torch.sum(log_prob, dim=1) + torch.sum(outside_support_penalty, dim=1)

    def sample(self, num_samples=1, context=None):
        samples, _ = self.forward(num_samples, context)
        return samples
    

class PoissonDistribution(BaseDistribution):
    """
    Poisson distribution for flow-based models
    """

    def __init__(self, shape, lambda_=1.0, trainable=True):
        """Constructor

        Args:
          shape: Shape of the distribution (assumes each dimension is independent)
          lambda_: Initial lambda value for Poisson distribution
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        self.shape = self._to_tuple(shape)
        self.n_dim = len(self.shape)
        self.d = np.prod(self.shape)

        self.lambda_ = nn.Parameter(torch.full(self.shape, float(lambda_))) if trainable else torch.full(self.shape, float(lambda_))

    def _to_tuple(self, shape):
        if isinstance(shape, int):
            return (shape,)
        elif isinstance(shape, list):
            return tuple(shape)
        return shape

    def forward(self, num_samples=1, context=None):
        # Pre-allocate tensors for samples and log probabilities
        samples = torch.empty(num_samples, *self.shape, dtype=torch.float)
        log_prob = torch.empty(num_samples, *self.shape)

        for i in range(self.shape[0]):
            dist = Poisson(self.lambda_[i])
            samples_i = dist.sample((num_samples,)).float()
            log_prob_i = dist.log_prob(samples_i)

            # Assign values to pre-allocated tensors
            samples[:, i] = samples_i
            log_prob[:, i] = log_prob_i

        return samples, torch.sum(log_prob, dim=1)

    def log_prob(self, z, context=None):
        # Clamp the input values to be non-negative integers
        z_clamped = torch.clamp(z, min=0.0).floor()

        log_prob = torch.zeros_like(z)
        for i in range(self.shape[0]):
            dist = Poisson(self.lambda_[i])
            log_prob[:, i] = dist.log_prob(z_clamped[:, i])

        # Penalize values that are not non-negative integers
        outside_support_penalty = torch.where(
            z != z.floor(),
            torch.full_like(z, -float('inf')),
            torch.zeros_like(z)
        )

        return torch.sum(log_prob, dim=1) + torch.sum(outside_support_penalty, dim=1)

    def sample(self, num_samples=1, context=None):
        samples, _ = self.forward(num_samples, context)
        return samples
    



class StudentTDistribution(BaseDistribution):
    """
    Student's t-distribution for flow-based models
    """

    def __init__(self, shape, nu=1.0, trainable=True):
        """Constructor

        Args:
          shape: Shape of the distribution (assumes each dimension is independent)
          nu: Initial degrees of freedom for the Student's t-distribution
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        self.shape = self._to_tuple(shape)
        self.n_dim = len(self.shape)
        self.d = np.prod(self.shape)

        self.nu = nn.Parameter(torch.full(self.shape, float(nu))) if trainable else torch.full(self.shape, float(nu))

    def _to_tuple(self, shape):
        if isinstance(shape, int):
            return (shape,)
        elif isinstance(shape, list):
            return tuple(shape)
        return shape

    def forward(self, num_samples=1, context=None):
        samples = torch.empty(num_samples, *self.shape)
        log_prob = torch.empty(num_samples, *self.shape)
        for i in range(self.shape[0]):
            dist = StudentT(self.nu[i])
            samples[:, i] = dist.sample((num_samples,))
            log_prob[:, i] = dist.log_prob(samples[:, i])
        return samples, torch.sum(log_prob, dim=1)

    def log_prob(self, z, context=None):
        log_prob = torch.zeros_like(z)
        for i in range(self.shape[0]):
            dist = StudentT(self.nu[i])
            log_prob[:, i] = dist.log_prob(z[:, i])
        return torch.sum(log_prob, dim=1)

    def sample(self, num_samples=1, context=None):
        samples, _ = self.forward(num_samples, context)
        return samples





class CauchyDistribution(BaseDistribution):
    """
    Cauchy distribution for flow-based models
    """

    def __init__(self, shape, x0=0.0, gamma=1.0, trainable=True):
        """Constructor

        Args:
          shape: Shape of the distribution (assumes each dimension is independent)
          x0: Initial location parameter for the Cauchy distribution
          gamma: Initial scale parameter for the Cauchy distribution
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        self.shape = self._to_tuple(shape)
        self.n_dim = len(self.shape)
        self.d = np.prod(self.shape)

        self.x0 = nn.Parameter(torch.full(self.shape, float(x0))) if trainable else torch.full(self.shape, float(x0))
        self.gamma = nn.Parameter(torch.full(self.shape, float(gamma))) if trainable else torch.full(self.shape, float(gamma))

    def _to_tuple(self, shape):
        if isinstance(shape, int):
            return (shape,)
        elif isinstance(shape, list):
            return tuple(shape)
        return shape

    def forward(self, num_samples=1, context=None):
        samples = torch.empty(num_samples, *self.shape)
        log_prob = torch.empty(num_samples, *self.shape)
        for i in range(self.shape[0]):
            dist = Cauchy(self.x0[i], self.gamma[i])
            samples[:, i] = dist.sample((num_samples,))
            log_prob[:, i] = dist.log_prob(samples[:, i])
        return samples, torch.sum(log_prob, dim=1)

    def log_prob(self, z, context=None):
        log_prob = torch.zeros_like(z)
        for i in range(self.shape[0]):
            dist = Cauchy(self.x0[i], self.gamma[i])
            log_prob[:, i] = dist.log_prob(z[:, i])
        return torch.sum(log_prob, dim=1)

    def sample(self, num_samples=1, context=None):
        samples, _ = self.forward(num_samples, context)
        return samples




class SawtoothDistribution(BaseDistribution):
    """
    Custom Sawtooth distribution for flow-based models
    """

    def __init__(self, shape, period=1.0, trainable=True):
        """Constructor

        Args:
          shape: Shape of the distribution (assumes each dimension is independent)
          period: Period of the sawtooth waveform
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        self.shape = self._to_tuple(shape)
        self.n_dim = len(self.shape)
        self.d = np.prod(self.shape)

        self.period = nn.Parameter(torch.full(self.shape, float(period))) if trainable else torch.full(self.shape, float(period))

    def _to_tuple(self, shape):
        if isinstance(shape, int):
            return (shape,)
        elif isinstance(shape, list):
            return tuple(shape)
        return shape

    def forward(self, num_samples=1, context=None):
        # Generate uniform samples and transform them into a sawtooth pattern
        uniform_samples = torch.rand(num_samples, *self.shape, device=self.period.device)
        sawtooth_samples = (uniform_samples * self.period) % 1.0

        # Calculate density of the sawtooth pattern
        sawtooth_density = 1.0 / self.period

        # Calculate log probabilities
        log_prob = torch.log(sawtooth_density) * torch.ones_like(sawtooth_samples)

        return sawtooth_samples, torch.sum(log_prob, dim=1)

    def log_prob(self, z, context=None):
        # Normalize z to be within [0, period]
        z_normalized = z % self.period

        # Compute the density of the sawtooth wave
        density = 2 * z_normalized / self.period

        # Compute log probability
        log_prob = torch.log(density)

        return torch.sum(log_prob, dim=1)

