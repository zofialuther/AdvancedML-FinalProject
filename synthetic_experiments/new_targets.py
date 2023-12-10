# Our implementations for our new targets

from normflows.distributions import Target
import torch
import torch.nn as nn
import numpy as np

# Spiral, Swiss Roll, Klein bottle, Möbius strip

class Spiral(Target):
    """
    Spiral two-dimensional distribution
    """

    def __init__(self, n_arms=2, spread=0.1):
        """
        Args:
          n_arms: Number of spiral arms
          spread: Spread of points around the spiral arms
        """
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0
        self.n_arms = n_arms
        self.spread = spread

    def log_prob(self, z):
        """
        Approximate log probability for points in the spiral distribution.
        
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        # Convert to polar coordinates
        r = torch.sqrt(z[:, 0]**2 + z[:, 1]**2)
        theta = torch.atan2(z[:, 1], z[:, 0])

        # Adjust theta to lie within the range [0, 2*pi*n_arms]
        theta = theta % (2 * np.pi * self.n_arms)

        # Spiral equation: r = a + b*theta
        # We choose 'a' such that the spiral starts near the origin, and 'b' controls the spread
        a, b = 0.0, 0.1
        spiral_r = a + b * theta

        # Calculate the distance of each point from the ideal spiral
        distance_from_spiral = torch.abs(r - spiral_r)

        # Compute log probability as negative square of distance from the spiral
        log_prob = -0.5 * (distance_from_spiral / self.spread) ** 2
        return log_prob



class SwissRoll(Target):
    """
    Swiss Roll distribution
    """

    def __init__(self, spread=0.1):
        """
        Args:
          spread: Spread of points around the Swiss Roll
        """
        super().__init__()
        self.n_dims = 3  # Swiss Roll is typically represented in 3D
        self.max_log_prob = 0.0
        self.spread = spread

    def log_prob(self, z):
        """
        Approximate log probability for points in the Swiss Roll distribution.
        
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        # Add a small positive value inside the square root to ensure non-negative input
        z_unwrapped = torch.sqrt(z[:, 0]**2 + z[:, 2]**2 + 1e-8)

        # Ensure spread is not too small
        spread = max(self.spread, 1e-8)

        # Compute the distance of each point from the theoretical Swiss Roll surface
        distance_from_surface = torch.abs(z[:, 1] - z_unwrapped % 1)

        # Compute log probability as negative square of distance from the surface
        log_prob = -0.5 * (distance_from_surface / spread) ** 2

        # Check for NaNs and set them to a large negative number
        log_prob = torch.where(torch.isnan(log_prob), torch.tensor(float('-inf')).to(log_prob.device), log_prob)

        return log_prob# Unwrap the Swiss Roll to a flat 2D space


class KleinBottle(Target):
    """
    Klein Bottle distribution
    """

    def __init__(self, spread=0.1):
        """
        Args:
          spread: Spread of points around the Klein Bottle
        """
        super().__init__()
        self.n_dims = 3
        self.max_log_prob = 0.0
        self.spread = spread

    def log_prob(self, z):
        """
        Approximate log probability for points in the Klein Bottle distribution.
        
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        # Simplified 3D representation of a Klein bottle
        # This representation is a rough approximation and may not perfectly capture the Klein bottle
        u = torch.atan2(z[:, 1], z[:, 0])
        v = torch.sqrt(z[:, 0]**2 + z[:, 1]**2) - 2

        # Klein bottle equations in 3D (approximate)
        x = (2 + torch.cos(u / 2) * torch.sin(v) - torch.sin(u / 2) * torch.sin(2 * v)) * torch.cos(u)
        y = (2 + torch.cos(u / 2) * torch.sin(v) - torch.sin(u / 2) * torch.sin(2 * v)) * torch.sin(u)
        z_approx = torch.sin(u / 2) * torch.sin(v) + torch.cos(u / 2) * torch.sin(2 * v)

        # Compute the distance of each point from the theoretical Klein bottle surface
        distance_from_surface = torch.sqrt((z[:, 0] - x)**2 + (z[:, 1] - y)**2 + (z[:, 2] - z_approx)**2)

        # Compute log probability as negative square of distance from the surface
        log_prob = -0.5 * (distance_from_surface / self.spread) ** 2
        return log_prob



class MobiusStrip(Target):
    """
    Möbius Strip distribution
    """

    def __init__(self, spread=0.1):
        """
        Args:
          spread: Spread of points around the Möbius Strip
        """
        super().__init__()
        self.n_dims = 3
        self.max_log_prob = 0.0
        self.spread = spread

    def log_prob(self, z):
        """
        Approximate log probability for points in the Möbius Strip distribution.
        
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        # Simplified 3D representation of a Möbius strip
        u = torch.atan2(z[:, 1], z[:, 0])
        v = torch.sqrt(z[:, 0]**2 + z[:, 1]**2) - 1

        # Möbius strip equations in 3D
        x = torch.cos(u) * (1 + v / 2 * torch.cos(u / 2))
        y = torch.sin(u) * (1 + v / 2 * torch.cos(u / 2))
        z_approx = v / 2 * torch.sin(u / 2)

        # Compute the distance of each point from the theoretical Möbius strip surface
        distance_from_surface = torch.sqrt((z[:, 0] - x)**2 + (z[:, 1] - y)**2 + (z[:, 2] - z_approx)**2)

        # Compute log probability as negative square of distance from the surface
        log_prob = -0.5 * (distance_from_surface / self.spread) ** 2
        return log_prob
