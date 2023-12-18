# -*- coding: utf-8 -*-
"""Samplers for tasks.

This module implements samplers for various machine learning tasks, which define
distributions over input-output pairs (x, y).

Date: June 2021, Dec 2023

Authors: Alessandro Ingrosso <aingrosso@ictp.com>
         Sebastian Goldt <goldt.sebastian@gmail.com>
         Eszter Szekely <szeszter94@gmail.com>
"""

from abc import ABCMeta, abstractmethod
import torch

import inputs
import numpy as np
import numpy.random as rnd


class Task(metaclass=ABCMeta):
    """
    Abstract class for all the tasks used in these experiments.
    """

    @abstractmethod
    def input_dim(self):
        """
        Dimension of the vectors for 1D models, width of inputs for 2D models.
        """

    @abstractmethod
    def get_dataset(
        self, train=True, P=None, dtype=torch.float, device=torch.device("cpu"),
    ):
        """
        Retrieve stored dataset.

        Parameters:
        -----------
        train : True for training data set, else a test data set is loaded.
        P : number of samples
        """

    @abstractmethod
    def sample(self, P):
        """
        Samples P samples from the task.

        Returns:
        --------

        xs : (P, D)
             P inputs in D dimensions
        ys : (P)
             P labels
        """


class Mixture(Task):
    """
    Mixture of distributions, one for each label.
    """

    def __init__(self, distributions, sample_50_50=False):
        """
        Parameters:
        -----------

        distributions: array
            a set of inputs, each of which will correspond to one label.
        """
        super().__init__()

        self.distributions = distributions
        self.num_classes = len(distributions)

        self.D = self.distributions[0].input_dim
        self.sample_50_50=sample_50_50

    def input_dim(self):
        return self.D

    def get_dataset(
        self, train=True, P=None, dtype=torch.float, device=torch.device("cpu"),
    ):
        num_p = P // self.num_classes
        dataset_desc = "training" if train else "testing"
        print(f"will generate {dataset_desc} set with {num_p} patterns per class")
        X = torch.empty((0, self.D), dtype=dtype, device=device)
        y = torch.empty(0, dtype=dtype, device=device)
        for p, distribution in enumerate(self.distributions):
            Xtemp = distribution.get_dataset(
                train=train, P=num_p, dtype=dtype, device=device
            )
            X = torch.cat([X, Xtemp])
            y = torch.cat([y, p * torch.ones(len(Xtemp), dtype=dtype, device=device)])

        return X, y

    def sample(self, P=1):
        if not self.sample_50_50:
            ys = torch.randint(self.num_classes, (P,))
        else:
            ys = torch.zeros(P)
            ys=torch.cat([torch.zeros(int(np.floor(P/2.))),torch.ones(int(np.ceil(P/2.)))])[torch.randperm(P)]

        xs = torch.zeros(P, self.D)

        for m in range(self.num_classes):
            num_samples = torch.sum(ys == m).item()
            xs[ys == m] = self.distributions[m].sample(num_samples)

        return xs, ys

    def __str__(self):
        dist_names = [str(dist) for dist in self.distributions]
        name = "_".join(dist_names)
        return name


def build_nlgp_mixture(
    input_names, xis, D, torus, gain, dim=1, xi_pow_pi=True, perturbation=1e-3
):
    """
    Constructs a mixture of distributions (NLGP / GP) with the given xis and gain.
    """
    distributions = [None] * len(input_names)

    for idx, input_name in enumerate(input_names):
        # create the covariance for the given correlation length
        xi = xis[idx]

        covariance = inputs.trans_inv_var(
            D,
            torus=torus,
            p=2,
            xi=xi,
            perturbation=perturbation,
            dim=dim,
            xi_pow_pi=xi_pow_pi,
        )
        # create the non-linear GP
        nlgp = inputs.NLGP("erf", covariance, gain=gain)
        if input_name == "gp":
            # create a Gaussian process with the same covariance
            distributions[idx] = inputs.GaussianProcess(nlgp.covariance())
        elif input_name == "nlgp":
            distributions[idx] = nlgp
        #elif input_name == "gaussian":
        #    ### this is not finished 
        #    #distributions[idx] = inputs.gaussian(
        #    rv = stats.multivariate_normal(mean=None)#, cov=nlgp.covariance)
        #    distributions[idx] += rv.rvs(size=num_samples)

        else:
            raise ValueError("Did not recognise input name (gp | nlgp)")

    return distributions


def build_whitened_mixture(
    input_names, D,  gain, dim=1, spike=[], distr="Rademacher", 
):
    """
    Constructs a mixture of distributions (spiked / non-spiked) with the given  gain.
    """
    distributions = [None] * len(input_names)
    if len(spike)==0:
        spike=np.sign(rnd.randn(D))

    for idx, input_name in enumerate(input_names):
        # create the non-linear GP
        nlgp = inputs.WhitenedModel(spike, gain=gain,distr=distr)
        if input_name == "gp":
            # create a reference class 
            distributions[idx] = inputs.WhitenedModel_RefClass(D) 
        elif input_name == "nlgp":
            distributions[idx] = nlgp
        else:
            raise ValueError("Did not recognise input name (gp | nlgp)")

    return distributions


def build_whitened_mixture_multi_spike(
    input_names, D,  gain, dim=1, num_spike=1,spikes=[], distr="Rademacher",
):
    """
    Constructs a mixture of distributions (spiked / non-spiked) with the given  gain.
    """
    distributions = [None] * len(input_names)
    if len(spikes)==0:
        spikes=[]
        for i in range(num_spike):
            spikes.append(np.sign(rnd.randn(D)))

    for idx, input_name in enumerate(input_names):
        # create the non-linear GP
        if not(num_spike==len(spikes)):
            print("problem with number of spikes:",num_spike,len(spikes))
        nlgp = inputs.WhitenedModelMultiSpike(spikes, gain=gain, distr=distr)
        if input_name == "gp":
            # create a reference class with the same whitening matrix
            distributions[idx] = inputs.WhitenedModel_RefClass(D) 
        elif input_name == "nlgp":
            distributions[idx] = nlgp
        else:
            raise ValueError("Did not recognise input name (gp | nlgp)")

    return distributions


def build_spiked_gaussian(
    input_names, D, gain, covariance=None, dim=1, spike=[]
):
    """
    Constructs a mixture of distributions (spiked / non-spiked) with the given  gain.
    """
    distributions = [None] * len(input_names)
    if covariance is None:
        covariance = torch.eye(D)
    if len(spike)==0:
        spike = np.sign( rnd.randn(D))

    for idx, input_name in enumerate(input_names):
        if input_name == "gp":
            # create a reference class with the same whitening matrix
            distributions[idx] = inputs.GaussianProcess(covariance, mean=None)
        elif input_name == "spiked":
            distributions[idx] = inputs.SpikedGaussian(spike, covariance, gain=gain)  
        else:
            raise ValueError("Did not recognise input name (spiked | gauss)")

    return distributions

def build_spiked_gaussian_multi_spike(
    input_names, D, gain, covariance=None, dim=1, num_spike=1, spikes=[]
):
    """
    Constructs a mixture of distributions (spiked / non-spiked) with the given  gain.
    """
    distributions = [None] * len(input_names)
    if covariance is None:
        covariance = torch.eye(D)
    if len(spikes)==0:
        spikes=[]
        for i in range(num_spike):
            spikes.append(np.sign(rnd.randn(D)))


    for idx, input_name in enumerate(input_names):
        if input_name == "gp":
            # create a reference class with the same whitening matrix
            distributions[idx] = inputs.GaussianProcess(covariance, mean=None)
        elif input_name == "spiked":
            distributions[idx] = inputs.SpikedGaussianMultiSpike(spikes, covariance, gain=gain)
        else:
            raise ValueError("Did not recognise input name (spiked | gauss)")

    return distributions




def build_mixture(D, params):
    """
    Factory method to create mixtures of distributions.

    Parameters:
    -----------
    D : input dimension
    param:
        params ...
    distribution : gp | ising | phi4
    """
    num_distributions = len(params)

    distributions = [None] * num_distributions
    for m, par in enumerate(params):
        distribution = par["distribution"]
        if distribution == "gp":
            dim = par["dim"]
            torus = par["torus"]
            exponent = par["exponent"]
            perturbation = par["perturbation"]
            xi = par["xi"]

            covariance = inputs.trans_inv_var(
                D, torus=torus, p=exponent, xi=xi, perturbation=perturbation, dim=dim
            )

            distributions[m] = inputs.GaussianProcess(covariance)
        elif distribution == "ising":
            dim = par["dim"]
            T = par["T"]
            num_steps_eq = par["num_steps_eq"]
            sampling_rate = par["sampling_rate"]
            load_dir = par["load_dir"]
            distributions[m] = inputs.Ising(
                dim=dim,
                N=D,
                T=T,
                num_steps_eq=num_steps_eq,
                sampling_rate=sampling_rate,
                load_dir=load_dir,
            )
        elif distribution == "phi4":
            dim = (par["dim"],)
            lambd = par["lambd"]
            musq = par["musq"]
            zscore = par["zscore"]
            normalize = par["normalize"]
            sampling_rate = par["sampling_rate"]
            buffer_size = par["buffer_size"]
            load_dir = par["load_dir"]
            distributions[m] = inputs.Phi4(
                dim=dim,
                D=D,
                lambd=lambd,
                musq=musq,
                zscore=zscore,
                normalize=normalize,
                sampling_rate=sampling_rate,
                buffer_size=buffer_size,
                load_dir=load_dir,
            )
        else:
            raise ValueError("What the fuck are you talking about?!")

    mixture = Mixture(distributions)

    return mixture
