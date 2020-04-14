# -*- coding: utf-8 -*-

import numpy as np
import numpy.random


class ParticleSwarm:
    def __init__(self, cost_func, dim, size=50, chi=0.72984, phi_p=2.05, phi_g=2.05):
        '''
        phi_p = C1
        phi_g = C2

        X = POS
        V = VELOCITY
        P = PARTICLE
        S = FITNESS
        g = BEST GLOBAL PARTICLE
        best_score = BEST FITNESS VALUE        
        '''

        self.cost_func = cost_func
        self.dim = dim

        self.size = size
        self.chi = chi
        self.phi_p = phi_p
        self.phi_g = phi_g

        # INITIALIZE POS DAN VELOCITY RANDOM
        self.X = np.random.uniform(size=(self.size, self.dim))
        self.V = np.random.uniform(size=(self.size, self.dim))

        self.P = self.X.copy()
        self.S = self.cost_func(self.X)
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()

    def update(self):
        # Velocities update
        R_p = np.random.uniform(size=(self.size, self.dim))
        R_g = np.random.uniform(size=(self.size, self.dim))

        self.V = self.chi * (self.V \
                + self.phi_p * R_p * (self.P - self.X) \
                + self.phi_g * R_g * (self.g - self.X))

        # Positions update
        self.X = self.X + self.V

        # Best scores
        scores = self.cost_func(self.X)

        better_scores_idx = scores < self.S
        self.P[better_scores_idx] = self.X[better_scores_idx]
        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
