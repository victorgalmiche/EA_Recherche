"""Approche constructive pour l'analyse de pire cas de SGD"""

import numpy as np
import sympy as sm
import cvxpy as cp
import matplotlib.pyplot as plt

# On fixe le nombre n de fonctions 
n = 3


################ Utilisation de Sympy pour déterminer les matrices du SDP #########
L = sm.Symbol('L')
mu = sm.Symbol('mu')
gamma = sm.Symbol('gamma')

R2 = sm.Symbol('R2')
v2 = sm.Symbol('v2')

x0 = sm.Symbol('x0')
xs = 0
f = [sm.Symbol(f'f_{i}') for i in range(n)]
fs = [sm.Symbol(f'fs_{i}') for i in range(n)]
g = [sm.Symbol(f'g_{i}') for i in range(n)]
gs = [sm.Symbol(f'gs_{i}') for i in range(n-1)]
gs.append(-sum(gs[i] for i in range(n-1))) # Dernier gradient est l'opposé de la somme des autres

objective = sum([(x0-xs-gamma*g[i])**2 for i in range(n)])/n # Fonction objectif

# Contraintes <= 0
constraints1 = [f[i]- fs[i] + g[i]*(xs-x0) + (g[i]-gs[i])**2/(2*L) + mu/(2*(1-mu/L))*(x0-xs+gs[i]/L-g[i]/L)**2 for i in range(n)]
constraints2 = [fs[i]- f[i] + gs[i]*(x0-xs) + (g[i]-gs[i])**2/(2*L) + mu/(2*(1-mu/L))*(x0-xs+gs[i]/L-g[i]/L)**2 for i in range(n)]
initial_constraint = (x0-xs)**2 - R2
variance_constraint = sum([g[i]**2 for i in range(n)])/n - v2

# Variables dans la matrice de Gram 
variables = [x0] + g + gs[:-1]

A_num = sm.lambdify(gamma, sm.simplify(sm.hessian(objective, variables)/2))
A_1 = [sm.lambdify([mu, L], sm.simplify(sm.hessian(constraints1[i], variables)/2)) for i in range(n)]
A_2 = [sm.lambdify([mu, L], sm.simplify(sm.hessian(constraints2[i], variables)/2)) for i in range(n)]
A_0 = sm.lambdify([], sm.simplify(sm.hessian(initial_constraint, variables)/2))
A_var = sm.lambdify([], sm.simplify(sm.hessian(variance_constraint, variables)/2))


dual = True
sauvegarde_figures = False

# On fixe les paramètres
mu, L = .1, 1
R2 = 1
v2 = 1

################# PROBLEME PRIMAL ###################
if not(dual):
    problem_primal_values = []
    gamma_values = np.linspace(0, .3)

    for gamma in gamma_values:
        
        Anum = A_num(gamma)
        A1 = [A_1[i](mu, L) for i in range(n)]
        A2 = [A_2[i](mu, L) for i in range(n)]
        A0 = A_0()
        Avar = A_var()

        f_0 = cp.Variable(n) # Valeurs des f_i en x_0
        f_star = cp.Variable(n) # Valeurs des f_i en x_\star
        G = cp.Variable((2*n,2*n)) # Matrice de Gram

        objective_primal = cp.Maximize(cp.trace(Anum @ G)) 
        constraints_primal = [G>>0, cp.trace(A0 @ G)<=R2, cp.trace(Avar @ G)<=v2]
        for i in range(n): # Contraintes d'interpolation
            constraints_primal.append(f_0[i]-f_star[i]+cp.trace(A1[i] @ G)<=0)
            constraints_primal.append(f_star[i]-f_0[i]+cp.trace(A2[i] @ G)<=0)

        problem_primal = cp.Problem(objective_primal, constraints_primal)
        problem_primal.solve()

        #Sauvegarde de la valeur calulée
        problem_primal_values.append(problem_primal.value)

    # Tracé courbes
    plt.title(r'Maximum pour $n=3, R=1, v=1, \mu=0.1, L=1$ en fonction de $\gamma$')
    plt.xlabel(r'Pas de descente $\gamma$')
    plt.ylabel(r'$\sup \mathbb{E}\left[\|x_1-x_\star\|^2\right]$')
    plt.plot(gamma_values, problem_primal_values, label="Valeur du pire cas")
    plt.legend(loc='best')

    if sauvegarde_figures:
        plt.savefig(r'C:\Users\victo\Desktop\POLYTECHNIQUE\3A\EA MAP511\Figures\SGD_primal.png')
    else:
        plt.show()


################ PROBLEME DUAL ################
if dual:
    # Cette fois ci on regarde la dépendance en L
    L_values = np.linspace(2, 10)

    problem_dual_values = []
    theoretical_values = []
    for L in L_values:

        # Calcul des matrices
        gamma=1/L
        Anum = A_num(gamma)
        A1 = [A_1[i](mu, L) for i in range(n)]
        A2 = [A_2[i](mu, L) for i in range(n)]
        A0 = A_0()
        Avar = A_var()

        # Résolution du dual
        lambda_1 = cp.Variable(n)
        lambda_2 = cp.Variable(n)
        rho = (1-mu/L)**2 # On force la valeur de rho
        tau = cp.Variable()

        # Fonction obectif
        objective_dual = cp.Minimize(tau)

        # Contraintes
        S = Anum - sum([lambda_1[i]*A1[i] + lambda_2[i]*A2[i] for i in range(n)]) - rho*A0 - tau*Avar
        constraints_dual = [S << 0, tau >= 0]
        for i in range(n):
            constraints_dual.append(lambda_1[i] == lambda_2[i])
            constraints_dual.append(lambda_1[i] >= 0)
            constraints_dual.append(lambda_2[i] >= 0)

        #Résolution
        problem_dual = cp.Problem(objective_dual, constraints_dual)
        problem_dual.solve()

        # Comparaison avec 2/L^2
        problem_dual_values.append(problem_dual.value)
        theoretical_values.append(2/(L**2))

    #Tracé des courbes
    plt.title(r'Résolution du problème dual en fixant $\rho = \left(1-\mu/L\right)^2$')
    plt.xlabel(r'Constante de Lipschitz L')
    plt.ylabel(r'$\tau_\star$')
    plt.plot(L_values, problem_dual_values, label=r'$\inf \tau$')
    plt.plot(L_values, theoretical_values, 'r+', label=r'$2/L^2$')
    plt.legend(loc='best')

    if sauvegarde_figures:
        plt.savefig(r'C:\Users\victo\Desktop\POLYTECHNIQUE\3A\EA MAP511\Figures\SGD_dual.png')
    else :
        plt.show()
        