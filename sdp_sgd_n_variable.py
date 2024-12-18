"""Comme sdp_sgd mais avec n variable : étude de la dépendance en le nombre de fonctions"""

import numpy as np
import sympy as sm
import cvxpy as cp
import matplotlib.pyplot as plt

problem_primal_values = [] 
n_values = np.arange(2, 21) # n étudiés

for n in n_values:

    # On détermine pour chaque n le SDP 
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

    objective = sum([(x0-xs-gamma*g[i])**2 for i in range(n)])/n

    # Contraintes <= 0
    constraints1 = [f[i]- fs[i] + g[i]*(xs-x0) + (g[i]-gs[i])**2/(2*L) + mu/(2*(1-mu/L))*(x0-xs+gs[i]/L-g[i]/L)**2 for i in range(n)]
    constraints2 = [fs[i]- f[i] + gs[i]*(x0-xs) + (g[i]-gs[i])**2/(2*L) + mu/(2*(1-mu/L))*(x0-xs+gs[i]/L-g[i]/L)**2 for i in range(n)]
    initial_constraint = (x0-xs)**2 - R2
    variance_constraint = sum([g[i]**2 for i in range(n)])/n - v2

    variables = [x0] + g + gs[:-1]

    A_num = sm.lambdify(gamma, sm.simplify(sm.hessian(objective, variables)/2))
    A_1 = [sm.lambdify([mu, L], sm.simplify(sm.hessian(constraints1[i], variables)/2)) for i in range(n)]
    A_2 = [sm.lambdify([mu, L], sm.simplify(sm.hessian(constraints2[i], variables)/2)) for i in range(n)]
    A_0 = sm.lambdify([], sm.simplify(sm.hessian(initial_constraint, variables)/2))
    A_var = sm.lambdify([], sm.simplify(sm.hessian(variance_constraint, variables)/2))

    # On fixe les paramètres
    mu, L = .1, 1
    R2 = 1
    v2 = 1
    gamma = .1
    
    Anum = A_num(gamma)
    A1 = [A_1[i](mu, L) for i in range(n)]
    A2 = [A_2[i](mu, L) for i in range(n)]
    A0 = A_0()
    Avar = A_var()



    # Résolution du problème 
    f_0 = cp.Variable(n)
    f_star = cp.Variable(n)
    G = cp.Variable((2*n,2*n))

    objective_primal = cp.Maximize(cp.trace(Anum @ G))
    constraints_primal = [G>>0, cp.trace(A0 @ G)<=R2, cp.trace(Avar @ G)<=v2]
    for i in range(n):
        constraints_primal.append(f_0[i]-f_star[i]+cp.trace(A1[i] @ G)<=0)
        constraints_primal.append(f_star[i]-f_0[i]+cp.trace(A2[i] @ G)<=0)

    problem_primal = cp.Problem(objective_primal, constraints_primal)
    problem_primal.solve()
    problem_primal_values.append(problem_primal.value)




# Tracé figure
plt.title(r'Résolution du SDP pour $R=1, v=1,\mu=0.1, L=1, \gamma=0.1$ en fonction de $n$')
plt.xlabel(r'Nombre $n$ de fonctions')
plt.ylabel(r'$\sup \mathbb{E}\left[\|x_1-x_\star\|^2\right]$')
plt.ylim((0.9, 1.2))
plt.plot(n_values, problem_primal_values, 'r+', label='Valeur du pire cas')
plt.legend(loc='best')
plt.savefig(r'C:\Users\victo\Desktop\POLYTECHNIQUE\3A\EA MAP511\Figures\SGD_primal_f(n).png')
