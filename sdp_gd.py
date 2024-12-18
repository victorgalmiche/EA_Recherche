import numpy as np
import cvxpy as cp
import sympy as sm
import matplotlib.pyplot as plt

dual = True

L = sm.Symbol('L')
mu = sm.Symbol('mu')
gamma = sm.Symbol('gamma')

R2 = sm.Symbol('R2')

x0 = sm.Symbol('x0')
xs = 0
f0 = sm.Symbol('f0')
fs = sm.Symbol('fs')
g0 = sm.Symbol('g0')
gs = 0

objective = (x0 - gamma*g0 - xs)**2
constraint_1 = f0 - fs + g0*(xs-x0) + (g0)**2/(2*L) + (x0-xs-g0/L)**2*mu/(2*(1-mu/L))
constraint_2 = fs - f0 + (g0)**2/(2*L) + (x0-xs-g0/L)**2*mu/(2*(1-mu/L))
constraint_0 = (x0-xs)**2 - R2

A = sm.lambdify(gamma, sm.simplify(sm.hessian(objective, (x0, g0))/2))
A_1 = sm.lambdify([mu, L], sm.simplify(sm.hessian(constraint_1, (x0, g0))/2))
A_2 = sm.lambdify([mu, L], sm.simplify(sm.hessian(constraint_2, (x0, g0))/2))
A_0 = sm.lambdify([], sm.simplify(sm.hessian(constraint_0, (x0, g0))/2))

mu, L = .1, 1
R2 = 1
gamma_values = np.linspace(0, 2.1)


################# PROBLEME PRIMAL ###################
if not(dual):
    problem_primal_values = []
    for gamma in gamma_values:
        Anum = A(gamma)
        Adenom = A_0()
        A1 = A_1(mu, L)
        A2 = A_2(mu, L)

        f_0 = cp.Variable()
        f_star = cp.Variable()
        G = cp.Variable((2,2))

        objective_primal = cp.Maximize(cp.trace(Anum @ G))
        constraints_primal = [G>>0, f_0-f_star+cp.trace(A1 @ G)<=0, f_star-f_0+cp.trace(A2 @ G)<=0, cp.trace(Adenom @ G)<=R2]

        problem_primal = cp.Problem(objective_primal, constraints_primal)
        problem_primal.solve()
        problem_primal_values.append(problem_primal.value)


    theoretical_values = np.maximum((1-gamma_values*mu)**2, (1-gamma_values*L)**2)
    errors = (problem_primal_values-theoretical_values)**2

    print(np.mean(errors))


    # plt.title(r'Valeur du SDP pour $R=1, \mu=0.1, L=1$ en fonction de $\gamma$')
    # plt.plot(gamma_values, problem_primal_values, label=r'$\sup \|x_1 - x_\star\|^2$')
    # plt.plot(gamma_values, theoretical_values, '+', label=r'Borne théorique : $R^2\text{max}((1-\mu\gamma)^2, (1-L\gamma)^2)$')
    # plt.xlabel(r'$\gamma$')
    # plt.ylabel(r'Valeur du pire cas')
    # plt.legend(loc='best')
    # # plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=10)
    # plt.tight_layout()
    # plt.savefig(r'C:\Users\victo\Desktop\POLYTECHNIQUE\3A\EA MAP511\Figures\GD_primal.png')




#################### PROBLEME DUAL #######################


if dual : 
    problem_dual_values = []
    for gamma in gamma_values:
        Anum = A(gamma)
        Adenom = A_0()
        A1 = A_1(mu, L)
        A2 = A_2(mu, L)


        lambda_1 = cp.Variable()
        lambda_2 = cp.Variable()
        tau = cp.Variable()

        S = Anum - tau*Adenom - lambda_1*A1 - lambda_2*A2 

        objective_dual = cp.Minimize(tau)
        constraints_dual = [S << 0, lambda_1 == lambda_2, lambda_1 >= 0, lambda_2 >= 0, tau >= 0]
        
        problem_dual = cp.Problem(objective_dual, constraints_dual)
        problem_dual.solve(verbose=True)
        # problem_dual.solve()
        problem_dual_values.append(problem_dual.value)



    plt.title(r'Valeur du problème dual pour $\mu=0.1, L=1$ en fonction de $\gamma$')
    plt.xlabel(r"Pas de descente $\gamma$ de l'algorithme")
    plt.ylabel(r'$\tau_\star$')
    plt.plot(gamma_values, problem_dual_values, label=r'$\inf \tau$')
    plt.legend(loc='best')
    plt.savefig(r'C:\Users\victo\Desktop\POLYTECHNIQUE\3A\EA MAP511\Figures\GD_dual.png')
