"""Approche constructive pour l'analyse de pire cas de SAGA"""
import numpy as np
import sympy as sm
import cvxpy as cp
import matplotlib.pyplot as plt

# On fixe n le nombre de fonctions
n = 3

########### Utilisation de Sympy pour déterminer les variables du SDP #########
L = sm.Symbol('L')
mu = sm.Symbol('mu')
gamma = sm.Symbol('gamma')
c = sm.Symbol('c')

x0 = sm.Symbol('x0') 
phi = [sm.Symbol(f'phi_{i}') for i in range(n)]
xs = 0

f = [sm.Symbol(f'f_{i}') for i in range(n)] # Valeurs en x_0
fphi = [sm.Symbol(f'fphi_{i}') for i in range(n)] # Valeurs de f_i en phi_i
fs = [sm.Symbol(f'fs_{i}') for i in range(n)] # Valeurs en x_star

g = [sm.Symbol(f'g_{i}') for i in range(n)] # Gradients en x_0
gphi = [sm.Symbol(f'gphi_{i}') for i in range(n)] # Gradients de f_i en phi_i
gs = [sm.Symbol(f'gs_{i}') for i in range(n-1)] # Gradients en x_star
gs.append(-sum(gs[i] for i in range(n-1))) # Dernier gradient est l'opposé de la somme des autres

# Fonction objectif = fonction de lyapunov en k=1
lyapunov = sum(f)/(n**2) + (1-1/n)*sum(fphi)/n - sum(fs)/n - (1-1/n)*sum([gs[i]*(phi[i]-xs) for i in range(n)])/n + c*sum([(x0-gamma*(g[j]-gphi[j]+sum(gphi)/n)-xs)**2 for j in range(n)])/n 

# Interpolation entre x_0 et x_star
constraints1 = [f[i]- fs[i] + g[i]*(xs-x0) + (g[i]-gs[i])**2/(2*L) + mu/(2*(1-mu/L))*(x0-xs+gs[i]/L-g[i]/L)**2 for i in range(n)]
constraints2 = [fs[i]- f[i] + gs[i]*(x0-xs) + (g[i]-gs[i])**2/(2*L) + mu/(2*(1-mu/L))*(x0-xs+gs[i]/L-g[i]/L)**2 for i in range(n)]

# Interpolation entre x_0 et phi
constraints3 = [f[i]- fphi[i] + g[i]*(phi[i]-x0) + (g[i]-gphi[i])**2/(2*L) + mu/(2*(1-mu/L))*(x0-phi[i]+gphi[i]/L-g[i]/L)**2 for i in range(n)]
constraints4 = [fphi[i]- f[i] + gphi[i]*(x0-phi[i]) + (g[i]-gphi[i])**2/(2*L) + mu/(2*(1-mu/L))*(x0-phi[i]+gphi[i]/L-g[i]/L)**2 for i in range(n)]

# Interpolation entre phi et x_star
constraints5 = [fphi[i]- fs[i] + gphi[i]*(xs-phi[i]) + (gphi[i]-gs[i])**2/(2*L) + mu/(2*(1-mu/L))*(phi[i]-xs+gs[i]/L-gphi[i]/L)**2 for i in range(n)]
constraints6 = [fs[i]- fphi[i] + gs[i]*(phi[i]-xs) + (gphi[i]-gs[i])**2/(2*L) + mu/(2*(1-mu/L))*(phi[i]-xs+gs[i]/L-gphi[i]/L)**2 for i in range(n)]

# Contrainte sur la valeur initiale du la focntion de Lyapunov
lyapunov0 = sum(fphi)/n - sum(fs)/n - sum([gs[i]*(phi[i]-xs) for i in range(n)])/n + c*(x0-xs)**2


variables = [x0] + phi + g + gphi + gs[:-1] # Variables de la matrice de Gram 


A_num = sm.lambdify([c, gamma], sm.simplify(sm.hessian(lyapunov, variables)/2))
A_1 = [sm.lambdify([mu, L], sm.simplify(sm.hessian(constraints1[i], variables)/2)) for i in range(n)]
A_2 = [sm.lambdify([mu, L], sm.simplify(sm.hessian(constraints2[i], variables)/2)) for i in range(n)]
A_3 = [sm.lambdify([mu, L], sm.simplify(sm.hessian(constraints3[i], variables)/2)) for i in range(n)]
A_4 = [sm.lambdify([mu, L], sm.simplify(sm.hessian(constraints4[i], variables)/2)) for i in range(n)]
A_5 = [sm.lambdify([mu, L], sm.simplify(sm.hessian(constraints5[i], variables)/2)) for i in range(n)]
A_6 = [sm.lambdify([mu, L], sm.simplify(sm.hessian(constraints6[i], variables)/2)) for i in range(n)]
A_0 = sm.lambdify(c, sm.simplify(sm.hessian(lyapunov0, variables)/2))



################## PROBLEME PRIMAL ################
L = 1
mu_values = np.linspace(.1, .7)
# c_values = np.linspace(0, 3)
problem_primal_values = []
theoretical_values = []


for mu in mu_values :    
    gamma = 1/(2*(mu*n + L)) 
    T = 1
    c = 1/(2*n*gamma*(1-mu*gamma))
    theoretical_values.append(1-gamma*mu)

    Anum = A_num(c, gamma)
    A1 = [A_1[i](mu, L) for i in range(n)]
    A2 = [A_2[i](mu, L) for i in range(n)]
    A3 = [A_3[i](mu, L) for i in range(n)]
    A4 = [A_4[i](mu, L) for i in range(n)]
    A5 = [A_5[i](mu, L) for i in range(n)]
    A6 = [A_6[i](mu, L) for i in range(n)]
    A0 = A_0(c)

    f_0 = cp.Variable(n)
    f_star = cp.Variable(n)
    f_phi = cp.Variable(n)
    G = cp.Variable((4*n,4*n))

    objective_primal = cp.Maximize(sum(f_0)/(n**2) + (1-1/n)*sum(f_phi)/n - sum(f_star)/n + cp.trace(Anum @ G))
    constraints_primal = [G>>0, cp.trace(A0 @ G) + sum(f_phi)/n - sum(f_star)/n <=T]
    for i in range(n):
        constraints_primal.append(f_0[i]-f_star[i]+cp.trace(A1[i] @ G)<=0)
        constraints_primal.append(f_star[i]-f_0[i]+cp.trace(A2[i] @ G)<=0)
        constraints_primal.append(f_0[i]-f_phi[i]+cp.trace(A3[i] @ G)<=0)
        constraints_primal.append(f_phi[i]-f_0[i]+cp.trace(A4[i] @ G)<=0)
        constraints_primal.append(f_phi[i]-f_star[i]+cp.trace(A5[i] @ G)<=0)
        constraints_primal.append(f_star[i]-f_phi[i]+cp.trace(A6[i] @ G)<=0)

    problem_primal = cp.Problem(objective_primal, constraints_primal)
    problem_primal.solve()
    problem_primal_values.append(problem_primal.value)


# plt.title(r"Valeur du problème pour $n=3, \mu=0.1, L=1, T=1, \gamma=1/(2(\mu n +L))$")
# plt.plot(c_values, problem_primal_values, label='Valeur du problème')
# plt.xlabel(r'$c$')
# plt.ylabel(r'Valeur du problème = $\max(T^1)$')
# plt.savefig(r'C:\Users\victo\Desktop\POLYTECHNIQUE\3A\EA MAP511\Figures\saga_primal.png')

# plt.title(r"Valeur de la constante $c$ mnimisant $\max T^1$ en fonction de $\mu$")
# plt.plot(mu_values, c_min, label=r'c_\star')
# plt.plot(mu_values, )
# plt.xlabel(r'$\mu$')
# plt.ylabel(r'c_\star')
# plt.savefig(r'C:\Users\victo\Desktop\POLYTECHNIQUE\3A\EA MAP511\Figures\saga_cmin.png')

erreur = (np.array(theoretical_values)-np.array(problem_primal_values))**2
print(np.mean(erreur))
plt.title(r'Valeur du problème pour $L=1, T=1, n=3$ en fonction de $\mu$')
plt.plot(mu_values, problem_primal_values, label=r'$\max \mathbb{E}\left[T^1\right]$')
plt.plot(mu_values, theoretical_values, '+', label=r'Borne théorique : $1-\gamma\mu$')
plt.xlabel(r'$\mu$')
plt.ylabel(r'Valeur du problème = $\max \mathbb{E}\left[T^1\right]$')
plt.legend(loc='best')
plt.savefig(r'C:\Users\victo\Desktop\POLYTECHNIQUE\3A\EA MAP511\Figures\saga_primal_f(mu).png')

