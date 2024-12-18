"""Tracé fonction fortement convexe et lisse"""
import numpy as np
import matplotlib.pyplot as plt

# Paramètres
X = np.linspace(-5, 5, 1000)
mu = 0.5  # Forte convexité
L = 3     # L-lisse
x0 = 2    # Point d'intérêt
y0 = x0**2  # f(x0)

plt.figure(figsize=(10, 6))

# Tracer les fonctions
plt.plot(X, X**2, 'b-', linewidth=2, label=r"$f(x) = x^2$")
plt.plot(X, mu/2*X**2 + (4-2*mu)*(X-1), 'g--', linewidth=1.5, label=r"Borne $\mu$-fortement convexe")
plt.plot(X, L/2*X**2 + (4-2*L)*(X-1), 'r--', linewidth=1.5, label=r"Borne $L$-lisse")

# Ajouter le point d'intérêt
plt.plot(x0, y0, 'ro', markersize=8, label=r"Point $(2,4)$")


plt.title(r"Fonction de $\mathcal{F}_{\mu,L}$ : $f(x)=x^2$ avec $\mu=0.5$ et $L=3$", fontsize=12, pad=15)
plt.xlabel(r"$x$", fontsize=11)
plt.ylabel(r"$f(x)$", fontsize=11)
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=10)

plt.tight_layout()
plt.savefig('smoothstronglyconvex.png', bbox_inches='tight', dpi=300)
plt.savefig(r'C:\Users\victo\Desktop\POLYTECHNIQUE\3A\EA MAP511\Figures\smoothstronglyconvex.png')