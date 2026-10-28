import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# define the parameters
HBAR    = 1
GAMMA   = 1 #2*np.pi*6*1e6
gs      = 1*GAMMA# 2*np.pi*5*1e6
gas     = 1*GAMMA
OMEGAp  = 3*GAMMA
OMEGAc  = 0.03*GAMMA
GAMMAd  = 0*GAMMA
DELTAp  = 20*GAMMA
DELTAc  = 0*GAMMA
KAPPAs  = 10*GAMMA
KAPPAas = 10*GAMMA# 10*GAMMA
# GAMMA23 = 0.5*GAMMA
# GAMMA13 = 0.5*GAMMA 

# Stark shifts
delta_g = OMEGAp**2 / (4 * DELTAp)
delta_e = gs**2/DELTAp

DELTAc = delta_e

# effective Raman rates

Gamma_s = gs**2 * OMEGAp**2 / (KAPPAs * DELTAp**2)

A_square = (KAPPAas + Gamma_s)**2 + 4 * (DELTAc - delta_g)**2

X = (Gamma_s + KAPPAas)**2 + OMEGAc**2 - 4 * (DELTAc - delta_g) * (delta_e - delta_g)
Y = 2 * (Gamma_s + KAPPAas) * (DELTAc + delta_e - 2 * delta_g)

nom_c1 = OMEGAc**2 * gas**2 * (Gamma_s + KAPPAas) * (A_square + OMEGAc**2)
nom_c2 = 4 * OMEGAc**2 * gas**2 * (Gamma_s + KAPPAas) * ((DELTAc - delta_g)**2 - (delta_e - delta_g)**2 + gas**2)

denom_1 = (DELTAc - delta_e) * X - 4 * (DELTAc - delta_g) * gas**2
denom_2 = (DELTAc - delta_e) * Y + 2 * (Gamma_s + KAPPAas) * gas**2

c1 = nom_c1 / (denom_1**2 + denom_2**2)
c2 = nom_c2 / (denom_1**2 + denom_2**2)

Gamma_as = c1

print(Gamma_s)
print(Gamma_as)

# atom number
atom_number = 1

# number states cutoff
ns_cutoff  = 2
nas_cutoff = 2

# base definition
states = [qt.basis(4, i) for i in range(4)]

state1   = states[0]
state2   = states[1]
state3   = states[2]
state4   = states[3]
basis_s  = qt.fock(ns_cutoff, 0)
basis_as = qt.fock(nas_cutoff, 0)

def atom_operator(op, atom, N):
    operators = []
    
    for i in range(N):
        if i == atom:
            operators.append(op)
        else:
            operators.append(I_atom)
            
    operators.append(I_s)
    operators.append(I_as)
    
    return qt.tensor(operators)


# identity operators for each subspace
I_atom = qt.qeye(4)
I_s    = qt.qeye(ns_cutoff)
I_as   = qt.qeye(nas_cutoff)
I      = qt.tensor([I_atom] * atom_number + [I_s, I_as]) 

# single atom transition operators

sigma_11_local = qt.projection(4, 0, 0)
sigma_22_local = qt.projection(4, 1, 1)
sigma_33_local = qt.projection(4, 2, 2)
sigma_44_local = qt.projection(4, 3, 3)

sigma_12_local = qt.projection(4, 0, 1)
sigma_21_local = sigma_12_local.dag() 

sigma_14_local = qt.projection(4, 0, 3)
sigma_41_local = sigma_14_local.dag()

sigma_24_local = qt.projection(4, 1, 3)
sigma_42_local = sigma_24_local.dag()

sigma_23_local = qt.projection(4, 1, 2)
sigma_32_local = sigma_23_local.dag()

sigma_13_local = qt.projection(4, 0, 2)
sigma_31_local = sigma_13_local.dag()

# collective operators

sigma_11 = []
sigma_22 = []
sigma_33 = []
sigma_44 = []

sigma_12 = []
sigma_21 = []

sigma_14 = []
sigma_41 = []

sigma_24 = []
sigma_42 = []

sigma_23 = []
sigma_32 = []

sigma_13 = []
sigma_31 = []

for i in range(atom_number):
    sigma_11.append(atom_operator(sigma_11_local, i, atom_number))
    sigma_22.append(atom_operator(sigma_22_local, i, atom_number))
    sigma_33.append(atom_operator(sigma_33_local, i, atom_number))
    sigma_44.append(atom_operator(sigma_44_local, i, atom_number))
    
    sigma_12.append(atom_operator(sigma_12_local, i, atom_number))
    sigma_21.append(atom_operator(sigma_21_local, i, atom_number))
    
    sigma_14.append(atom_operator(sigma_14_local, i, atom_number))
    sigma_41.append(atom_operator(sigma_41_local, i, atom_number))
    
    sigma_24.append(atom_operator(sigma_24_local, i, atom_number))
    sigma_42.append(atom_operator(sigma_42_local, i, atom_number))
    
    sigma_23.append(atom_operator(sigma_23_local, i, atom_number))
    sigma_32.append(atom_operator(sigma_32_local, i, atom_number))
    
    sigma_13.append(atom_operator(sigma_13_local, i, atom_number))
    sigma_31.append(atom_operator(sigma_31_local, i, atom_number))

Sigma_11 = sum(sigma_11)
Sigma_22 = sum(sigma_22)
Sigma_33 = sum(sigma_33)
Sigma_44 = sum(sigma_44)

Sigma_12 = sum(sigma_12)
Sigma_21 = sum(sigma_21)

Sigma_14 = sum(sigma_14)
Sigma_41 = sum(sigma_41)

Sigma_24 = sum(sigma_24)
Sigma_42 = sum(sigma_42)

Sigma_23 = sum(sigma_23)
Sigma_32 = sum(sigma_32)

Sigma_13 = sum(sigma_13)
Sigma_31 = sum(sigma_31)

print("Operators defined.")

# bosonic operators definition
a_s = qt.tensor([I_atom for _ in range(atom_number)] + [qt.destroy(ns_cutoff), I_as])
a_s_dag = a_s.dag()
a_as = qt.tensor([I_atom for _ in range(atom_number)] + [I_s, qt.destroy(nas_cutoff)])
a_as_dag = a_as.dag()
Ns = a_s_dag * a_s
Nas = a_as_dag * a_as

print("State definition: |Atom, Stokes, anti-Stokes>")
print("State definition: |Atom_1, ..., Atom_N, Stokes, anti-Stokes>")
print(f"Full Hilbert-space dimensions: {I.shape}")
print(f"Total Hilbert-space dimension: {I.shape[0]}")
# initial state
psi_atoms = [qt.basis(4, 1) for _ in range(atom_number)]
psi_s    = qt.fock(ns_cutoff, 0)
psi_as   = qt.fock(nas_cutoff, 0)

psi0 = qt.tensor(psi_atoms + [psi_s, psi_as])

# Hamiltonian definition
H_atom = 0
H_int = 0

H_atom = DELTAc * Sigma_33 + DELTAp * Sigma_44 + delta_e * Sigma_22 + delta_g * Sigma_11
H_int = (OMEGAp/2) * (Sigma_14 + Sigma_41) \
        + gs * (a_s_dag * Sigma_24 + a_s * Sigma_42) \
        + (OMEGAc/2) * (Sigma_23 + Sigma_32) \
        + gas * (a_as_dag * Sigma_13 + a_as * Sigma_31) 

H = H_int + H_atom

# define collapse operators and rates
c_ops = []

c_ops.append(np.sqrt(KAPPAs) * a_s)
c_ops.append(np.sqrt(KAPPAas) * a_as)
# c_ops.append(np.sqrt(GAMMA23) * Sigma_23)
# c_ops.append(np.sqrt(GAMMA13) * Sigma_13)

e_ops = [Sigma_11, Sigma_22, Sigma_33, Ns, Nas]

print("Formal master equation defined.")
print("Starting to solve the MCWF method to solve time dynamics.")

# solver master equation
tlist = np.linspace(0, 2000, 1000)
data = qt.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=e_ops)
# data = qt.mcsolve(H, psi0, tlist, c_ops=c_ops, e_ops=e_ops, ntraj=2000)


# numerics with data
pop1_expect = data.expect[0]
pop2_expect = data.expect[1]
pop3_expect = data.expect[2]
populations = [pop1_expect, pop2_expect, pop3_expect]

Ns_expect = data.expect[3]
Nas_expect = data.expect[4]

max_ns = max(Ns_expect)
max_nas = max(Nas_expect)


# calculate the theoretical population of state 1 

# def ode_system(t, y):
#     rho_ee, rho_gg = y
#     rho_aa = 1.0 - rho_ee - rho_gg
#     d_rho_ee = c1 * (rho_aa - rho_ee) + c2 * (rho_aa - rho_gg)
#     d_rho_gg = -d_rho_ee
#     return [d_rho_ee, d_rho_gg]

# y0 = [1.0, 0.0]
# sol = solve_ivp(ode_system, (tlist[0], tlist[-1]), y0, t_eval=tlist)

# pop2_theory = sol.y[0]
# pop1_theory = sol.y[1]
# gamma_fit = c1 - c2

Gamma_relax = Gamma_s + c1

pop1_theory = c1 / (c1 + Gamma_s) * (1 - np.exp(-Gamma_relax * tlist))


print("=====================================")
print(rf"Theoretical Gamma_s: {Gamma_s:.6e}")
print(rf"Theoretical Gamma_as: {Gamma_as:.6e}")
print(rf"Fitted Gamma_as: {Gamma_relax:.6e}")
print(f"relative deviation: {abs(Gamma_relax - Gamma_as - Gamma_s) / Gamma_as * 100:.2f}%")
print("=====================================")

print(rf"c_1: {c1}")
print(rf"c_2: {c2}")
# plotting the expectation and theoretical values

print("Starting to plot time dynamics.")

plt.figure(figsize=(6, 4))

# plt.rcParams['text.usetex'] = True

for i, population in enumerate(populations, start=1):
    plt.plot(tlist, population, label=fr'$\langle \hat{{S}}_{{{i}{i}}}\rangle$')
    
plt.plot(tlist, pop1_theory, 'k--', label=r"$\Gamma_\text{relax} = \Gamma_\text{{s}} + \Gamma_\text{{as}}$ theory")    
    
plt.title("Population")
plt.xlabel(r'$\Gamma t$')
plt.ylabel(r'$\langle \hat{{S}}_{ii}\rangle$')
plt.legend(loc='center right')
plt.grid(True)

plt.figure(figsize=(6, 4))

photon_number = [(Ns_expect, "Stokes"), (Nas_expect, "anti-Stokes")]

for expectation, label in photon_number:
    plt.plot(tlist, expectation, label=label)

plt.title("Photon Number")
plt.xlabel(fr'$\Gamma t$')
plt.ylabel(r'$\langle \hat{N} \rangle$')

if OMEGAp != 0:
    plt.legend(loc="upper right")
else:
    plt.legend(loc='center right')
    
plt.grid(True)

plt.show()