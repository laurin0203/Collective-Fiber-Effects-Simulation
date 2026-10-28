using QuantumOptics
using QuantumOpticsBase
using PyPlot
using LinearAlgebra

#################################################
#            parameters definition              #
#################################################
const hbar  = 1.0
const Γ     = 1
const gs    = 1
const gas   = 1
const Ωp    = 3.0Γ
const Ωc    = 0.03Γ
const κs    = 10Γ
const κas   = 10Γ
const ω2    = 0 #2π*6.83468261090429*1e9 # -> can practically be set to 0 due to high oscillations, no effect on dynamics at all
const γ42   = 0#0.05Γ
const γ31   = 0#0.05Γ
const γ21   = 0#0.06Γ
const Δp    = 20Γ
const Δc    = 0Γ

#################################################
#            number states cutoff               #
#################################################
const n_s_cutoff    = 2
const n_as_cutoff   = 2

#################################################
#               base definition                 #
#################################################
const b_spin    = NLevelBasis(4)
const b_s_fock  = FockBasis(n_s_cutoff)
const b_as_fock = FockBasis(n_as_cutoff)


#################################################
#           atom base definition                #
#################################################
const σ44    = transition(b_spin, 4, 4)
const σ33    = transition(b_spin, 3, 3)
const σ22    = transition(b_spin, 2, 2)
const σ11    = transition(b_spin, 1, 1)
const σ12    = transition(b_spin, 1, 2)
const σ21    = transition(b_spin, 2, 1)
const σ13    = transition(b_spin, 1, 3)
const σ31    = transition(b_spin, 3, 1)
const σ14    = transition(b_spin, 1, 4)
const σ41    = transition(b_spin, 4, 1)
const σ24    = transition(b_spin, 2, 4)
const σ42    = transition(b_spin, 4, 2)
const σ23    = transition(b_spin, 2, 3)
const σ32    = transition(b_spin, 3, 2)
const σ_unit = identityoperator(b_spin)

println("Unit operator of the 4-level-system:")
display(dense(σ_unit).data)

#################################################
#           field base definition               #
#################################################
const a_s       = destroy(b_s_fock)
const adag_s    = create(b_s_fock)
const a_as      = destroy(b_as_fock)
const adag_as   = create(b_as_fock)
const a_s_unit  = identityoperator(b_s_fock)
const a_as_unit = identityoperator(b_as_fock)

println("Bosonic Operators of the Stokes field:")
display(dense(a_s).data)
println("Bosonic Operators of the anti-Stokes field:")
display(dense(a_as).data)

#################################################
#        full Hilbert space definitions         #
#################################################
const Sigma44       = tensor(tensor(σ44, a_s_unit), a_as_unit)
const Sigma33       = tensor(tensor(σ33, a_s_unit), a_as_unit)
const Sigma22       = tensor(tensor(σ22, a_s_unit), a_as_unit)
const Sigma11       = tensor(tensor(σ11, a_s_unit), a_as_unit)
const Sigma12       = tensor(tensor(σ12, a_s_unit), a_as_unit)
const Sigma21       = tensor(tensor(σ21, a_s_unit), a_as_unit)
const Sigma13       = tensor(tensor(σ13, a_s_unit), a_as_unit)
const Sigma31       = tensor(tensor(σ31, a_s_unit), a_as_unit)
const Sigma14       = tensor(tensor(σ14, a_s_unit), a_as_unit)
const Sigma41       = tensor(tensor(σ41, a_s_unit), a_as_unit)
const Sigma32       = tensor(tensor(σ32, a_s_unit), a_as_unit)
const Sigma23       = tensor(tensor(σ23, a_s_unit), a_as_unit)
const Sigma24       = tensor(tensor(σ24, a_s_unit), a_as_unit)
const Sigma42       = tensor(tensor(σ42, a_s_unit), a_as_unit)
const A_s           = tensor(tensor(σ_unit, a_s), a_as_unit)
const A_as          = tensor(tensor(σ_unit, a_s_unit), a_as)
const Adag_s        = tensor(tensor(σ_unit, adag_s), a_as_unit)
const Adag_as       = tensor(tensor(σ_unit, a_s_unit), adag_as)

println("State definition: |Atom, Stokes, anti-Stokes>")

println("Atom dim = ", length(b_spin))
println("Stokes dim = ", length(b_s_fock))
println("anti-Stokes dim = ", length(b_as_fock))         

#################################################
#       initial state and density matrix        #
#################################################
const Ψatom = nlevelstate(b_spin, 1)
const Ψs    = fockstate(b_s_fock, 0)
const Ψas   = fockstate(b_as_fock, 0) 
const Ψ0    = tensor(tensor(Ψatom, Ψs), Ψas) 

const ρ0 = dm(Ψ0)

println("Density matrix size = ", size(dense(ρ0).data))  

println("Initial density operator in state |1,0,0>")
display(Ψ0.data)

#################################################
#       Hamiltonian in the rotated frame        #
#################################################
H_atom = Δp*Sigma44 + Δc*Sigma33
H_int = gs*(Adag_s*Sigma24 + A_s*Sigma42) + gas*(Adag_as*Sigma13 + A_as*Sigma31) + Ωp/2*(Sigma14 + Sigma41) + Ωc/2*(Sigma23 + Sigma32)

H = H_atom + H_int

#################################################
#        defining the collapse operators        #
#################################################
C_s_cav  = sqrt(κs) * A_s
C_as_cav = sqrt(κas) * A_as
# C_42     = sqrt(γ42) * Sigma42
# C_31     = sqrt(γ31) * Sigma31
# C_21     = sqrt(γ21) * Sigma21

collapse = [C_s_cav, C_as_cav] #, C_42, C_31, C_21]

#################################################
#            density matrix numerics            #
#################################################
t_end = 1500
t_span = range(0, t_end, length=2001)

@assert ishermitian(dense(H).data)
@assert tr(dense(ρ0).data) ≈ 1

tout, ρt = timeevolution.master_h(t_span, ρ0, H, collapse)

#################################################
#              expectation values               #
#################################################
exp_ns_master = expect(A_s' * A_s, ρt)
exp_nas_master = expect(A_as' * A_as, ρt)

pop4 = expect(Sigma44, ρt)
pop3 = expect(Sigma33, ρt)
pop2 = expect(Sigma22, ρt)
pop1 = expect(Sigma11, ρt)

#################################################
#                    plotting                   #
#################################################
figure(figsize=(12,6))

# Index von 1 bis 6
subplot(2, 3, 1)
plot(tout, exp_ns_master)
title("Stokes photon number")
xlabel("t")
ylabel(L"\langle n_\text{s} \rangle")
ylim([0, n_s_cutoff])

subplot(2, 3, 4)
plot(tout, exp_nas_master)
title("anti-Stokes photon number")
xlabel("t")
ylabel(L"\langle n_\text{as} \rangle")
ylim([0, n_as_cutoff])

subplot(2, 3, 3)
plot(tout, pop4)
title(L"\text{Population of } |4\rangle")
xlabel("t")
ylabel(L"\langle \sigma_{44} \rangle")
ylim([0,1])

subplot(2, 3, 2)
plot(tout, pop3)
title(L"\text{Population of } |3\rangle")
xlabel("t")
ylabel(L"\langle \sigma_{33} \rangle")
ylim([0,1])

subplot(2, 3, 5)
plot(tout, pop2)
title(L"\text{Population of } |2\rangle")
xlabel("t")
ylabel(L"\langle \sigma_{22} \rangle")
ylim([0,1])

subplot(2, 3, 6)
plot(tout, pop1)
title(L"\text{Population of } |1\rangle")
xlabel("t")
ylabel(L"\langle \sigma_{11} \rangle")
ylim([0,1])

tight_layout()
show()
