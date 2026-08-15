using QuantumOptics
using QuantumOpticsBase
export @samebases, AbstractOperator, AbstractTimeDependentOperator, Basis, Bra, ChargeBasis, ChiMatrix, CompositeBasis, DataOperator, DenseChiMatrix, DenseOpType, DenseOperator, DensePauliTransferMatrix, DenseSuperOpType, DenseSuperOperator, EyeOpType, FermionBitstring, FockBasis, GenericBasis, Ket, LazyDirectSum, LazyKet, LazyProduct, LazySum, LazyTensor, ManyBodyBasis, MomentumBasis, NLevelBasis, Operator, PPT, PauliBasis, PauliTransferMatrix, PositionBasis, QuantumOpticsBase, ShiftedChargeBasis, SparseOpType, SparseOperator, SparseSuperOpType, SparseSuperOperator, SpinBasis, StateVector, SubspaceBasis, SumBasis, SuperOperator, TimeDependentSum, avg_gate_fidelity, basis, basisstate, bosonstates, chargeop, chargestate, coherentspinstate, coherentstate, coherentstate!, coherentthermalstate, cosφ, create, current_time, dagger, dense, destroy, diagonaloperator, directsum, displace, displace_analytical, displace_analytical!, dm, embed, entanglement_entropy, entropy_renyi, entropy_vn, expect, expiφ, fermionstates, fidelity, fockstate, gaussianstate, getblock, identityoperator, identitysuperoperator, lazytensor_cachesize, lazytensor_clear_cache, lazytensor_disable_cache, lazytensor_enable_cache, lazytensor_use_cache, liouvillian, logarithmic_negativity, manybodyoperator, momentum, negativity, nlevelstate, norm, normalize, normalize!, number, onebodyexpect, passive_state, paulix, pauliy, pauliz, permutesystems, phase_average, position, potentialoperator, projector, ptrace, ptranspose, qfunc, qfuncsu2, randoperator, randstate, randstate_haar, randunitary_haar, reduced, samplepoints, set_time!, setblock!, sigmam, sigmap, sigmax, sigmay, sigmaz, sinφ, spacing, sparse, sparsebasisstate, sparseprojector, spindown, spinup, spost, spre, sprepost, squeeze, static_operator, tensor, thermalstate, time_restrict, time_shift, time_stretch, tr, tracedistance, tracedistance_h, tracedistance_nh, tracenorm, tracenorm_h, tracenorm_nh, transform, transition, variance, wigner, wignersu2, ⊕, ⊗
using PyPlot
using LinearAlgebra
using Statistics
using Base.Threads
using OrdinaryDiffEq
import Dates

println("Threads: ", Threads.nthreads())

#################################################
#            number states cutoff               #
#################################################
const n_s_cutoff    = 2

#################################################
#            total number of atoms              #
#################################################
const atom_number = 1

#################################################
#            parameters definition              #
#################################################
const hbar  = 1.0
const Γ     = 1.0 # renormalize to Γ = 2π * 6 * 1e6 (MHz)
const Ωp    = 3Γ
const gs    = 2Γ
const κs    = 20Γ
const γ41   = 0Γ
const γ42   = 0Γ
const γ21   = 0Γ
const Δp    = 20Γ

#################################################
#        single atom base definition            #
#################################################
const b_spin    = NLevelBasis(3) # now only states 1, 2, 3
# the NLevelBasis is used for one single 4Ls, whereas the ManyBodyBasis is implemented for N atoms where the Hilbert space scales
# non exponentially but rather distinguishes the atoms as bosons -> like for the symmetric Dicke-subspace
const b_atoms   = ManyBodyBasis(b_spin, bosonstates(b_spin, atom_number))
const b_s_fock  = FockBasis(n_s_cutoff)

#################################################
#                  identities                   #
#################################################
const I_atoms = identityoperator(b_atoms)
const I_s     = identityoperator(b_s_fock)

#################################################
#           field base definition               #
#################################################
const a_s       = destroy(b_s_fock)
const adag_s    = create(b_s_fock)
const a_s_unit  = identityoperator(b_s_fock)

#################################################
#full Hilbert space definitions for single atom #
#################################################
b_total = tensor(b_atoms, b_s_fock)

const Sigma44 = transition(b_atoms, 3, 3)
const Sigma22 = transition(b_atoms, 2, 2)
const Sigma11 = transition(b_atoms, 1, 1)
const Sigma12 = transition(b_atoms, 1, 2)
const Sigma21 = transition(b_atoms, 2, 1)
const Sigma14 = transition(b_atoms, 1, 3)
const Sigma41 = transition(b_atoms, 3, 1)
const Sigma24 = transition(b_atoms, 2, 3)
const Sigma42 = transition(b_atoms, 3, 2)

const Sigma44_coll = tensor(Sigma44, I_s)
const Sigma22_coll = tensor(Sigma22, I_s)
const Sigma11_coll = tensor(Sigma11, I_s)
const Sigma12_coll = tensor(Sigma12, I_s)
const Sigma21_coll = tensor(Sigma21, I_s)
const Sigma14_coll = tensor(Sigma14, I_s)
const Sigma41_coll = tensor(Sigma41, I_s)
const Sigma24_coll = tensor(Sigma24, I_s)
const Sigma42_coll = tensor(Sigma42, I_s)

const A_s_coll           = tensor(I_atoms , a_s)
const Adag_s_coll        = tensor(I_atoms, adag_s)

println("State definition: |Atom, Stokes>")

println("Atom dim = ", length(b_spin))
println("Stokes dim = ", length(b_s_fock))

#################################################
#       initial state and density matrix        #
#################################################
const Ψatom = basisstate(b_atoms, [atom_number, 0, 0])# nlevelstate(b_spin, 1)
const Ψs    = fockstate(b_s_fock, 0)
const Ψ0    = tensor(Ψatom, Ψs) 

const ρ0 = dm(Ψ0)

println("Density matrix size = ", size(dense(ρ0).data))  

println("Initial density operator in state |1,0>")

#################################################
#       Hamiltonian in the rotated frame        #
#################################################
H_atom = -Δp * Sigma44_coll
H_int = gs*(Adag_s_coll*Sigma24_coll + A_s_coll*Sigma42_coll) + (Ωp/2)*(Sigma14_coll + Sigma41_coll)
H = H_atom + H_int

#################################################
#        defining the collapse operators        #
#################################################
C_s_cav  = sqrt(κs) * A_s_coll
C_24     = sqrt(γ42) * Sigma24_coll
C_14     = sqrt(γ41) * Sigma14_coll
C_12     = sqrt(γ21) * Sigma12_coll

collapse = [C_s_cav] #, C_14, C_24]

#################################################
#            density matrix numerics            #
#################################################
t_end = 1200
t_span = range(0, t_end, length=1200)

@assert ishermitian(H)
@assert tr(ρ0) ≈ 1


Ns = Adag_s_coll * A_s_coll

# number of individual Monte-Carlo trajectories
N_traj = 2000
N_t = length(t_span)


# initialize arrays with zeros
exp_ns_mcwf  = zeros(Float64, N_t)
pop4_mcwf    = zeros(Float64, N_t)
pop2_mcwf    = zeros(Float64, N_t)
pop1_mcwf    = zeros(Float64, N_t)

println("Starting to solve for the wave function using $N_traj monte carlo trajectories.")


results = Vector{NamedTuple}(undef, N_traj)

Threads.@threads for i in 1:N_traj
    tout, Ψt = timeevolution.mcwf(t_span, Ψ0, H, collapse; reltol=1e-6, abstol=1e-8)

    results[i] = (
        ns  = real.(expect(Ns, Ψt)),
        p4  = real.(expect(Sigma44_coll, Ψt)),
        p2  = real.(expect(Sigma22_coll, Ψt)),
        p1  = real.(expect(Sigma11_coll, Ψt))
    )
end

for r in results
    exp_ns_mcwf  .+= r.ns
    pop4_mcwf    .+= r.p4
    pop2_mcwf    .+= r.p2
    pop1_mcwf    .+= r.p1
end

println("Solving done. Calculating averages...")

exp_ns_mcwf  ./= N_traj
pop4_mcwf    ./= N_traj
pop2_mcwf    ./= N_traj
pop1_mcwf    ./= N_traj

exp_ns_mcwf  ./= κs

println("Expectation values have been calculated.")

ns_max  = maximum(exp_ns_mcwf)


#################################################
#                    plotting                   #
#################################################

println("Maximum number of Stokes photons in cavity: $ns_max")

timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")

figure(figsize=(5, 3))

plot(t_span, pop1_mcwf, label=L"\langle\sigma_{%$(1)%$(1)}\rangle")
plot(t_span, pop2_mcwf, label=L"\langle\sigma_{%$(2)%$(2)}\rangle")
plot(t_span, pop4_mcwf, label=L"\langle\sigma_{%$(4)%$(4)}\rangle")

title("population probability")
xlabel(L"Γt")
ylabel(L"\text{population }|i\rangle\langle i|")
legend(loc="upper right")
grid(true)

tight_layout()

figure(figsize=(5, 3))

max_stokes = maximum(real.(exp_ns_mcwf/atom_number))

plot(t_span, exp_ns_mcwf/atom_number)

title("intensity on Stokes transition")
xlabel(L"Γt")
ylabel("intensity/atom numer / a.u.")
ylim(-0.05*max_stokes, 1.1*max_stokes)
grid(true)

tight_layout()

show()