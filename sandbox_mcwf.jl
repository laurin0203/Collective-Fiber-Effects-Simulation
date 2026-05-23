using QuantumOptics
using QuantumOpticsBase
export @samebases, AbstractOperator, AbstractTimeDependentOperator, Basis, Bra, ChargeBasis, ChiMatrix, CompositeBasis, DataOperator, DenseChiMatrix, DenseOpType, DenseOperator, DensePauliTransferMatrix, DenseSuperOpType, DenseSuperOperator, EyeOpType, FermionBitstring, FockBasis, GenericBasis, Ket, LazyDirectSum, LazyKet, LazyProduct, LazySum, LazyTensor, ManyBodyBasis, MomentumBasis, NLevelBasis, Operator, PPT, PauliBasis, PauliTransferMatrix, PositionBasis, QuantumOpticsBase, ShiftedChargeBasis, SparseOpType, SparseOperator, SparseSuperOpType, SparseSuperOperator, SpinBasis, StateVector, SubspaceBasis, SumBasis, SuperOperator, TimeDependentSum, avg_gate_fidelity, basis, basisstate, bosonstates, chargeop, chargestate, coherentspinstate, coherentstate, coherentstate!, coherentthermalstate, cosφ, create, current_time, dagger, dense, destroy, diagonaloperator, directsum, displace, displace_analytical, displace_analytical!, dm, embed, entanglement_entropy, entropy_renyi, entropy_vn, expect, expiφ, fermionstates, fidelity, fockstate, gaussianstate, getblock, identityoperator, identitysuperoperator, lazytensor_cachesize, lazytensor_clear_cache, lazytensor_disable_cache, lazytensor_enable_cache, lazytensor_use_cache, liouvillian, logarithmic_negativity, manybodyoperator, momentum, negativity, nlevelstate, norm, normalize, normalize!, number, onebodyexpect, passive_state, paulix, pauliy, pauliz, permutesystems, phase_average, position, potentialoperator, projector, ptrace, ptranspose, qfunc, qfuncsu2, randoperator, randstate, randstate_haar, randunitary_haar, reduced, samplepoints, set_time!, setblock!, sigmam, sigmap, sigmax, sigmay, sigmaz, sinφ, spacing, sparse, sparsebasisstate, sparseprojector, spindown, spinup, spost, spre, sprepost, squeeze, static_operator, tensor, thermalstate, time_restrict, time_shift, time_stretch, tr, tracedistance, tracedistance_h, tracedistance_nh, tracenorm, tracenorm_h, tracenorm_nh, transform, transition, variance, wigner, wignersu2, ⊕, ⊗
using PyPlot
using LinearAlgebra
using Statistics
using Base.Threads

println("Threads: ", Threads.nthreads())

#################################################
#            number states cutoff               #
#################################################
const n_s_cutoff    = 2
const n_as_cutoff   = 2

#################################################
#            total number of atoms              #
#################################################
const atom_number = 1

#################################################
#            parameters definition              #
#################################################
# const hbar  = 1.0
# const Γ     = 1.0 # renormalize to Γ = 2π * 6 * 1e6 (MHz)
# const gs    = 1Γ
# const gas   = 0.5Γ
# const Ωp    = 0.2Γ
# const Ωc    = 0.01Γ
# const κs    = 1Γ
# const κas   = 1Γ
# const ω2    = 0 #2π*6.83468261090429*1e9 # -> can practically be set to 0 due to high oscillations, no effect on dynamics at all
# const γ42   = 0
# const γ41   = 0
# const γ31   = 0
# const γ32   = 0
# const γ21   = 0
# const Δp    = 5Γ
# const Δc    = 0Γ

# const hbar  = 1.0
# const Γ     = 1.0 # renormalize to Γ = 2π * 6 * 1e6 (MHz)
# const gs    = 1Γ
# const gas   = 0.3Γ
# const Ωp    = 0.4Γ
# const Ωc    = 0.3Γ
# const κs    = 1Γ
# const κas   = 1Γ
# const ω2    = 0 #2π*6.83468261090429*1e9 # -> can practically be set to 0 due to high oscillations, no effect on dynamics at all
# const γ42   = 0
# const γ41   = 0
# const γ31   = 0.01Γ
# const γ32   = 0.01Γ
# const γ21   = 0
# const Δp    = 5Γ
# const Δc    = 1Γ

const hbar  = 1.0
const Γ     = 1.0 # renormalize to Γ = 2π * 6 * 1e6 (MHz)
const gs    = 1Γ
const Ωp    = 1.1Γ
const κs    = 1Γ
const κas   = 1Γ
const ω2    = 0
const γ42   = 0
const γ41   = 0
const γ31   = 0.01Γ
const γ32   = 0.01Γ
const γ21   = 0
const Δp    = 5Γ
const Δc    = 1Γ

# define gas and Ωc in dependence on the parameters gs, Ωp of the Stokes photon generation
gas = ( (4 * κas * gs^2 * Ωp^2 * (((γ31 + γ32) / 2)^2 + Δc^2)) / (κs * Δp^2) )^(1/4)

Ωc = gas


#################################################
#      check for superradiant laser regime      #
#################################################

function check_for_superradiant_regime(pump, coupling, loss_rate, text)
    if loss_rate > coupling && pump < (coupling^2 / loss_rate)
        println("You are working in the superradiant regime for the $text.")
    else 
        println("You are not working in the superradiant regime for the $text.")
    end
    println("Rabi-frequency Ω: $pump")
    println("Coupling constant: $coupling")
    println("Cavity loss rate: $loss_rate")
end

check_for_superradiant_regime(Ωp, gs, κs, "pump")
check_for_superradiant_regime(Ωc, gas, κas, "control")

#################################################
#        single atom base definition            #
#################################################
const b_spin    = NLevelBasis(4)
# the NLevelBasis is used for one single 4Ls, whereas the ManyBodyBasis is implemented for N atoms where the Hilbert space scales
# non exponentially but rather distinguishes the atoms as bosons -> like for the symmetric Dicke-subspace
const b_atoms   = ManyBodyBasis(b_spin, bosonstates(b_spin, atom_number))
const b_s_fock  = FockBasis(n_s_cutoff)
const b_as_fock = FockBasis(n_as_cutoff)

#################################################
#                  identities                   #
#################################################
const I_atoms = identityoperator(b_atoms)
const I_s     = identityoperator(b_s_fock)
const I_as    = identityoperator(b_as_fock)

#################################################
#           field base definition               #
#################################################
const a_s       = destroy(b_s_fock)
const adag_s    = create(b_s_fock)
const a_as      = destroy(b_as_fock)
const adag_as   = create(b_as_fock)
const a_s_unit  = identityoperator(b_s_fock)
const a_as_unit = identityoperator(b_as_fock)

# println("Bosonic Operators of the Stokes field:")
# display(dense(a_s).data)
# println("Bosonic Operators of the anti-Stokes field:")
# display(dense(a_as).data)

#################################################
#full Hilbert space definitions for single atom #
#################################################
b_total = tensor(b_atoms, b_s_fock, b_as_fock)

const Sigma44 = transition(b_atoms, 4, 4)
const Sigma33 = transition(b_atoms, 3, 3)
const Sigma22 = transition(b_atoms, 2, 2)
const Sigma11 = transition(b_atoms, 1, 1)
const Sigma12 = transition(b_atoms, 1, 2)
const Sigma21 = transition(b_atoms, 2, 1)
const Sigma13 = transition(b_atoms, 1, 3)
const Sigma31 = transition(b_atoms, 3, 1)
const Sigma14 = transition(b_atoms, 1, 4)
const Sigma41 = transition(b_atoms, 4, 1)
const Sigma23 = transition(b_atoms, 2, 3)
const Sigma32 = transition(b_atoms, 3, 2)
const Sigma24 = transition(b_atoms, 2, 4)
const Sigma42 = transition(b_atoms, 4, 2)

# collective operators in total Hilbert space
const Sigma44_coll = tensor(Sigma44, I_s, I_as)
const Sigma33_coll = tensor(Sigma33, I_s, I_as)
const Sigma22_coll = tensor(Sigma22, I_s, I_as)
const Sigma11_coll = tensor(Sigma11, I_s, I_as)
const Sigma12_coll = tensor(Sigma12, I_s, I_as)
const Sigma21_coll = tensor(Sigma21, I_s, I_as)
const Sigma13_coll = tensor(Sigma13, I_s, I_as)
const Sigma31_coll = tensor(Sigma31, I_s, I_as)
const Sigma14_coll = tensor(Sigma14, I_s, I_as)
const Sigma41_coll = tensor(Sigma41, I_s, I_as)
const Sigma23_coll = tensor(Sigma23, I_s, I_as)
const Sigma32_coll = tensor(Sigma32, I_s, I_as)
const Sigma24_coll = tensor(Sigma24, I_s, I_as)
const Sigma42_coll = tensor(Sigma42, I_s, I_as)

# println("Collective operator dim = ", length(collective_basis))
# display(dense(collective_identity).data)

const A_s_coll           = tensor(I_atoms , a_s, I_as)
const A_as_coll          = tensor(I_atoms, I_s, a_as)
const Adag_s_coll        = tensor(I_atoms, adag_s, I_as)
const Adag_as_coll       = tensor(I_atoms, I_s, adag_as)

println("State definition: |Atom, Stokes, anti-Stokes>")

println("Atom dim = ", length(b_spin))
println("Stokes dim = ", length(b_s_fock))
println("anti-Stokes dim = ", length(b_as_fock))         

#################################################
#       initial state and density matrix        #
#################################################
const Ψatom = basisstate(b_atoms, [atom_number, 0, 0, 0])# nlevelstate(b_spin, 1)
const Ψs    = fockstate(b_s_fock, 0)
const Ψas   = fockstate(b_as_fock, 0) 
const Ψ0    = tensor(Ψatom, Ψs, Ψas) 

const ρ0 = dm(Ψ0)

println("Density matrix size = ", size(dense(ρ0).data))  

println("Initial density operator in state |1,0,0>")
# display(Ψ0.data)

#################################################
#       Hamiltonian in the rotated frame        #
#################################################
H_atom = hbar*Δp*Sigma44_coll + hbar*Δc*Sigma33_coll + hbar*ω2*Sigma22_coll
H_int = gs*(Adag_s_coll*Sigma24_coll + A_s_coll*Sigma42_coll) + gas*(Adag_as_coll*Sigma13_coll + A_as_coll*Sigma31_coll) + (Ωp/2)*(Sigma14_coll + Sigma41_coll) + (Ωc/2)*(Sigma23_coll + Sigma32_coll)

H = H_atom + H_int

#################################################
#        defining the collapse operators        #
#################################################
C_s_cav  = sqrt(κs) * A_s_coll
C_as_cav = sqrt(κas) * A_as_coll
C_24     = sqrt(γ42) * Sigma24_coll
C_14     = sqrt(γ41) * Sigma14_coll
C_23     = sqrt(γ32) * Sigma23_coll
C_13     = sqrt(γ31) * Sigma13_coll
C_12     = sqrt(γ21) * Sigma12_coll

collapse = [C_s_cav, C_as_cav, C_14, C_24, C_23, C_13] # , C_14, C_13, C_12, C_23, C_24]

#################################################
#            density matrix numerics            #
#################################################
t_end = 5000
t_span = range(0, t_end, length=800)

@assert ishermitian(H)
@assert tr(ρ0) ≈ 1


Ns = Adag_s_coll * A_s_coll
Nas = Adag_as_coll * A_as_coll

N_traj = 300
N_t = length(t_span)


# initialize arrays with zeros
exp_ns_mcwf  = zeros(Float64, N_t)
exp_nas_mcwf = zeros(Float64, N_t)
pop4_mcwf    = zeros(Float64, N_t)
pop3_mcwf    = zeros(Float64, N_t)
pop2_mcwf    = zeros(Float64, N_t)
pop1_mcwf    = zeros(Float64, N_t)

println("Starting to solve for the wave function using $N_traj monte carlo trajectories.")


results = Vector{NamedTuple}(undef, N_traj)

Threads.@threads for i in 1:N_traj
    tout, Ψt = timeevolution.mcwf(t_span, Ψ0, H, collapse)

    results[i] = (
        ns  = real.(expect(Ns, Ψt)),
        nas = real.(expect(Nas, Ψt)),
        p4  = real.(expect(Sigma44_coll, Ψt)),
        p3  = real.(expect(Sigma33_coll, Ψt)),
        p2  = real.(expect(Sigma22_coll, Ψt)),
        p1  = real.(expect(Sigma11_coll, Ψt))
    )
end

for r in results
    exp_ns_mcwf  .+= r.ns
    exp_nas_mcwf .+= r.nas
    pop4_mcwf    .+= r.p4
    pop3_mcwf    .+= r.p3
    pop2_mcwf    .+= r.p2
    pop1_mcwf    .+= r.p1
end

println("Solving done. Calculating averages...")

# expectation values
# 4*1e-4 from where
exp_ns_mcwf  ./= N_traj
exp_nas_mcwf ./= N_traj
pop4_mcwf    ./= N_traj
pop3_mcwf    ./= N_traj
pop2_mcwf    ./= N_traj
pop1_mcwf    ./= N_traj
exp_ns_mcwf  ./= (4*1e-4)
exp_nas_mcwf  ./= (4*1e-4)

println("Expectation values have been calculated.")

#################################################
#                    plotting                   #
#################################################
figure(figsize=(16,6))

subplot(2, 3, 1)
plot(t_span, pop4_mcwf)
title(L"\text{Population of } |4\rangle")
xlabel(L"Γt")
ylabel(L"\langle \sigma_{44} \rangle")
ylim([0, atom_number])
println("Maximum population propability of |4>: ", maximum(pop4_mcwf))


subplot(2, 3, 2)
plot(t_span, pop3_mcwf)
title(L"\text{Population of } |3\rangle")
xlabel(L"Γt")
ylabel(L"\langle \sigma_{33} \rangle")
ylim([0, atom_number])
println("Maximum population propability of |3>: ", maximum(pop3_mcwf))

subplot(2, 3, 3)
plot(t_span, pop2_mcwf)
title(L"\text{Population of } |2\rangle")
xlabel(L"Γt")
ylabel(L"\langle \sigma_{22} \rangle")
ylim([0, atom_number])

subplot(2, 3, 4)
plot(t_span, pop1_mcwf)
title(L"\text{Population of } |1\rangle")
xlabel(L"Γt")
ylabel(L"\langle \sigma_{11} \rangle")
ylim([0, atom_number])

subplot(2, 3, 5)
plot(t_span, exp_ns_mcwf/atom_number)
title("Relative Intensity on Stokes transition")
xlabel(L"Γt")
ylabel("intensity/atom number / a.u.")
ylim([0, maximum(real.(exp_ns_mcwf/atom_number)) + 0.1*maximum(real.(exp_ns_mcwf/atom_number))])

subplot(2, 3, 6)
plot(t_span, exp_nas_mcwf/atom_number)
title("Relative Intensity on anti-Stokes transition")
xlabel(L"Γt")
ylabel("intensity/atom number / a.u.")
ylim([0, maximum(real.(exp_nas_mcwf/atom_number)) + 0.1*maximum(real.(exp_nas_mcwf/atom_number))])

tight_layout()
show()
