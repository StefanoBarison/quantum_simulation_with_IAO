# This is the main script of subspace construction: we consider a molecular geometry
# and construct, starting from a big basis with many virtual orbitals, a 'small basis'
# using different procedures ('iao,'low_hf','low_mp','casscf')

import os
import numpy as np
from scipy import linalg as LA

def generate_hamiltonian(geometry,basis,procedure,n_orb,n_freeze,path):
 import numpy as np
 from pyscf import gto,scf,ao2mo,mp,cc,fci,tools
 from iao   import IAO_construction
 from utils import get_MP2_natural_orbitals,frozen_core,get_ES,get_CASSCF_orbitals
 from print_rcas import print_RCAS_wavefunction

 # Create the molecule and make a RHF calculation in a big basis

 mol = gto.M(atom=geometry,charge=0,spin=0,basis=basis,symmetry=True,verbose=4)
 mf  = scf.RHF(mol)
 Ehf = mf.kernel()
 
 # now we want to produce a basis that is the same size of a minimal STO-6G basis
 n  = n_orb
 # decide how many orbitals to freeze
 nfreeze = n_freeze

 # Different way to extract the subspaces
 #1: lowest energy HF orbitals
 if(procedure=='low_hf'):
   C = mf.mo_coeff[:,:n]
 #2: natural orbitals of the second-order perturbative theory MP2
 if(procedure=='low_mp'):
   C = get_MP2_natural_orbitals(mf,n,nfreeze)
 #3: Gerald's intrinsic atomic orbitals procedure
 if(procedure=='iao'):
   C = IAO_construction(mol,mf,nfreeze)
 #4: CASSCF lowest energy orbitals
 if(procedure=='casscf'):
   C = get_CASSCF_orbitals(mf,n,nfreeze)
 
 # consctruct the matrix elements of the Hamiltonian in the desired basis
 na,nb = mol.nelectron//2,mol.nelectron//2
 E0 = mol.energy_nuc()
 h1 = np.einsum('ab,ai,bk->ik',mf.get_hcore(),C,C)
 h2 = ao2mo.restore(1,ao2mo.kernel(mol,C),n)
 
 # frezze the desired orbitals
 n,na,nb,E0,h1,h2 = frozen_core(n,na,nb,E0,h1,h2,nfreeze,C)

 # confront the original basis energies with those of the reduced basis
 mcc = cc.CCSD(mf,nfreeze)
 Ecc = mcc.kernel()[0]
 Ehf_2,Ecc_2,mol_2,mf_2 = get_ES(n,na,nb,E0,h1,h2)

 fname = path+'_'+mol.basis+'_'+procedure+'.out'
 print(fname)
 outf  = open(fname,'w+')
 outf.write("original basis "+str(  Ehf)+" "+str(    Ecc+Ehf)+"\n")
 outf.write("reduced  basis "+str(Ehf_2)+" "+str(Ecc_2+Ehf_2)+"\n")

 mp2_solver  = mp.MP2(mf_2,frozen=0)
 MP_energy,t = mp2_solver.kernel()
 rho         = mp2_solver.make_rdm1()
 nat_occ,nat_orb = LA.eigh(-rho)

 # Write down natural orbitals as linear combination of the reduced basis orbitals (C) 

 nat_orb = np.dot(mf_2.mo_coeff,nat_orb)
 
 cisolver = fci.FCI(mf_2,nat_orb)
 E,ci     = cisolver.kernel()
 outf.write("reduced  basis FCI "+str(E+E0)+"\n")

 # FCI expansion of the reduced basis (optional but useful)
 norb  = mol_2.nao_nr()
 na,nb = (mol_2.nelectron+mol_2.spin)//2,(mol_2.nelectron-mol_2.spin)//2
 eps   = 0.0001
 print_RCAS_wavefunction(cisolver,norb,na,nb,1-eps,outf,0,path)

 # matrix elements in the new natural orbital basis
 h1 = np.einsum('pi,pq->iq',nat_orb,h1)
 h1 = np.einsum('qj,iq->ij',nat_orb,h1)
 h2 = np.einsum('pi,prqs->irqs',nat_orb,h2)
 h2 = np.einsum('rj,irqs->ijqs',nat_orb,h2)
 h2 = np.einsum('qk,ijqs->ijks',nat_orb,h2)
 h2 = np.einsum('sl,ijks->ijkl',nat_orb,h2)

 outf.close()

 # reduced basis    :      |ci) = \sum_a C(a,i) |a)  (A: atomic orbitals)
 # natural orbitals : |nk) = \sum_i N(i,k) |ci) (I: orbitals from the little basis)
 # therefore         : |nk) = \sum_ia N(i,k) C(a,i) |a)
 extended_nat_orbs = np.eye(nat_orb.shape[0]+nfreeze)
 extended_nat_orbs[nfreeze:,nfreeze:] = nat_orb[:,:]
 Ctilde = np.einsum('ai,ik->ak',C,extended_nat_orbs)
 tools.molden.from_mo(mol,path+'_'+procedure+'_NO.molden',Ctilde)

 # write the hamiltonian on a file that will be processed using qiskit PySCFDriver

 fname = path+'_'+mol.basis+'_'+procedure+'.h.csv'
 outf = open(fname,'w+')
 outf.write('%d,%d,%d\n' % (n,na,nb))
 outf.write('%f\n' % E0)
 for i in range(n):
  for j in range(n):
   outf.write('%d,%d,%f\n' % (i,j,h1[i,j]))
 for p in range(n):
  for r in range(n):
   for q in range(n):
    for s in range(n):
     outf.write('%d,%d,%d,%d,%f \n' % (p,r,q,s,h2[p,r,q,s]))
 outf.close()

# ==========

# The second routine takes the .h.csv file produced by generate_hamiltonian 
# and substitutes it into the PySCFDriver of Qiskit that has already been built 
# with a dummy molecule 


def overwrite_molecule(filename,m):

 import numpy as np
 from pyscf import scf,gto,ao2mo,cc

 #============================================#
 # input file, must contain                   #
 # nbasis,nup,ndown                           #
 # E0                                         #
 # i,j,h[i,j]                                 #
 # p,r,q,s h[p,r,q,s] in Chemist's notation   #
 # in an orthonormal basis                    #
 #============================================#
 linf = open(filename,'r').read().split('\n')
 linf = [x.split(',') for x in linf]
 n,na,nb = linf[0]
 Enuc    = linf[1][0]
 n,na,nb = int(n),int(na),int(nb)
 Enuc    = float(Enuc)
 s1 = np.zeros((n,n))
 h1 = np.zeros((n,n))
 h2 = np.zeros((n,n,n,n))
 count = 2
 for mu in range(n**2):
  II,JJ = linf[count][0],linf[count][1]
  II,JJ = int(II),int(JJ)
  hij   = linf[count][2]
  hij   = float(hij)
  if(II==JJ): s1[II,JJ] = 1
  else:       s1[II,JJ] = 0
  h1[II,JJ] = hij
  count += 1
 for mu in range(n**4):
  PP,RR,QQ,SS = linf[count][0],linf[count][1],linf[count][2],linf[count][3]
  PP,RR,QQ,SS = int(PP),int(RR),int(QQ),int(SS)
  vprqs = linf[count][4]
  vprqs = float(vprqs)
  h2[PP,RR,QQ,SS] = vprqs
  count += 1

 # SCF for internal checks
 mol = gto.M(verbose=2)
 mol.charge = 0
 mol.nelectron = na+nb
 mol.spin = na-nb
 mol.incore_anyway = True
 mol.nao_nr = lambda *args : n
 mol.energy_nuc = lambda *args: Enuc

 if(mol.spin==0): mf = scf.RHF(mol)
 if(mol.spin==1): mf = scf.ROHF(mol)

 mf.get_hcore  = lambda *args: h1
 mf.get_ovlp   = lambda *args: s1
 mf._eri       = ao2mo.restore(8,h2,n)
 mf.init_guess = '1e'
 E0 = mf.kernel() +Enuc
 if(not mf.converged):
  mf = scf.newton(mf)
  E0 = mf.kernel() +Enuc

 if(n<11):
  from pyscf import mcscf
  mycas = mcscf.CASSCF(mf,ncas=n,nelecas=mol.nelectron)
  E1 = mycas.kernel()[0] +Enuc
  print("PySCF energies: SCF, FCI %f %f \n " % (E0,E1))
 else:
  print("PySCF energy: SCF %f " % (E0))

 m.origin_driver_name = 'user-defined'
 m.origin_driver_version = None
 m.origin_driver_config = None

 m.hf_energy = E0
 m.nuclear_repulsion_energy = Enuc
 m.num_orbitals = n
 m.num_alpha = na
 m.num_beta = nb
 m.mo_coeff = mf.mo_coeff
 m.mo_coeff_B = None
 m.orbital_energies = mf.mo_energy
 m.orbital_energies_B = None

 m.molecular_charge = None
 m.multiplicity = None
 m.num_atoms = None
 m.atom_symbol = None
 m.atom_xyz = None

 m.hcore = h1
 m.hcore_B = None
 m.kinetic = np.zeros((n,n))
 m.overlap = np.eye(n)
 m.eri = h2

 m.mo_onee_ints = h1
 m.mo_onee_ints_B = None
 m.mo_eri_ints = h2
 m.mo_eri_ints_BB = None
 m.mo_eri_ints_BA = None

 m.mo_onee_ints = np.einsum('pi,pr->ir',m.mo_coeff,m.mo_onee_ints)
 m.mo_onee_ints = np.einsum('ra,ir->ia',m.mo_coeff,m.mo_onee_ints)

 m.mo_eri_ints = np.einsum('pi,prqs->irqs',m.mo_coeff,m.mo_eri_ints)
 m.mo_eri_ints = np.einsum('ra,irqs->iaqs',m.mo_coeff,m.mo_eri_ints)
 m.mo_eri_ints = np.einsum('qj,iaqs->iajs',m.mo_coeff,m.mo_eri_ints)
 m.mo_eri_ints = np.einsum('sb,iajs->iajb',m.mo_coeff,m.mo_eri_ints)

 m.x_dip_ints = None
 m.y_dip_ints = None
 m.z_dip_ints = None

 m.x_dip_mo_ints = None
 m.x_dip_mo_ints_B = None
 m.y_dip_mo_ints = None
 m.y_dip_mo_ints_B = None
 m.z_dip_mo_ints = None
 m.z_dip_mo_ints_B = None
 m.nuclear_dipole_moment = None
 m.reverse_dipole_sign = None

 #from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
 #singles,doubles = UCCSD.compute_excitation_lists([na,nb],2*n)

 mycc = cc.CCSD(mf)
 E,t1,t2 = mycc.kernel()

 m.t_amplitudes = (t1,t2)
