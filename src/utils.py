import numpy as np
from pyscf import gto,scf,ao2mo,mp,cc,mcscf

# subroutine to generate antural orbitals of MP"
def get_MP2_natural_orbitals(mf,n,nfreeze):
 mp2  = mp.MP2(mf,nfreeze)
 mp2.kernel()
 C_mf = mf.mo_coeff
 rho     = mp2.make_rdm1()
 n_mp,UU = np.linalg.eigh(rho)
 idx     = n_mp.argsort()[::-1]
 n_mp    = n_mp[idx]
 UU      = UU[:,idx]
 C_mp    = np.dot(mf.mo_coeff,UU)
 return C_mp[:,:n]

def get_CASSCF_orbitals(mf,n,nfreeze):
 na,nb = int(sum(mf.mo_occ))//2,int(sum(mf.mo_occ))//2
 cas = mcscf.CASSCF(mf,ncas=n-nfreeze,nelecas=(na-nfreeze,nb-nfreeze))
 cas.natorb = True
 cas.kernel()
 return cas.mo_coeff[:,:n]

# subroutine to implement frozen core
def frozen_core(n,na,nb,Enuc,h1,h2,nfreeze,C):
 import numpy as np
 # --- density operator of the orbitals to freeze
 rho_up = np.zeros((n,n))
 rho_dn = np.zeros((n,n))
 for i in range(nfreeze): rho_up[i,i]=1.0
 for i in range(nfreeze): rho_dn[i,i]=1.0
 # --- energetic shift
 Enuc += np.einsum('ij,ji',h1,rho_up+rho_dn)
 Enuc += 0.5*np.einsum('prqs,pr,qs',h2,rho_up+rho_dn,rho_up+rho_dn)
 Enuc -= 0.5*np.einsum('prqs,ps,qr',h2,rho_up,rho_up)
 Enuc -= 0.5*np.einsum('prqs,ps,qr',h2,rho_dn,rho_dn)
 # --- pseudopotential
 V1  = np.einsum('prqs,pr->qs',h2,rho_up+rho_dn)
 V1 -= 0.5*np.einsum('prqs,ps->qr',h2,rho_up)
 V1 -= 0.5*np.einsum('prqs,ps->qr',h2,rho_dn)
 h1 += V1
 # --- cutoff of the frozen orbitals
 C  = C[:,nfreeze:]
 h1 = h1[nfreeze:,nfreeze:]
 h2 = h2[nfreeze:,nfreeze:,nfreeze:,nfreeze:]
 return n-nfreeze,na-nfreeze,nb-nfreeze,Enuc,h1,h2

# subroutine to simulate in the extracted subspace
def get_ES(n,na,nb,Enuc,h1,h2):

    mol_FC               = gto.M(verbose=0)
    mol_FC.charge        = 0
    mol_FC.nelectron     = na+nb
    mol_FC.spin          = na-nb
    mol_FC.incore_anyway = True
    mol_FC.nao_nr        = lambda *args: n
    mol_FC.energy_nuc    = lambda *args: Enuc
    mf_FC            = scf.RHF(mol_FC)
    mf_FC.get_hcore  = lambda *args: h1
    mf_FC.get_ovlp   = lambda *args: np.eye(n)
    mf_FC._eri       = ao2mo.restore(8,h2,n)
    rho              = np.zeros((n,n))
    for i in range(na): rho[i,i] = 2.0
    Ehf = mf_FC.kernel(rho)+Enuc
    Ecc = (cc.CCSD(mf_FC)).kernel()[0]

    return Ehf,Ecc,mol_FC,mf_FC

