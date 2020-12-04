import numpy as np
import pyscf
from   pyscf import gto,scf,ao2mo,lo

# ====

atomic_spins       = { 'H' : 1, 'He' : 0, 'Li' : 1, 'C' : 2, 'O' : 2, 'N' : 3, 'S' : 2, 'Al' : 1, 'Cl' : 1 }
minimal_basis_size = { 'H' : 1, 'He' : 1, 'Li' : 5, 'C' : 5, 'O' : 5, 'N' : 5, 'S' : 9, 'Al' : 9, 'Cl' : 9 }

# Useful function for linear algebra 

def projector(C,S):
    return np.dot(C,np.dot(C.T,S))

def orthonormalize(C,S,task):
    from scipy import linalg as LA
    if(task=='orthonormalize'):
       M       = np.dot(C.T,np.dot(S,C))
       val,vec = LA.eigh(M)
       idx     = val > 1.e-12
       U       = np.dot(vec[:,idx]*1.0/np.sqrt(val[idx]),vec[:,idx].T)
       C       = np.dot(C,U)
    if(task=='normalize'):
       val = np.diag(np.dot(C.T,np.dot(S,C)))
       C  /= np.sqrt(val)
    return C

def project(C,S,Cprime,task):
    from   scipy import linalg as sla
    if(task=='along'):
       M   = np.dot(np.dot(C.T,S),Cprime)
       Q,R = sla.qr(M,mode='economic')
       return np.dot(C,Q)
    else:
       M     = np.dot(np.dot(C.T,S),Cprime)
       U,s,V = sla.svd(M, full_matrices=True)
       return np.dot(C,U[:,Cprime.shape[1]:])

# This is the subroutine to construct the IAO basis 

def IAO_construction(mol,mf,nfreeze=0):
    rho  = mf.make_rdm1()
    nb   = mol.nao_nr()
    S    = mf.get_ovlp()
    C_oc = mf.mo_coeff[:,mf.mo_occ>0]
    C_vl = np.zeros((nb,nb))
   
    cs,cf = 0,0
    if(type(mol.atom)!=list):
       mol_atom = mol.atom.split(';')
       mol_atom = [ list(x.split()) for x in mol_atom]
       mol_atom = [ [x[0],(x[1],x[2],x[3])] for x in mol_atom]
    else:
       mol_atom = mol.atom
    for [a,x],(sh0,sh1,ao0,ao1) in zip(mol_atom,mol.offset_nr_by_atom()):
        pmol          = gto.Mole()
        pmol.atom     = [[a,(0.0,0.0,0.0)]]
        pmol.basis    = mol.basis
        pmol.charge   = 0
        pmol.verbose  = 0
        pmol.spin     = atomic_spins[a]
        pmol.symmetry = True
        pmol.build()
        pmf = scf.ROHF(pmol)
        pmf.kernel()
        C_vl_a = pmf.mo_coeff[:,:minimal_basis_size[a]]
        cf = cs + C_vl_a.shape[1]
        C_vl[ao0:ao1,cs:cf] = C_vl_a[:,:]
        cs = cf
   
    C_vl   = C_vl[:,:cf]
    Px     = orthonormalize(C_vl,S,'orthonormalize')
    C_oc_p = project(Px,S,C_oc,'along')
   
    M1  = projector(C_oc,S)
    M2  = projector(C_oc_p,S)
    IAO = np.dot(np.dot(M1,M2),Px) + np.dot(np.dot(np.eye(nb)-M1,np.eye(nb)-M2),Px)
    IAO = orthonormalize(IAO,S,'orthonormalize')
    IAO = lo.Boys(mol,IAO).kernel()

    if(nfreeze>0): 
       C_freeze = C_oc[:,:nfreeze]
       IAO_unfrozen = project(IAO,S,C_freeze,'outside')        
       IAO[:,:nfreeze] = C_freeze[:,:]
       IAO[:,nfreeze:] = IAO_unfrozen[:,:]

    return IAO

