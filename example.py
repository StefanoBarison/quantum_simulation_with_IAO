# This is an example on how to use the IAO_construction routine to extract
# an active little subspace from a big basis set calculation and the use these 
# orbitals to initialize a VQE calculation in Qiskit

import numpy as np
import sys

sys.path.append('./src/')


# Import Qiskit useful libraries
#=========================================================
import qiskit
from qiskit                            import IBMQ    
from qiskit                            import BasicAer,Aer
from qiskit.providers.aer 			   import QasmSimulator    
from qiskit.aqua                       import set_qiskit_aqua_logging, QuantumInstance
from qiskit.aqua.operators             import Z2Symmetries, WeightedPauliOperator
from qiskit.aqua.algorithms            import VQE
from qiskit.aqua.algorithms            import NumPyEigensolver
from qiskit.aqua.components.optimizers import L_BFGS_B,CG,SPSA,SLSQP, COBYLA

from qiskit.chemistry.components.initial_states                    import HartreeFock
from qiskit.chemistry.drivers                                      import PySCFDriver, UnitsType, HFMethodType
from qiskit.chemistry.core                                         import Hamiltonian 

from qiskit.chemistry.drivers                                      import UnitsType, HFMethodType

from qiskit.chemistry.core                                         import TransformationType, QubitMappingType
from qiskit.chemistry                                              import set_qiskit_chemistry_logging,qmolecule
from qiskit.chemistry.components.variational_forms                 import UCCSD
#=========================================================

# for the IAO is sufficient to import generate_hamiltonian

from generate_hamiltonian import *

# Choose a geometry from the ones proposed in ./molecule_geometries
# with the form [at,(x,y,z)]

geometry = [['N', (0, 0, 0)], 
			['H', (0.46402132948798935, 0, 1.1066545105769003)],
			['H', (-0.61376455480392034,-0.8124442579595943, 0)],
			['H', (-0.61376455480392034, 0.8124442579595943, 0)]]

# And generate the hamiltonian of a basis with minimal size

filename  = 'NH3'
basis     = 'aug-cc-pvqz'
procedure = 'iao'
print('Creating the molecule...')

generate_hamiltonian(geometry,basis,procedure,8,0,filename)


# Now we can use the molecule created to initialize a VQE calculation

# 1) initialize a dummy molecule and then overwrite it

driver            = PySCFDriver(atom='''H 0.0 0.0 0.0; H 0.4 0.0 0.0''',unit=UnitsType.ANGSTROM,charge=0,
	                            spin=0,basis='sto-6g',hf_method=HFMethodType.ROHF)
  
molecule          = driver.run()

print('Overwriting PySCFDriver...')
overwrite_molecule(filename+'_'+basis+'_'+procedure+'.h.csv',molecule)

# Initialize the Hamiltonian and resctrict the simulation to a small subspace

orb_red = [0,1,2,6,7]
core              = Hamiltonian(qubit_mapping=QubitMappingType.PARITY,two_qubit_reduction=True,freeze_core=False,orbital_reduction=orb_red) 
qubit_op, aux_ops = core.run(molecule)

# we decide to measure also N, S^2 and S_z
aux_ops           = aux_ops[:3]
aux_ops.append(qubit_op)
dE      = core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy

# 2) Construct the HF initial state

init_state = HartreeFock(num_orbitals=core._molecule_info['num_orbitals'],
                      qubit_mapping=core._qubit_mapping, two_qubit_reduction=core._two_qubit_reduction,
                      num_particles=core._molecule_info['num_particles'])

# 3) Ansatz for the VQE 

var_form = UCCSD(reps=1,num_orbitals=core._molecule_info['num_orbitals'],num_particles=core._molecule_info['num_particles'],
	             active_occupied=None,active_unoccupied=None,initial_state=init_state,qubit_mapping=core._qubit_mapping,
	             two_qubit_reduction=core._two_qubit_reduction,num_time_slices=1)
init_parm = np.zeros(var_form.num_parameters)

# 4) Initialize VQE with statevector simulator and run

print("Running  VQE on statevector simulator ...")
optimizer 		 = CG(maxiter=100) # We chose the CG optimizer but you can choose COBYLA, SPSA, L_BFGS_B, ...
algo  			 = VQE(qubit_op,var_form,optimizer,aux_operators=aux_ops,initial_point=init_parm)
simulator        = Aer.get_backend("qasm_simulator")
backend_options = {"method": "statevector"}
quantum_instance = QuantumInstance(backend=simulator,backend_options=backend_options)
algo_result      = algo.run(quantum_instance)


# 5) Print the results

print("\nResults")

result = algo_result
naux = result['aux_operator_eigenvalues'].shape[0]
name = ['number','spin-squared','spin-z','energy']
	
for i in range(naux):
	if(name[i]=='energy'):
		result['aux_operator_eigenvalues'][i,0] += dE
	print("operator: ",name[i],result['aux_operator_eigenvalues'][i,0])
