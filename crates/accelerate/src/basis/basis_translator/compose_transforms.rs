// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::{HashMap, HashSet};
use pyo3::{exceptions::PyTypeError, prelude::*};
use qiskit_circuit::converters::circuit_to_dag;
use qiskit_circuit::imports::{GATE, PARAMETER_VECTOR, QUANTUM_CIRCUIT, QUANTUM_REGISTER};
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::parameter_table::ParameterUuid;
use qiskit_circuit::{
    circuit_data::CircuitData,
    dag_circuit::{DAGCircuit, NodeType},
    operations::{Operation, Param},
};
use smallvec::SmallVec;

// Custom types
// TODO: Remove these and use the version from `basis_search`
pub type BasisTransforms = Vec<(String, u32, SmallVec<[Param; 3]>, CircuitRep)>;
pub type MappedTransforms = HashMap<(String, u32), (SmallVec<[Param; 3]>, DAGCircuit)>;
// TODO: Remove these and use the version from `EquivalenceLibrary`

/// Representation of QuantumCircuit which the original circuit object + an
/// instance of `CircuitData`.
#[derive(Debug, Clone)]
pub struct CircuitRep(pub CircuitData);

impl FromPyObject<'_> for CircuitRep {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if ob.is_instance(QUANTUM_CIRCUIT.get_bound(ob.py()))? {
            let data: Bound<PyAny> = ob.getattr("_data")?;
            let data_downcast: Bound<CircuitData> = data.downcast_into()?;
            let data_extract: CircuitData = data_downcast.extract()?;
            Ok(Self(data_extract))
        } else {
            Err(PyTypeError::new_err(
                "Provided object was not an instance of QuantumCircuit",
            ))
        }
    }
}

impl IntoPy<PyObject> for CircuitRep {
    fn into_py(self, py: Python<'_>) -> PyObject {
        QUANTUM_CIRCUIT
            .get_bound(py)
            .call_method1("_from_circuit_data", (self.0,))
            .unwrap()
            .unbind()
    }
}

impl ToPyObject for CircuitRep {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.clone().into_py(py)
    }
}

#[pyfunction(name = "compose_transforms")]
pub(super) fn py_compose_transforms(
    py: Python,
    basis_transforms: BasisTransforms,
    source_basis: HashSet<(String, u32)>,
    source_dag: &DAGCircuit,
) -> PyResult<MappedTransforms> {
    compose_transforms(py, &basis_transforms, &source_basis, source_dag)
}

pub(super) fn compose_transforms<'a>(
    py: Python,
    basis_transforms: &'a BasisTransforms,
    source_basis: &'a HashSet<(String, u32)>,
    source_dag: &'a DAGCircuit,
) -> PyResult<MappedTransforms> {
    let example_gates = *get_example_gates(py, source_dag, None)?;
    let mut mapped_instructions: MappedTransforms = HashMap::new();

    for (gate_name, gate_num_qubits) in source_basis.iter().cloned() {
        // Need to grab a gate instance to find num_qubits and num_params.
        // Can be removed following https://github.com/Qiskit/qiskit-terra/pull/3947 .
        let example_gate = &example_gates[&(gate_name.clone(), gate_num_qubits)];
        let num_params = example_gate.params_view().len();

        let placeholder_params: SmallVec<[Param; 3]> = PARAMETER_VECTOR
            .get_bound(py)
            .call1((&gate_name, num_params))?
            .extract()?;

        let mut dag = DAGCircuit::new(py)?;
        // Create the mock gate and add to the circuit, use Python for this.
        let qubits = QUANTUM_REGISTER.get_bound(py).call1((gate_num_qubits,))?;
        dag.add_qreg(py, &qubits)?;

        let gate = GATE.get_bound(py).call1((
            &gate_name,
            gate_num_qubits,
            placeholder_params
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<SmallVec<[Param; 3]>>(),
        ))?;

        dag.py_apply_operation_back(py, gate, Some(qubits.extract()?), None, false)?;
        mapped_instructions.insert(
            (gate_name.clone(), gate_num_qubits),
            (placeholder_params, dag),
        );

        for (gate_name, gate_num_qubits, equiv_params, equiv) in basis_transforms {
            for ((_mapped_instr_name, _), (_dag_params, dag)) in &mut mapped_instructions {
                let doomed_nodes = dag
                    .op_nodes(true)
                    .filter_map(|node| {
                        if let Some(NodeType::Operation(op)) = dag.dag().node_weight(node) {
                            if (gate_name.as_str(), *gate_num_qubits)
                                == (op.op.name(), op.op.num_qubits())
                            {
                                Some((
                                    node,
                                    op.params_view()
                                        .iter()
                                        .map(|x| x.clone_ref(py))
                                        .collect::<SmallVec<[Param; 3]>>(),
                                ))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                for (node, params) in doomed_nodes {
                    let param_mapping: HashMap<ParameterUuid, Param> = equiv_params
                        .iter()
                        .map(|x| ParameterUuid::from_parameter(x.to_object(py).bind(py)))
                        .zip(params)
                        .map(|(uuid, param)| -> PyResult<(ParameterUuid, Param)> {
                            Ok((uuid?, param.clone_ref(py)))
                        })
                        .collect::<PyResult<_>>()?;
                    let mut replacement = equiv.clone();
                    replacement
                        .0
                        .assign_parameters_from_mapping(py, param_mapping)?;
                    let replace_dag: DAGCircuit =
                        DAGCircuit::from_circuit_data(py, replacement.0, true)?;
                    let op_node = dag.get_node(py, node)?;
                    dag.substitute_node_with_dag(py, op_node.bind(py), &replace_dag, None, true)?;
                }
            }
        }
    }
    Ok(mapped_instructions)
}

fn get_example_gates(
    py: Python,
    dag: &DAGCircuit,
    example_gates: Option<Box<HashMap<(String, u32), PackedInstruction>>>,
) -> PyResult<Box<HashMap<(String, u32), PackedInstruction>>> {
    let mut example_gates = example_gates.unwrap_or_default();
    for node in dag.op_nodes(true) {
        if let Some(NodeType::Operation(op)) = dag.dag().node_weight(node) {
            example_gates.insert((op.op.name().to_string(), op.op.num_qubits()), op.clone());
            if op.op.control_flow() {
                let OperationRef::Instruction(inst) = op.op.view() else {
                    panic!("Control flow op can only be an instruction")
                };
                let inst_bound = inst.instruction.bind(py);
                let blocks = inst_bound.getattr("blocks")?;
                for block in blocks.iter()? {
                    let block_as_dag = circuit_to_dag(py, block?.extract()?, true, None, None)?;
                    example_gates = get_example_gates(py, &block_as_dag, Some(example_gates))?;
                }
            }
        }
    }
    Ok(example_gates)
}