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

use std::{
    any::{Any, TypeId},
    fmt::Debug,
    num::NonZero,
};

use ndarray::Array2;
use numpy::Complex64;
use smallvec::SmallVec;

use crate::{
    circuit_data::CircuitData,
    imports::{GATE, INSTRUCTION},
    operations::{Operation, Param},
    packed_instruction::PackedOperation,
};
use pyo3::{exceptions::PyNotImplementedError, prelude::*, types::PyType};

/// A gate instance that serves as a placeholder.
/// It contains properties such as the number of bits and parameters, and an optional label.
#[derive(Debug, Clone)]
pub struct OpaqueGate {
    /// The inner instruction instance.
    instruction: OpaqueInstruction,
}

impl OpaqueGate {
    /// Creates a new instance of [OpaqueGate]
    pub fn new(name: String, num_qubits: u32, num_params: u32) -> Self {
        Self {
            instruction: OpaqueInstruction {
                name,
                num_qubits,
                num_clbits: 0,
                num_params,
                label: None,
                directive: false,
                definition: None,
            },
        }
    }

    /// Creates an instance of [OpaqueGate] with a label
    pub fn with_label(mut self, label: String) -> Self {
        self.instruction.label.replace(label);
        self
    }

    pub fn with_fixed_definition(mut self, definition: CircuitData) -> Self {
        self.instruction = self.instruction.with_fixed_definition(definition);
        self
    }

    pub fn with_parametric_definition(mut self, definition: CircuitData) -> Self {
        self.instruction = self.instruction.with_parametric_definition(definition);
        self
    }

    /// Retrieves the set label of a Gate if applicable.
    pub fn label(&self) -> Option<&str> {
        self.instruction.label()
    }

    /// Since this
    pub fn matrix(&self) -> Option<Array2<Complex64>> {
        None
    }

    pub fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        self.instruction.definition(params)
    }

    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<SmallVec<[Param; 3]>>,
    ) -> PyResult<Py<PyAny>> {
        let gate_class = GATE.get(py);
        let definition = if let Some(params) = params.as_deref() {
            self.definition(params)
        } else {
            None
        };
        let gate = gate_class.call1(
            py,
            (
                self.name(),
                self.num_qubits(),
                self.num_clbits(),
                params.unwrap_or_default(),
                self.label(),
            ),
        )?;
        if let Some(definition) = definition {
            gate.setattr(py, "definition", definition)?;
        }
        Ok(gate)
    }
}

impl Operation for OpaqueGate {
    fn name(&self) -> &str {
        self.instruction.name()
    }

    fn num_qubits(&self) -> u32 {
        self.instruction.num_qubits()
    }

    fn num_clbits(&self) -> u32 {
        self.instruction.num_clbits()
    }

    fn num_params(&self) -> u32 {
        self.instruction.num_params()
    }

    fn directive(&self) -> bool {
        self.instruction.directive()
    }
}

#[derive(Debug, Clone)]
pub struct OpaqueInstruction {
    name: String,
    num_qubits: u32,
    num_clbits: u32,
    num_params: u32,
    label: Option<String>,
    directive: bool,
    definition: Option<CircuitData>,
}

impl OpaqueInstruction {
    pub fn new(
        name: String,
        num_qubits: u32,
        num_clbits: u32,
        num_params: u32,
        directive: bool,
    ) -> Self {
        Self {
            name,
            num_qubits,
            num_clbits,
            num_params,
            label: None,
            directive,
            definition: None,
        }
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.label.replace(label);
        self
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn with_fixed_definition(mut self, definition: CircuitData) -> Self {
        if self.num_params() != 0 {
            panic!(
                "Used a parametric circuit with different amount of free parameters than the instruction."
            );
        }
        if !definition.parameters().is_empty() {
            // TODO: Use Result.
            panic!(
                "Used a parametric circuit with free parameters for an instruction with no parameters."
            );
        }
        self.definition.replace(definition);
        self
    }

    pub fn with_parametric_definition(mut self, definition: CircuitData) -> Self {
        if self.num_params() as usize != definition.parameters().len() {
            panic!(
                "Used a parametric circuit with free parameters for an instruction with no parameters."
            );
        }
        self.definition.replace(definition);
        self
    }

    pub fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        let definition = self.definition.as_ref()?;
        if params.len() != definition.parameters().len() {
            return None;
        }
        // Clone only after we have checked that this definition has enough
        // free parameters.
        let mut definition = definition.clone();
        definition
            .assign_parameters_from_slice(params)
            .ok()
            .map(|_| definition)
    }

    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<SmallVec<[Param; 3]>>,
    ) -> PyResult<Py<PyAny>> {
        let instruction_class = INSTRUCTION.get(py);
        let definition = if let Some(params) = params.as_deref() {
            self.definition(params)
        } else {
            None
        };
        let instruction = instruction_class.call1(
            py,
            (
                self.name(),
                self.num_qubits(),
                self.num_clbits(),
                params.unwrap_or_default(),
                self.label(),
            ),
        )?;
        if let Some(definition) = definition {
            instruction.setattr(py, "definition", definition)?;
        }
        Ok(instruction)
    }
}

impl Operation for OpaqueInstruction {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    fn num_clbits(&self) -> u32 {
        self.num_clbits
    }

    fn num_params(&self) -> u32 {
        self.num_params
    }

    fn directive(&self) -> bool {
        self.directive
    }
}

// TODO: Change these to be Gate and Instruction instead of Opaque and Custom.
/// Describes a custom opaque Operation type.
#[derive(Debug, Clone)]
pub enum OpaqueOperation {
    Gate(OpaqueGate),
    Instruction(OpaqueInstruction),
}

impl From<OpaqueGate> for OpaqueOperation {
    fn from(value: OpaqueGate) -> Self {
        Self::Gate(value)
    }
}

impl From<OpaqueInstruction> for OpaqueOperation {
    fn from(value: OpaqueInstruction) -> Self {
        Self::Instruction(value)
    }
}

impl Operation for OpaqueOperation {
    fn name(&self) -> &str {
        match self {
            OpaqueOperation::Gate(opaque_gate) => opaque_gate.name(),
            OpaqueOperation::Instruction(opaque_instruction) => opaque_instruction.name(),
        }
    }

    fn num_qubits(&self) -> u32 {
        match self {
            OpaqueOperation::Gate(opaque_gate) => opaque_gate.num_qubits(),
            OpaqueOperation::Instruction(opaque_instruction) => opaque_instruction.num_qubits(),
        }
    }

    fn num_clbits(&self) -> u32 {
        match self {
            OpaqueOperation::Gate(opaque_gate) => opaque_gate.num_clbits(),
            OpaqueOperation::Instruction(opaque_instruction) => opaque_instruction.num_clbits(),
        }
    }

    fn num_params(&self) -> u32 {
        match self {
            OpaqueOperation::Gate(opaque_gate) => opaque_gate.num_params(),
            OpaqueOperation::Instruction(opaque_instruction) => opaque_instruction.num_params(),
        }
    }

    fn directive(&self) -> bool {
        match self {
            OpaqueOperation::Gate(opaque_gate) => opaque_gate.directive(),
            OpaqueOperation::Instruction(opaque_instruction) => opaque_instruction.directive(),
        }
    }
}

impl OpaqueOperation {
    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<SmallVec<[Param; 3]>>,
    ) -> PyResult<Py<PyAny>> {
        match self {
            OpaqueOperation::Gate(opaque_gate) => opaque_gate.create_py_op(py, params),
            OpaqueOperation::Instruction(opaque_instruction) => {
                opaque_instruction.create_py_op(py, params)
            }
        }
    }

    pub fn label(&self) -> Option<&str> {
        match self {
            OpaqueOperation::Gate(opaque_gate) => opaque_gate.label(),
            OpaqueOperation::Instruction(opaque_instruction) => opaque_instruction.label(),
        }
    }

    pub fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        match self {
            OpaqueOperation::Gate(opaque_gate) => opaque_gate.definition(params),
            OpaqueOperation::Instruction(opaque_instruction) => {
                opaque_instruction.definition(params)
            }
        }
    }
}

pub trait BaseOperation: Operation + Any + Debug {
    fn create_py_op(
        &self,
        _py: Python,
        _params: Option<SmallVec<[Param; 3]>>,
    ) -> PyResult<Bound<PyAny>> {
        Err(PyNotImplementedError::new_err(format!(
            "There is currently no Python implementation for operation '{}'",
            self.name()
        )))
    }
    fn py_type(&self, _py: Python) -> PyResult<Bound<PyType>> {
        Err(PyNotImplementedError::new_err(format!(
            "There is currently no Python implementation for operation '{}'",
            self.name()
        )))
    }
}

pub trait CustomOperation: BaseOperation {
    fn clone_dyn(&self) -> Box<dyn CustomInstruction>;
}

impl dyn CustomOperation + 'static {
    pub fn downcast_ref<T: CustomOperation + 'static>(&self) -> Option<&T> {
        (self.type_id() == TypeId::of::<T>()).then(|| unsafe { &*(self as *const _ as *const T) })
    }
}

macro_rules! instruction_methods {
    () => {
        fn label(&self) -> Option<&str> {
            None
        }
        fn inverse(&self, _params: &[Param]) -> Option<(PackedOperation, SmallVec<[Param; 3]>)> {
            None
        }
        fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
            None
        }
    };
}
pub trait CustomInstruction: Operation + Debug + Any + BaseOperation {
    instruction_methods! {}
    fn clone_dyn(&self) -> Box<dyn CustomInstruction>;
}

impl dyn CustomInstruction + 'static {
    // Trait implementation needs to be repeated here as upcasting
    // is stabilized in Rust 1.86+ and we barely missed the cutoff.
    pub fn downcast_ref<T: CustomInstruction + 'static>(&self) -> Option<&T> {
        (self.type_id() == TypeId::of::<T>()).then(|| unsafe { &*(self as *const _ as *const T) })
    }
}

pub trait CustomGate: Operation + Debug + Any + BaseOperation {
    instruction_methods! {}
    fn clone_dyn(&self) -> Box<dyn CustomGate>;
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        None
    }
    fn num_ctrl_qubits(&self) -> Option<NonZero<u32>> {
        None
    }
    fn is_controlled_gate(&self) -> bool {
        self.num_ctrl_qubits().is_some()
    }
}

impl dyn CustomGate + 'static {
    // Trait implementation needs to be repeated here as upcasting
    // is stabilized in Rust 1.86+ and we barely missed the cutoff.
    pub fn downcast_ref<T: CustomGate + 'static>(&self) -> Option<&T> {
        (self.type_id() == TypeId::of::<T>()).then(|| unsafe { &*(self as *const _ as *const T) })
    }
}

#[cfg(test)]
mod test {
    use crate::Qubit;
    use crate::circuit_data::CircuitData;
    use crate::custom_operations::BaseOperation;
    use crate::custom_operations::CustomGate;
    use crate::gate_matrix::H_GATE;
    use crate::operations::Operation;
    use crate::operations::OperationRef;
    use crate::operations::Param;
    use crate::operations::StandardGate;
    use ndarray::aview2;
    use smallvec::smallvec;

    use std::f64::consts::PI;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct CustomH;
    impl Operation for CustomH {
        fn name(&self) -> &str {
            "h"
        }

        fn num_qubits(&self) -> u32 {
            1
        }

        fn num_clbits(&self) -> u32 {
            0
        }

        fn num_params(&self) -> u32 {
            0
        }

        fn directive(&self) -> bool {
            false
        }
    }

    impl BaseOperation for CustomH {}
    impl CustomGate for CustomH {
        fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
            CircuitData::from_standard_gates(
                1,
                [(StandardGate::H, smallvec![], smallvec![Qubit(0)])],
                0.0.into(),
            )
            .ok()
        }

        fn matrix(&self, params: &[Param]) -> Option<ndarray::Array2<numpy::Complex64>> {
            params.is_empty().then_some(aview2(&H_GATE).to_owned())
        }

        fn clone_dyn(&self) -> Box<dyn CustomGate> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn try_custom_h_gate() {
        let gate: Box<dyn CustomGate> = Box::new(CustomH);

        // Try downcasting
        let gate = gate
            .downcast_ref::<CustomH>()
            .expect("Should downcast to an H gate");

        assert_eq!(
            gate.name(),
            "h",
            "Gate names did not match, expected 'H' obtained '{}'",
            gate.name()
        );
        assert_eq!(
            gate.num_qubits(),
            1,
            "Gate num_qubits did not match, expected '1' obtained '{}'",
            gate.num_qubits()
        );
        assert_eq!(
            gate.num_params(),
            0,
            "Gate num_qubits did not match, expected '0' obtained '{}'",
            gate.num_params()
        );
        assert_eq!(
            gate.label(),
            None,
            "Gate labels did not match, expected 'None' obtained '{:?}'",
            gate.label()
        );
        let matrix_res = gate.matrix(&[]);
        let matrix_exp = Some(aview2(&H_GATE));
        assert_eq!(
            matrix_res.as_ref().map(|mat| mat.view()),
            matrix_exp,
            "Gate matrix did not match, expected {:?} obtained '{:?}'",
            matrix_exp,
            matrix_res,
        );

        let matrix_res = gate.matrix(&[Param::Float(PI)]);
        let matrix_exp = None;
        assert_eq!(
            matrix_res, matrix_exp,
            "Gate matrix did not match, expected {:?} obtained '{:?}'",
            matrix_exp, matrix_res
        );

        let circuit = gate.definition(&[]).expect("Circuit should exist.");
        assert_eq!(
            circuit.__len__(),
            1,
            "Definition length mismatch, expected {} got {}.",
            1,
            circuit.__len__()
        );

        let hgate = circuit.iter().next().expect("Should be H gate");
        assert_eq!(
            hgate.op.standard_gate(),
            StandardGate::H,
            "Definition length mismatch, expected {:?} got {:?}.",
            hgate.op.standard_gate(),
            StandardGate::H
        );
    }

    #[test]
    fn try_add_to_circuit() {
        let mut circuit = CircuitData::with_capacity(1, 0, 1, 0.0.into())
            .expect("Circuit with small capacity should be built.");

        let gate: Box<dyn CustomGate> = Box::new(CustomH);

        // Try downcasting
        circuit
            .push_packed_operation(gate.clone_dyn().into(), None, &[Qubit(0)], &[])
            .expect("Instruction should be added to the circuit.");

        // Retrieve operation
        let retrieved_gate = &circuit.data()[0];

        let OperationRef::CustomGate(gate_as_h) = retrieved_gate.op.view() else {
            panic!("Gate should be a custom gate of type CustomH");
        };

        let Some(downcast_gate) = gate_as_h.downcast_ref::<CustomH>() else {
            panic!("Gate should be a custom gate of type CustomH");
        };

        assert_eq!(gate.downcast_ref::<CustomH>(), Some(downcast_gate))
    }
}
