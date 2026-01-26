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

use std::sync::Arc;

use ndarray::Array2;
use numpy::Complex64;
use smallvec::SmallVec;

use crate::{
    circuit_data::CircuitData,
    imports::{GATE, INSTRUCTION},
    operations::{Operation, Param},
};
use pyo3::prelude::*;

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
            },
        }
    }

    /// Creates an instance of [OpaqueGate] with a label
    pub fn with_label(mut self, label: String) -> Self {
        self.instruction.label.replace(label);
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

    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<SmallVec<[Param; 3]>>,
    ) -> PyResult<Py<PyAny>> {
        let gate = GATE.get(py);
        gate.call1(py, (self.name(), self.num_qubits(), params, self.label()))
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
        }
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.label.replace(label);
        self
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<SmallVec<[Param; 3]>>,
    ) -> PyResult<Py<PyAny>> {
        let gate = INSTRUCTION.get(py);
        gate.call1(
            py,
            (
                self.name(),
                self.num_qubits(),
                self.num_clbits(),
                params.unwrap_or_default(),
                self.label(),
            ),
        )
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

/// A fully functional custom instruction
pub struct CustomInstruction {
    instruction: OpaqueInstruction,
    /// The definition of the instruction as a Circuit.
    definition: Option<Arc<CircuitData>>,
}

impl Operation for CustomInstruction {
    fn name(&self) -> &str {
        self.instruction.name()
    }

    fn num_qubits(&self) -> u32 {
        self.instruction.num_params()
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

impl CustomInstruction {
    pub fn new(
        name: String,
        num_qubits: u32,
        num_clbits: u32,
        num_params: u32,
        directive: bool,
    ) -> Self {
        Self {
            instruction: OpaqueInstruction {
                name,
                num_qubits,
                num_clbits,
                num_params,
                label: None,
                directive,
            },
            definition: None,
        }
    }

    pub fn with_definition<C>(mut self, definition: C) -> Self
    where
        C: Into<Arc<CircuitData>>,
    {
        let circuit = definition.into();

        // TODO: Implement results for the errors here.
        if circuit.num_qubits() != self.num_qubits() as usize {
            panic!("Number of qubits mismatched in definition.")
        }
        if circuit.num_clbits() != self.num_clbits() as usize {
            panic!("Number of clbits mismatched in definition.")
        }

        self.definition.replace(circuit);
        self
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.instruction.label.replace(label);
        self
    }

    pub fn definition(&self) -> Option<&CircuitData> {
        self.definition.as_deref()
    }

    pub fn label(&self) -> Option<&str> {
        self.instruction.label()
    }
}

type ParametricFunction<T> = Box<dyn Fn(&[Param]) -> Option<T>>;

enum MatrixType {
    Fixed(Array2<Complex64>),
    Parametric(ParametricFunction<Array2<Complex64>>),
}

/// A fully functional custom gate
pub struct CustomGate {
    gate: OpaqueGate,
    definition: Option<Arc<CircuitData>>,
    matrix_definition: Option<MatrixType>,
}

impl Operation for CustomGate {
    fn name(&self) -> &str {
        self.gate.name()
    }

    fn num_qubits(&self) -> u32 {
        self.gate.num_qubits()
    }

    fn num_clbits(&self) -> u32 {
        self.gate.num_clbits()
    }

    fn num_params(&self) -> u32 {
        self.gate.num_params()
    }

    fn directive(&self) -> bool {
        self.gate.directive()
    }
}

impl CustomGate {
    pub fn new(name: String, num_qubits: u32, num_params: u32) -> Self {
        Self {
            gate: OpaqueGate::new(name, num_qubits, num_params),
            definition: None,
            matrix_definition: None,
        }
    }

    pub fn with_fixed_matrix(mut self, matrix: Array2<Complex64>) -> Self {
        // TODO: Implement results for the errors here.
        if self.num_params() != 0 {
            panic!("Cannot add a fixed matrix to a parametric gate.");
        }
        self.matrix_definition.replace(MatrixType::Fixed(matrix));
        self
    }

    pub fn with_parametric_matrix<T>(mut self, matrix: T) -> Self
    where
        T: Fn(&[Param]) -> Option<Array2<Complex64>> + 'static,
    {
        // TODO: Implement results for the errors here.
        if self.num_params() == 0 {
            panic!("Cannot add a parametric matrix to a gate with no parameters.");
        }
        self.matrix_definition
            .replace(MatrixType::Parametric(Box::new(matrix)));
        self
    }

    pub fn with_definition<C>(mut self, definition: C) -> Self
    where
        C: Into<Arc<CircuitData>>,
    {
        let circuit = definition.into();

        // TODO: Implement results for the errors here.
        if circuit.num_qubits() != self.num_qubits() as usize {
            panic!("Number of qubits mismatched in definition.")
        }

        self.definition.replace(circuit);
        self
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.gate = self.gate.with_label(label);
        self
    }

    pub fn definition(&self) -> Option<&CircuitData> {
        self.definition.as_deref()
    }

    pub fn label(&self) -> Option<&str> {
        self.gate.label()
    }

    pub fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        match self.matrix_definition.as_ref() {
            Some(MatrixType::Fixed(matrix)) => params.is_empty().then_some(matrix.clone()),
            Some(MatrixType::Parametric(parametric)) => parametric(params),
            None => None,
        }
    }
}

pub trait Instruction: Operation {
    type InverseType: Clone;
    fn label(&self) -> Option<&str>;
    fn inverse(&self, params: &[Param]) -> Self::InverseType;
    fn definition(&self) -> Option<&CircuitData>;
}

pub trait Gate: Instruction {
    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>>;
}

#[cfg(test)]
mod test {
    use crate::Qubit;
    use crate::circuit_data::CircuitData;
    use crate::custom_operations::CustomGate;
    use crate::gate_matrix::H_GATE;
    use crate::operations::Operation;
    use crate::operations::Param;
    use crate::operations::StandardGate;
    use ndarray::aview2;
    use num_complex::c64;
    use smallvec::smallvec;
    use std::f64::consts::FRAC_1_SQRT_2;
    use std::f64::consts::PI;

    #[test]
    fn try_custom_h_gate() {
        let h_matrix = [
            [c64(FRAC_1_SQRT_2, 0.), c64(FRAC_1_SQRT_2, 0.)],
            [c64(FRAC_1_SQRT_2, 0.), c64(-FRAC_1_SQRT_2, 0.)],
        ];
        let gate = CustomGate::new("H".into(), 1, 0)
            .with_fixed_matrix(aview2(&h_matrix).to_owned())
            .with_definition(
                CircuitData::from_standard_gates(
                    1,
                    [(StandardGate::H, smallvec![], smallvec![Qubit(0)])],
                    0.0.into(),
                )
                .expect("Circuit should work"),
            );

        assert_eq!(
            gate.name(),
            "H",
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
            matrix_res,
            matrix_exp
        );

        let matrix_res = gate.matrix(&[Param::Float(PI)]);
        let matrix_exp = None;
        assert_eq!(
            matrix_res, matrix_exp,
            "Gate matrix did not match, expected {:?} obtained '{:?}'",
            matrix_res, matrix_exp
        );

        let circuit = gate.definition().expect("Circuit should exist.");
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
}
