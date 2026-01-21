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


use ndarray::Array2;
use numpy::Complex64;
use smallvec::SmallVec;

use crate::{
    imports::{GATE, INSTRUCTION},
    operations::{Operation, Param},
};
use pyo3::prelude::*;

#[derive(Debug, Clone)]
pub struct OpaqueGate {
    instruction: OpaqueInstruction,
}

impl OpaqueGate {
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

    pub fn with_label(mut self, label: String) -> Self {
        self.instruction.label.replace(label);
        self
    }

    pub fn label(&self) -> Option<&str> {
        self.instruction.label()
    }

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

/// Describes a custom opaque Operation type.
#[derive(Debug, Clone)]
pub enum OpaqueOperation {
    Gate(OpaqueGate),
    Instruction(OpaqueInstruction),
}
