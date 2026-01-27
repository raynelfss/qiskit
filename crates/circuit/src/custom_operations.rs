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

use std::any::{Any, TypeId};

use ndarray::Array2;
use numpy::Complex64;
use smallvec::SmallVec;

use crate::{
    circuit_data::CircuitData,
    imports::{GATE, INSTRUCTION},
    operations::{Operation, Param},
    packed_instruction::PackedOperation,
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

pub trait CustomOperation: Operation + Any {
    fn clone_dyn(&self) -> Box<dyn CustomOperation>;
}

impl dyn CustomOperation + 'static {
    pub fn downcast_ref<T: CustomOperation + 'static>(&self) -> Option<&T> {
        (self.type_id() == TypeId::of::<T>()).then(|| unsafe { &*(self as *const _ as *const T) })
    }
}

pub trait CustomInstruction: CustomOperation {
    fn label(&self) -> Option<&str>;
    fn inverse(&self, params: &[Param]) -> Option<(PackedOperation, SmallVec<[Param; 3]>)>;
    fn definition(&self) -> Option<CircuitData>;
}

impl dyn CustomInstruction + 'static {
    // Trait implementation needs to be repeated here as upcasting
    // is stabilized in Rust 1.86+ and we barely missed the cutoff.
    pub fn downcast_ref<T: CustomInstruction + 'static>(&self) -> Option<&T> {
        (self.type_id() == TypeId::of::<T>()).then(|| unsafe { &*(self as *const _ as *const T) })
    }
}

pub trait CustomGate: CustomInstruction {
    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>>;
    fn num_ctrl_qubits(&self) -> u32;
    fn is_controlled_gate(&self) -> bool {
        self.num_ctrl_qubits() > 0
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
    use crate::custom_operations::CustomGate;
    use crate::custom_operations::CustomInstruction;
    use crate::custom_operations::CustomOperation;
    use crate::gate_matrix::H_GATE;
    use crate::operations::Operation;
    use crate::operations::Param;
    use crate::operations::StandardGate;
    use ndarray::aview2;
    use smallvec::smallvec;

    use std::f64::consts::PI;

    #[test]
    fn try_custom_h_gate() {
        #[derive(Debug, Clone)]
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

        impl CustomOperation for CustomH {
            fn clone_dyn(&self) -> Box<dyn CustomOperation> {
                Box::new(self.clone())
            }
        }

        impl CustomInstruction for CustomH {
            fn label(&self) -> Option<&str> {
                None
            }

            fn definition(&self) -> Option<CircuitData> {
                CircuitData::from_standard_gates(
                    1,
                    [(StandardGate::H, smallvec![], smallvec![Qubit(0)])],
                    0.0.into(),
                )
                .ok()
            }

            fn inverse(
                &self,
                _params: &[Param],
            ) -> Option<(
                crate::packed_instruction::PackedOperation,
                smallvec::SmallVec<[Param; 3]>,
            )> {
                None
            }
        }

        impl CustomGate for CustomH {
            fn matrix(&self, _params: &[Param]) -> Option<ndarray::Array2<numpy::Complex64>> {
                _params.is_empty().then_some(aview2(&H_GATE).to_owned())
            }

            fn num_ctrl_qubits(&self) -> u32 {
                0
            }
        }

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
            matrix_res,
            matrix_exp
        );

        let matrix_res = gate.matrix(&[Param::Float(PI)]);
        let matrix_exp = None;
        assert_eq!(
            matrix_res, matrix_exp,
            "Gate matrix did not match, expected {:?} obtained '{:?}'",
            matrix_exp, matrix_res
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
