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
    imports::{RUST_GATE, RUST_INSTRUCTION},
    operations::{Operation, Param, PyNativeOperation},
    packed_instruction::PackedOperation,
};
use pyo3::prelude::*;

pub trait CustomOperation: Operation + Any + Debug + Send + Sync {
    fn label(&self) -> Option<&str> {
        None
    }
    fn inverse(&self, _params: &[Param]) -> Option<(PackedOperation, SmallVec<[Param; 3]>)> {
        None
    }
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        None
    }
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        None
    }
    fn num_ctrl_qubits(&self) -> Option<NonZero<u32>> {
        None
    }
    fn is_controlled_gate(&self) -> bool {
        self.num_ctrl_qubits().is_some()
    }
    fn create_py_op<'py>(
        &self,
        py: Python<'py>,
        params: Option<SmallVec<[Param; 3]>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.clone_dyn().into();
        let py_class = PyNativeOperation::new(inner, params);
        match self.is_unitary() {
            true => {
                let custom_gate = RUST_GATE.get_bound(py);
                custom_gate.call1((py_class,))
            }
            false => {
                let custom_inst = RUST_INSTRUCTION.get_bound(py);
                custom_inst.call1((py_class,))
            }
        }
    }
    fn py_type<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, pyo3::types::PyType>> {
        match self.is_unitary() {
            true => Ok(RUST_GATE.get_bound(py).cast::<pyo3::types::PyType>()?),
            false => Ok(RUST_INSTRUCTION
                .get_bound(py)
                .cast::<pyo3::types::PyType>()?),
        }
    }
    fn clone_dyn(&self) -> Box<dyn CustomOperation>;
    fn is_unitary(&self) -> bool;
}

impl dyn CustomOperation + 'static {
    // Trait implementation needs to be repeated here as upcasting
    // is stabilized in Rust 1.86+ and we barely missed the cutoff.
    pub fn downcast_ref<T: CustomOperation + 'static>(&self) -> Option<&T> {
        (self.type_id() == TypeId::of::<T>()).then(|| unsafe { &*(self as *const _ as *const T) })
    }
}

#[cfg(test)]
mod test {
    use crate::Qubit;
    use crate::circuit_data::CircuitData;
    use crate::custom_operations::CustomOperation;
    use crate::gate_matrix::H_GATE;
    use crate::gate_matrix::rx_gate;

    use crate::operations::NativeOperation;
    use crate::operations::Operation;
    use crate::operations::OperationRef;
    use crate::operations::Param;
    use crate::operations::StandardGate;
    use ndarray::aview2;

    use pyo3::prelude::*;

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

    impl CustomOperation for CustomH {
        fn clone_dyn(&self) -> Box<dyn CustomOperation> {
            Box::new(self.clone())
        }
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

        fn is_unitary(&self) -> bool {
            true
        }
    }

    // Py Test
    #[derive(Debug, Clone)]
    pub struct CustomRX;

    impl Operation for CustomRX {
        fn name(&self) -> &str {
            "custom_rx"
        }

        fn num_qubits(&self) -> u32 {
            1
        }

        fn num_clbits(&self) -> u32 {
            0
        }

        fn num_params(&self) -> u32 {
            1
        }

        fn directive(&self) -> bool {
            false
        }
    }

    impl CustomOperation for CustomRX {
        fn clone_dyn(&self) -> Box<dyn CustomOperation> {
            Box::new(self.clone())
        }
        fn definition(&self, params: &[Param]) -> Option<CircuitData> {
            (params.len() == 1).then_some(
                CircuitData::from_standard_gates(
                    1,
                    [(
                        StandardGate::RX,
                        smallvec![params[0].clone()],
                        smallvec![Qubit(0)],
                    )],
                    0.0.into(),
                )
                .expect("Circuit should be built"),
            )
        }

        fn matrix(&self, params: &[Param]) -> Option<ndarray::Array2<numpy::Complex64>> {
            if params.len() == 1 {
                let Param::Float(param) = params[0] else {
                    return None;
                };
                Some(aview2(&rx_gate(param)).to_owned())
            } else {
                None
            }
        }

        fn is_unitary(&self) -> bool {
            true
        }
    }

    #[test]
    fn try_custom_h_gate() {
        let gate: Box<dyn CustomOperation> = Box::new(CustomH);

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

        let gate: NativeOperation = CustomH.into();

        // Try downcasting
        circuit
            .push_packed_operation(gate.clone().into(), None, &[Qubit(0)], &[])
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

    // #[cfg(all(not(miri), test))]
    #[test]
    // #[cfg(not(miri))]
    fn try_python_custom_gate() {
        let mut circuit = CircuitData::with_capacity(1, 0, 1, 0.0.into())
            .expect("Circuit with small capacity should be built.");

        let gate = NativeOperation::from(CustomRX);

        // Try downcasting
        circuit
            .push_packed_operation(
                gate.clone().into(),
                Some(crate::instruction::Parameters::Params(smallvec![
                    3.14.into()
                ])),
                &[Qubit(0)],
                &[],
            )
            .expect("Instruction should be added to the circuit.");
        circuit
            .push_packed_operation(
                NativeOperation::from(CustomH).into(),
                None,
                &[Qubit(0)],
                &[],
            )
            .expect("Instruction should be added to the circuit.");

        // Retrieve operation
        let retrieved_gate = &circuit.data()[0];
        let retrieved_h_gate = &circuit.data()[1];

        let OperationRef::CustomGate(gate_as_rx) = retrieved_gate.op.view() else {
            panic!("Gate should be a custom gate of type CustomH");
        };

        if gate_as_rx.downcast_ref::<CustomH>().is_some() {
            panic!("Gate should not be a custom gate of type CustomH");
        };

        let Some(_) = gate_as_rx.downcast_ref::<CustomRX>() else {
            panic!("Gate should be a custom gate of type CustomRX");
        };

        // Try Python:
        Python::attach(|py| -> PyResult<()> {
            let unpacked_operation = circuit.unpack_py_op(py, retrieved_gate)?.into_bound(py);
            println!("{}", unpacked_operation.repr()?);
            println!("{}", unpacked_operation.call_method0("to_matrix")?.repr()?);
            println!("{}", unpacked_operation.getattr("params")?.repr()?);
            println!("{}", unpacked_operation.getattr("definition")?.repr()?);

            let unpacked_operation_h = circuit.unpack_py_op(py, retrieved_h_gate)?.into_bound(py);
            println!("{}", unpacked_operation_h.repr()?);
            println!(
                "{}",
                unpacked_operation_h.call_method0("to_matrix")?.repr()?
            );
            println!("{}", unpacked_operation_h.getattr("params")?.repr()?);
            print!("{}", unpacked_operation_h.getattr("definition")?.repr()?);

            Ok(())
        })
        .expect("Something went wrong on the Python side.");
    }
}
