// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::TranspilerError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::{DelayUnit, OperationRef, StandardInstruction};

#[derive(Debug, thiserror::Error)]
pub enum InstructionDurationCheckError {
    #[error("Delay instruction missing duration parameter")]
    MissingDuration,
    #[error("Delay duration must have dt unit for checking alignment.")]
    NoDTUnit,
    #[error("The provided Delay duration is not in terms of dt.")]
    IncorrectDurationUnit,
    #[error(transparent)]
    PythonIntExtraction(PyErr),
}

impl From<InstructionDurationCheckError> for PyErr {
    fn from(value: InstructionDurationCheckError) -> Self {
        match value {
            InstructionDurationCheckError::MissingDuration => {
                PyValueError::new_err(value.to_string())
            }
            InstructionDurationCheckError::NoDTUnit
            | InstructionDurationCheckError::IncorrectDurationUnit => {
                TranspilerError::new_err(value.to_string())
            }
            InstructionDurationCheckError::PythonIntExtraction(py_err) => py_err,
        }
    }
}

/// Run duration validation passes.
///
/// Args:
///     dag: DAG circuit to check instruction durations.
///     acquire_align: Integer number representing the minimum time resolution to
///         trigger acquisition instruction in units of dt.
///     pulse_align: Integer number representing the minimum time resolution to
///         trigger gate instruction in units of ``dt``.
/// Returns:
///     True if rescheduling is required, False otherwise.
#[pyfunction]
#[pyo3(name="run_instruction_duration_check", signature=(dag, acquire_align, pulse_align))]
fn py_run_instruction_duration_check(
    dag: &DAGCircuit,
    acquire_align: u32,
    pulse_align: u32,
) -> PyResult<bool> {
    run_instruction_duration_check(dag, acquire_align, pulse_align).map_err(Into::into)
}

pub fn run_instruction_duration_check(
    dag: &DAGCircuit,
    acquire_align: u32,
    pulse_align: u32,
) -> Result<bool, InstructionDurationCheckError> {
    let num_stretches = dag.num_stretches();

    // Rescheduling is not necessary
    if (acquire_align == 1 && pulse_align == 1) || num_stretches != 0 {
        return Ok(false);
    }

    // Check delay durations
    for (_, packed_op) in dag.op_nodes(false) {
        if let OperationRef::StandardInstruction(StandardInstruction::Delay(unit)) =
            packed_op.op.view()
        {
            let params = packed_op.params_view();
            let param = params
                .first()
                .ok_or_else(|| InstructionDurationCheckError::MissingDuration)?;

            if unit != DelayUnit::DT {
                return Err(InstructionDurationCheckError::NoDTUnit);
            }
            let duration = match param {
                Param::Obj(val) => Python::attach(|py| val.bind(py).extract::<u32>())
                    .map_err(InstructionDurationCheckError::PythonIntExtraction),
                _ => Err(InstructionDurationCheckError::IncorrectDurationUnit),
            }?;

            if !(duration % acquire_align == 0 || duration % pulse_align == 0) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub fn instruction_duration_check_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_instruction_duration_check))?;
    Ok(())
}
