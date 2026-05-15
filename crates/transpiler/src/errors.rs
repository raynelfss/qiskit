// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::PyErr;
use qiskit_circuit::{circuit_data::CircuitDataError, dag_circuit::DAGCircuitInnerError};
use thiserror::Error;

use crate::{
    passes::errors::{
        BasisTranslatorError, CommutationAnalysisError, CommutationCancelError,
        ConsolidateBlocksError, ConstrainedRescheduleError, DisjointLayoutError,
        InstructionDurationCheckError, RemoveIdentityEquivError, Split2QUnitariesError,
        TranspilerError, UnitarySynthesisError,
    },
    target::TargetError,
};

/// Collection of errors that can happen within a transpiler pass.
#[derive(Debug, Error)]
pub enum NativeTranspilerError {
    // Errors related to CircuitData
    #[error(transparent)]
    Circuit(#[from] CircuitDataError),
    // Errors related to DAGCircuit
    #[error(transparent)]
    DAGCircuit(#[from] DAGCircuitInnerError),
    // Error coming from DisjointLayout
    #[error(transparent)]
    DisjointLayout(#[from] DisjointLayoutError),
    // Error coming from BasisTranslator
    #[error(transparent)]
    BasisTranslator(#[from] BasisTranslatorError),
    // CommutationAnalysis errors
    #[error(transparent)]
    CommutationAnalysis(#[from] CommutationAnalysisError),
    // CommutativeCancelation errors,
    #[error(transparent)]
    CommutationCancel(#[from] CommutationCancelError),
    // ConstrainedReschedule errors,
    #[error(transparent)]
    ConstrainedReschedule(#[from] ConstrainedRescheduleError),
    // ConsolidateBlocks errors,
    #[error(transparent)]
    ConsolidateBlocks(#[from] ConsolidateBlocksError),
    // InstructionDurationCheck errors,
    #[error(transparent)]
    InstructionDurationCheck(#[from] InstructionDurationCheckError),
    /// RemoveIdentityEquiv error
    #[error(transparent)]
    RemoveIdentityEquiv(#[from] RemoveIdentityEquivError),
    /// Split2QUnitaries error
    #[error(transparent)]
    Split2QUnitaries(#[from] Split2QUnitariesError),
    // UnitarySynthesis error
    #[error(transparent)]
    UnitarySynthesis(#[from] UnitarySynthesisError),
    // Target error
    #[error(transparent)]
    Target(#[from] TargetError),
    // Generic transpiler error with a message
    #[error(transparent)]
    Generic(#[from] anyhow::Error),
    // Python error
    #[error(transparent)]
    Python(PyErr),
}

impl From<NativeTranspilerError> for PyErr {
    fn from(value: NativeTranspilerError) -> Self {
        match value {
            NativeTranspilerError::Circuit(circuit_data_error) => circuit_data_error.into(),
            NativeTranspilerError::BasisTranslator(basis_translator_error) => {
                basis_translator_error.into()
            }
            NativeTranspilerError::Generic(error) => TranspilerError::new_err(error.to_string()),
            NativeTranspilerError::CommutationAnalysis(commutation_error) => {
                commutation_error.into()
            }
            NativeTranspilerError::DisjointLayout(disjoint_layout_error) => {
                disjoint_layout_error.into()
            }
            NativeTranspilerError::CommutationCancel(commutation_cancel_error) => {
                commutation_cancel_error.into()
            }
            NativeTranspilerError::ConstrainedReschedule(constrained_reschedule_error) => {
                constrained_reschedule_error.into()
            }
            NativeTranspilerError::ConsolidateBlocks(consolidate_blocks_error) => {
                consolidate_blocks_error.into()
            }
            NativeTranspilerError::DAGCircuit(dagcircuit_inner_error) => {
                dagcircuit_inner_error.into()
            }
            NativeTranspilerError::Python(py_err) => py_err,
            NativeTranspilerError::InstructionDurationCheck(instruction_duration_check_error) => {
                instruction_duration_check_error.into()
            }
            NativeTranspilerError::RemoveIdentityEquiv(remove_identity_equiv_error) => {
                remove_identity_equiv_error.into()
            }
            NativeTranspilerError::Split2QUnitaries(split2_qunitaries_error) => {
                split2_qunitaries_error.into()
            }
            NativeTranspilerError::UnitarySynthesis(unitary_synthesis_error) => {
                unitary_synthesis_error.into()
            }
            NativeTranspilerError::Target(target_error) => target_error.into(),
        }
    }
}
