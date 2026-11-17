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

use smallvec::smallvec;
use std::f64::consts::PI;

use qiskit_circuit::Qubit;
use qiskit_circuit::bit::QuantumRegister;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGError};
use qiskit_circuit::operations::{
    CustomOperation, Operation, Param, StandardGate, multiply_param, radd_param,
};
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;

#[derive(Clone, Debug, PartialEq)]
pub struct EfficientlyControlledGate {
    base_gate: StandardGate,
    control_state: u32,
    num_ctrl_bits: u32,
    name: String,
}

impl EfficientlyControlledGate {
    pub fn new(base: StandardGate, control_state: &[bool]) -> Option<Self> {
        if !EFFICIENTLY_CONTROLLED_GATES.contains(&base.name()) {
            return None;
        }
        let mut num_ctrl_bits = control_state.len() as u32;
        if num_ctrl_bits > 32 {
            panic!("State exceeds allowed length")
        };

        let base_name = match base {
            StandardGate::CH => StandardGate::H.name(),
            StandardGate::CX => StandardGate::X.name(),
            StandardGate::CY => StandardGate::Y.name(),
            StandardGate::CZ => StandardGate::Z.name(),
            StandardGate::CPhase => StandardGate::Phase.name(),
            StandardGate::CRX => StandardGate::RX.name(),
            StandardGate::CRY => StandardGate::RY.name(),
            StandardGate::CRZ => StandardGate::RZ.name(),
            StandardGate::CS => StandardGate::S.name(),
            StandardGate::CSdg => StandardGate::Sdg.name(),
            StandardGate::CSX => StandardGate::SX.name(),
            StandardGate::CU => StandardGate::U.name(),
            StandardGate::CU1 => StandardGate::U1.name(),
            StandardGate::CU3 => StandardGate::U2.name(),
            StandardGate::CCX => StandardGate::X.name(),
            StandardGate::CCZ => StandardGate::Z.name(),
            StandardGate::CSwap => StandardGate::Swap.name(),
            StandardGate::C3X => StandardGate::X.name(),
            StandardGate::C3SX => StandardGate::SX.name(),
            _ => base.name(),
        };

        if base.is_controlled_gate() {
            num_ctrl_bits += base.num_ctrl_qubits();
        }

        let name = if num_ctrl_bits > 2 {
            format!("c{}{}", num_ctrl_bits, base_name)
        } else {
            let cs: String = (0..num_ctrl_bits).map(|_| "c").collect();
            format!("{}{}", cs, base_name)
        };

        // Convert to integer, use big endian.
        let control_state = control_state
            .iter()
            .fold(0, |acc, bin| acc << 1 | *bin as u32);
        Some(Self {
            base_gate: base,
            control_state,
            num_ctrl_bits,
            name,
        })
    }
}

impl Operation for EfficientlyControlledGate {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> u32 {
        self.num_ctrl_bits + self.base_gate.num_qubits()
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        self.base_gate.num_params()
    }

    fn directive(&self) -> bool {
        self.base_gate.directive()
    }
}

impl CustomOperation for EfficientlyControlledGate {
    fn is_unitary(&self) -> bool {
        true
    }

    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        if params.len() as u32 != self.num_params() {
            return None;
        }
        let target_num_qubits = self.base_gate.num_qubits();
        let mut controlled_definition = CircuitData::new(None, None, 0.0.into())
            .expect("Empty circuit should be able to build.");

        let q_control = QuantumRegister::new_owning("control", self.num_ctrl_bits);
        let q_target = QuantumRegister::new_owning("target", target_num_qubits);

        for reg in [q_control, q_target] {
            controlled_definition
                .add_qreg(reg, true)
                .expect("Registers should not exist in circuit.");
        }

        apply_basic_controlled_gate(&mut controlled_definition, &self.base_gate, params).ok()?;

        Some(controlled_definition)
    }

    fn num_ctrl_qubits(&self) -> Option<std::num::NonZero<u32>> {
        self.num_ctrl_bits.try_into().ok()
    }
}

const EFFICIENTLY_CONTROLLED_GATES: [&str; 13] = [
    "p", "u", "x", "z", "y", "h", "sx", "sxdg", "rx", "ry", "rz", "cx", "cz",
];

fn synth_mcrx(param: Param, num_ctrl_qubits: u32) -> Result<CircuitData, CircuitDataError> {
    if num_ctrl_qubits == 0 {
        CircuitData::from_standard_gates(
            1,
            [(StandardGate::RX, smallvec![param], smallvec![Qubit(0)])],
            0.0.into(),
        )
    } else if num_ctrl_qubits == 1 {
        CircuitData::from_standard_gates(
            2,
            [(
                StandardGate::CU,
                smallvec![param, (-PI / 2.0).into(), (PI / 2.0).into()],
                (0..(num_ctrl_qubits + 1)).map(Qubit).collect(),
            )],
            0.0.into(),
        )
    } else {
        mcu_graycode_circ(
            num_ctrl_qubits,
            &[param, (-PI / 2.0).into(), (PI / 2.0).into()],
        )
    }
}

fn synth_mcry(param: Param, num_ctrl_qubits: u32) -> Result<CircuitData, CircuitDataError> {
    if num_ctrl_qubits == 0 {
        CircuitData::from_standard_gates(
            1,
            [(StandardGate::RY, smallvec![param], smallvec![Qubit(0)])],
            0.0.into(),
        )
    } else if num_ctrl_qubits == 1 {
        CircuitData::from_standard_gates(
            2,
            [(
                StandardGate::CU,
                smallvec![param, 0.0.into(), 0.0.into()],
                (0..(num_ctrl_qubits + 1)).map(Qubit).collect(),
            )],
            0.0.into(),
        )
    } else {
        mcu_graycode_circ(num_ctrl_qubits, &[param, 0.0.into(), 0.0.into()])
    }
}

fn synth_mcrz(param: Param, num_ctrl_qubits: u32) -> Result<CircuitData, CircuitDataError> {
    if num_ctrl_qubits == 0 {
        CircuitData::from_standard_gates(
            1,
            [(StandardGate::RZ, smallvec![param], smallvec![Qubit(0)])],
            0.0.into(),
        )
    } else if num_ctrl_qubits == 1 {
        let lam_2 = match param {
            Param::ParameterExpression(expr) => {
                Param::ParameterExpression(expr.div(&ParameterExpression::from_f64(2.0))?.into())
            }
            Param::Float(lam) => (lam / 2.0).into(),
            Param::Int(_) | Param::Obj(_) => {
                panic!("Int and obj are not supported by standard gates.")
            }
        };
        let lam_2_neg = match &lam_2 {
            Param::ParameterExpression(expr) => Param::ParameterExpression(expr.neg().into()),
            Param::Float(lam) => (*lam * -1.0).into(),
            Param::Int(_) | Param::Obj(_) => {
                panic!("Int and obj are not supported by standard gates.")
            }
        };

        CircuitData::from_standard_gates(
            2,
            [
                (
                    StandardGate::U,
                    smallvec![0.0.into(), 0.0.into(), lam_2],
                    smallvec![Qubit(num_ctrl_qubits + 1)],
                ),
                (
                    StandardGate::CX,
                    smallvec![],
                    (0..(num_ctrl_qubits + 1)).map(Qubit).collect(),
                ),
                (
                    StandardGate::U,
                    smallvec![0.0.into(), 0.0.into(), lam_2_neg],
                    smallvec![Qubit(num_ctrl_qubits + 1)],
                ),
                (
                    StandardGate::CX,
                    smallvec![],
                    (0..(num_ctrl_qubits + 1)).map(Qubit).collect(),
                ),
            ],
            0.0.into(),
        )
    } else {
        mcsu2_real_diagonal_rz(param, num_ctrl_qubits)
    }
}

fn mcsu2_real_diagonal_rz(
    param: Param,
    num_ctrl_qubits: u32,
) -> Result<CircuitData, CircuitDataError> {
    let theta = match param {
        Param::ParameterExpression(expr) => {
            Param::ParameterExpression(expr.neg().div(&ParameterExpression::from_f64(4.0))?.into())
        }
        Param::Float(theta) => (-theta / 4.0).into(),
        Param::Int(_) | Param::Obj(_) => panic!("Int and obj are not supported by standard gates."),
    };
    let (rz_inverse, rz_inverse_params) = StandardGate::RZ
        .inverse(std::slice::from_ref(&theta))
        .expect("RZ inverse should work.");

    let k_1 = num_ctrl_qubits.div_ceil(2) as usize;
    let k_2 = (num_ctrl_qubits / 2) as usize;

    let controls: Vec<Qubit> = (0..num_ctrl_qubits).map(Qubit).collect();
    let target = Qubit(num_ctrl_qubits);

    let mcx1 = crate::multi_controlled::synth_mcx_n_dirty_i15(k_1, false, false)?;
    let mcx1_num_ancillas = mcx1.num_qubits() - k_1 - 1;
    let mcx_1_qubits: Vec<Qubit> = (0..k_1)
        .map(|idx| controls[idx])
        .chain([target])
        .chain((k_1..mcx1_num_ancillas).map(|idx| controls[idx]))
        .collect();

    let mcx2 = crate::multi_controlled::synth_mcx_n_dirty_i15(k_2, false, false)?;
    let mcx2_num_ancillas = mcx2.num_qubits() - k_2 - 1;
    let mcx_2_qubits: Vec<Qubit> = (0..k_1)
        .map(|idx| controls[idx])
        .chain([target])
        .chain((k_2..mcx2_num_ancillas).map(|idx| controls[idx]))
        .collect();

    let mut circuit = CircuitData::with_capacity(
        num_ctrl_qubits + 1,
        0,
        mcx1.len() + mcx2.len() + 6,
        0.0.into(),
    )?;

    compose_circuit(&mut circuit, mcx1.clone(), &mcx_1_qubits);
    circuit.push_standard_gate(StandardGate::RZ, std::slice::from_ref(&theta), &[target])?;

    compose_circuit(&mut circuit, mcx2.clone(), &mcx_2_qubits);
    circuit.push_standard_gate(rz_inverse, &rz_inverse_params, &[target])?;

    compose_circuit(&mut circuit, mcx1, &mcx_1_qubits);
    circuit.push_standard_gate(StandardGate::RZ, std::slice::from_ref(&theta), &[target])?;

    compose_circuit(&mut circuit, mcx2, &mcx_2_qubits);
    circuit.push_standard_gate(rz_inverse, &rz_inverse_params, &[target])?;

    Ok(circuit)
}

fn mcu_graycode(num_bits: usize) -> Vec<String> {
    let mut result: Vec<u32> = vec![0];
    for num in 0..num_bits {
        let mut temp = vec![];
        result
            .iter()
            .rev()
            .inspect(|val| temp.push(**val + 2_u32.pow(num as u32)));
        result.extend(temp);
    }
    Vec::from_iter(result.iter().map(|num| format!("{num:num_bits$b}")))
}

fn mcu_graycode_circ(
    num_ctrl_qubits: u32,
    params: &[Param],
) -> Result<CircuitData, CircuitDataError> {
    let mut circ =
        CircuitData::with_capacity(num_ctrl_qubits + 1, 0, num_ctrl_qubits as usize, 0.0.into())?;
    let qargs: Vec<Qubit> = (0..(num_ctrl_qubits + 1)).map(Qubit).collect();
    let gray = mcu_graycode(num_ctrl_qubits as usize);
    let mut last_pattern = None;
    for pattern in gray.iter() {
        if !pattern.contains("1") {
            continue;
        }
        if last_pattern.is_none() {
            last_pattern = Some(pattern);
        }

        // We proved the pattern contains '1'
        let lm_pos = pattern.find("1").unwrap();
        let last_unwrapped = last_pattern.unwrap();
        let pos = pattern
            .chars()
            .zip(last_unwrapped.chars())
            .map(|(i, j)| i != j)
            .position(|idx| idx);

        if let Some(pos) = pos {
            if pos != lm_pos {
                circ.push_standard_gate(StandardGate::CX, &[], &[qargs[lm_pos], qargs[pos]]);
            } else {
                for index in pattern
                    .chars()
                    .enumerate()
                    .filter_map(|(idx, char)| (char == '1').then_some(idx))
                {
                    circ.push_standard_gate(StandardGate::CX, &[], &[qargs[index], qargs[lm_pos]])?;
                }
            }
        }

        if pattern.chars().filter(|bit| *bit == '1').count() % 2 == 0 {
            let new_params: Vec<Param> = params
                .iter()
                .cloned()
                .map(|param| match param {
                    Param::ParameterExpression(expr) => {
                        Ok(Param::ParameterExpression(expr.neg().into()))
                    }
                    Param::Float(f) => Ok((-f).into()),
                    // This is currently unsupported
                    Param::Int(_) | Param::Obj(_) => Err(CircuitDataError::InvalidParameter),
                })
                .collect::<Result<_, CircuitDataError>>()?;
            circ.push_standard_gate(
                StandardGate::CU,
                &new_params,
                &[qargs[lm_pos], qargs[qargs.len() - 1]],
            )?;
        } else {
            circ.push_standard_gate(
                StandardGate::CU,
                params,
                &[qargs[lm_pos], qargs[qargs.len() - 1]],
            )?;
        }
        last_pattern = Some(pattern);
    }
    Ok(circ)
}

// TODO: Add `CircuitData::compose` method.
fn compose_circuit(
    circ: &mut CircuitData,
    mc_definition: CircuitData,
    qubits: &[Qubit],
) -> Result<(), DAGError> {
    let mut circ_as_dag = DAGCircuit::from_circuit_data(circ, false, None, None)?;

    let mc_definition_as_dag = DAGCircuit::from_circuit_data(&mc_definition, false, None, None)?;

    circ_as_dag.compose(
        &mc_definition_as_dag,
        Some(qubits),
        None,
        Default::default(),
        false,
    )?;

    *circ = CircuitData::from_dag_ref(&circ_as_dag)?;
    Ok(())
}

fn apply_basic_controlled_gate(
    definition: &mut CircuitData,
    gate: &StandardGate,
    params: &[Param],
) -> Result<(), DAGError> {
    // Since this originates from `native_add_control` this should fully result
    // in the control register being the first one here, and the target register
    // being the second.
    let qregs = definition.qregs();
    let [q_ctrl, q_trgt] = [&qregs[0], &qregs[1]];
    let ctrl_bits: Vec<Qubit> = q_ctrl
        .iter()
        .map(|bit| definition.qubits().find(&bit))
        .collect::<Option<_>>()
        .expect("Qubit should be present in original circuit");
    let target_bits: Vec<Qubit> = q_trgt
        .iter()
        .map(|bit| definition.qubits().find(&bit))
        .collect::<Option<_>>()
        .expect("Qubit should be present in original circuit");
    let all_qubits: Vec<Qubit> = ctrl_bits
        .iter()
        .chain(target_bits.iter())
        .copied()
        .collect();
    let num_ctrl_qubits = q_ctrl.len();
    match gate {
        StandardGate::X => {
            let new_def = crate::multi_controlled::synth_mcx_noaux_sp22(num_ctrl_qubits)?;
            compose_circuit(definition, new_def, &all_qubits)?;
        }
        StandardGate::Phase => {
            let circ =
                crate::multi_controlled::synth_mcp_noaux_sp22(num_ctrl_qubits, params[0].clone())?;
            compose_circuit(definition, circ, &all_qubits)?;
        }
        StandardGate::Z => {
            definition.push_standard_gate(StandardGate::H, &[], &target_bits)?;
            let new_def = crate::multi_controlled::synth_mcx_noaux_sp22(num_ctrl_qubits)?;
            compose_circuit(definition, new_def, &all_qubits)?;
            definition.push_standard_gate(StandardGate::H, &[], &target_bits)?;
        }
        StandardGate::Y => {
            definition.push_standard_gate(StandardGate::Sdg, &[], &target_bits)?;
            let new_def = crate::multi_controlled::synth_mcx_noaux_sp22(num_ctrl_qubits)?;
            compose_circuit(definition, new_def, &all_qubits)?;
            definition.push_standard_gate(StandardGate::S, &[], &target_bits)?;
        }
        StandardGate::H => {
            definition.push_standard_gate(StandardGate::S, &[], &target_bits)?;
            definition.push_standard_gate(StandardGate::H, &[], &target_bits)?;
            definition.push_standard_gate(StandardGate::T, &[], &target_bits)?;
            let new_def = crate::multi_controlled::synth_mcx_noaux_sp22(num_ctrl_qubits)
                .expect("MCX composition should be infallible.");
            compose_circuit(definition, new_def, &all_qubits)?;
            definition.push_standard_gate(StandardGate::Tdg, &[], &target_bits)?;
            definition.push_standard_gate(StandardGate::H, &[], &target_bits)?;
            definition.push_standard_gate(StandardGate::Sdg, &[], &target_bits)?;
        }
        StandardGate::SX => {
            definition.push_standard_gate(StandardGate::H, &[], &target_bits)?;
            let new_def =
                crate::multi_controlled::synth_mcp_noaux_sp22(num_ctrl_qubits, (PI / 2.0).into())?;
            compose_circuit(definition, new_def, &all_qubits)?;
            definition.push_standard_gate(StandardGate::H, &[], &target_bits)?;
        }
        StandardGate::SXdg => {
            definition.push_standard_gate(StandardGate::H, &[], &target_bits)?;
            let new_def = crate::multi_controlled::synth_mcp_noaux_sp22(
                num_ctrl_qubits,
                (3.0 * PI / 2.0).into(),
            )?;
            compose_circuit(definition, new_def, &all_qubits)?;
            definition.push_standard_gate(StandardGate::H, &[], &target_bits)?;
        }
        StandardGate::RX => {
            let new_def = synth_mcrx(params[0].clone(), num_ctrl_qubits as u32)?;
            compose_circuit(definition, new_def, &all_qubits)?;
        }
        StandardGate::RY => {
            let new_def = synth_mcry(params[0].clone(), num_ctrl_qubits as u32)?;
            compose_circuit(definition, new_def, &all_qubits)?;
        }
        StandardGate::RZ => {
            let new_def = synth_mcrz(params[0].clone(), num_ctrl_qubits as u32)?;
            compose_circuit(definition, new_def, &all_qubits)?;
        }
        StandardGate::U => {
            let [theta, phi, lamb] = params else {
                return Err(CircuitDataError::ParameterSliceLenMismatch.into());
            };
            if num_ctrl_qubits == 1 {
                if compare_param_float(theta, 0.0) && compare_param_float(phi, 0.0) {
                    definition.push_standard_gate(
                        StandardGate::CPhase,
                        std::slice::from_ref(lamb),
                        &ctrl_bits[0..1],
                    )?;
                } else {
                    definition.push_standard_gate(
                        StandardGate::CU,
                        &[theta.clone(), phi.clone(), lamb.clone()],
                        &ctrl_bits[0..1],
                    )?;
                }
            } else if compare_param_float(phi, -PI / 2.0) && compare_param_float(lamb, PI / 2.0) {
                let new_def = synth_mcrx(theta.clone(), num_ctrl_qubits as u32)?;
                compose_circuit(definition, new_def, &all_qubits)?;
            } else if compare_param_float(phi, 0.0) && compare_param_float(lamb, 0.0) {
                let new_def = synth_mcry(theta.clone(), num_ctrl_qubits as u32)?;
                compose_circuit(definition, new_def, &all_qubits)?;
            } else if compare_param_float(theta, 0.0) && compare_param_float(phi, 0.0) {
                let new_def =
                    crate::multi_controlled::synth_mcp_noaux_sp22(num_ctrl_qubits, lamb.clone())?;
                compose_circuit(definition, new_def, &all_qubits)?;
            } else {
                let top_mcrz = synth_mcrz(lamb.clone(), num_ctrl_qubits as u32)?;
                compose_circuit(definition, top_mcrz, &all_qubits)?;
                let mcry = synth_mcry(theta.clone(), num_ctrl_qubits as u32)?;
                compose_circuit(definition, mcry, &all_qubits)?;
                let bottom_mcrz = synth_mcry(phi.clone(), num_ctrl_qubits as u32)?;
                compose_circuit(definition, bottom_mcrz, &all_qubits)?;

                let phi_plus_lamb_div_2 =
                    multiply_param(&radd_param(phi.clone(), lamb.clone()), 0.5);
                let bits: Vec<Qubit> = ctrl_bits[1..]
                    .iter()
                    .chain(ctrl_bits[0..1].iter())
                    .copied()
                    .collect();
                let mcp = crate::multi_controlled::synth_mcp_noaux_sp22(
                    num_ctrl_qubits,
                    phi_plus_lamb_div_2,
                )?;
                compose_circuit(definition, mcp, &bits)?;
            }
        }
        StandardGate::CX => {
            let new_def = crate::multi_controlled::synth_mcx_noaux_sp22(num_ctrl_qubits + 1)?;
            compose_circuit(definition, new_def, &all_qubits)?;
        }
        StandardGate::CZ => {
            definition.push_standard_gate(StandardGate::H, params, &target_bits[1..])?;

            let new_def = crate::multi_controlled::synth_mcx_noaux_sp22(num_ctrl_qubits + 1)?;
            compose_circuit(definition, new_def, &all_qubits)?;

            definition.push_standard_gate(StandardGate::H, params, &target_bits[1..])?;
        }
        _ => return Err(DAGError::General("Gate is not basic.".to_string())),
    };
    Ok(())
}

fn compare_param_float<P: Into<Param>>(param: &Param, val: P) -> bool {
    param.eq(&val.into()).is_ok_and(|b| b)
}
