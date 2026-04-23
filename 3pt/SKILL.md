---
name: 3pt
description: >
 Specialized LQCD three-point skill for Lambda->proton transitions.
 Derives and applies the sequential-source contraction chain with the
 conventions used in this workspace: DeGrand-Rossi gamma basis,
 gamma5-hermiticity, two dagger steps (seq source and post-inversion
 reorientation), and explicit einsum-ready tensor layouts.
 Trigger on: Lambda->p 3pt setup, weak vector/axial current insertion,
 sequential propagator construction, or requests for concrete
 contraction code consistent with these conventions.
---

# 3pt Lambda→proton Skill

## Purpose

Use this skill when the task is specifically the Lambda→proton three-point
correlator with sequential-source implementation details.

## What this skill must return

- A consistent chain from operator definition to executable contraction code.
- Propagator requirements for forward light/strange and sequential light solves.
- Code-level notes that preserve the two-dagger convention:
  1) `η_seq = γ₅ · B† · γ₅` after sink-block assembly,
  2) `G_l_seq_dag` from `G_l_seq` after inversion for kernel index layout.
- Final contraction forms for vector (Γ = I) and axial (Γ = γ₅).

## Mandatory workflow in this workspace

- After the 3pt source/sink/current operators are fully defined, perform Wick contraction using:
  `../WickContraction`
- Prefer the dedicated skill entry at:
  `../WickContraction/SKILL.md`
- Treat this WickContraction path as the default backend for all future 3pt tasks in this workspace.
- Scope restriction: generate only 3pt correlator outputs; do not add 2pt, spectrum, fitting, or unrelated byproducts unless explicitly requested.
- Keep the output interface minimal and focused on 3pt arrays/files needed by downstream analysis.

## Input checklist (ask if missing)

- Ensemble and lattice size / boundary conditions
- Source position / time and sink separation tseq
- Momentum choices p_i, p_f (or q) and current direction μ
- Projector choice T and sink/source smearing choices

## Output style constraints

- Keep notation consistent with the equations in `references/theory.md`
  and code symbols in `references/contraction.md` / `references/code-c24p29.md`.
- If compact formula uses `G_l^seq` but code uses `G_l_seq_dag`, explicitly
  state they are implementation-equivalent under the index convention.

## References

- **Theory / equations** → `references/theory.md`
- **Einsum contraction code** → `references/contraction.md`
- **Full pyquda code (C24P29)** → `references/code-c24p29.md`
- **Wick contraction backend (local clone)** → `../WickContraction/SKILL.md`
