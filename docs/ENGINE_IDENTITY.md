# Ferrite Engine Identity

Ferrite is a unified engine tree.

It is not meant to be a loose collection of adjacent projects, and it is not
meant to be split into separate repos for users to assemble by hand. The
installed checkout under `ferrite/` is the engine.

## Core Idea

The product shape is:

- one installer
- one top-level engine directory
- one operator CLI surface
- one integrated source tree containing runtime, substrate, programmable GPU
  layer, examples, and optional subsystems

That means the directory created by install is not just a sample app checkout.
It is the full Ferrite engine tree.

## What Ferrite Is

Ferrite is a vertically integrated AI execution stack hosted on Linux and CUDA.

It includes:

- runtime and operator interface
- WASM/WIT host-guest boundary
- inference engine
- native CUDA/PTX runtime pieces
- lower-level Ferrite OS substrate
- programmable GPU scripting/language layer
- optional graphics and extended platform modules

The correct mental model is:

`one engine, one tree, many layers`

For the visual and structural version, see
[Architecture](ARCHITECTURE.md).

## Layer Map

Product/runtime layer:

- [`crates/ferrite-cli`](../crates/ferrite-cli)
- [`crates/ferrite-wasm-host`](../crates/ferrite-wasm-host)
- [`crates/ferrite-core`](../crates/ferrite-core)
- [`crates/ferrite-sdk`](../crates/ferrite-sdk)

Guest/example layer:

- [`examples/wasm`](../examples/wasm)
- [`examples/orin-inference`](../examples/orin-inference)

Platform substrate:

- [`ferrite-os`](../ferrite-os)

Programmable GPU layer:

- [`ferrite-gpu-lang`](../ferrite-gpu-lang)

Optional extended subsystem:

- [`external/ferrite-graphics`](../external/ferrite-graphics)

## Monorepo Principle

Ferrite should remain one repo and one installable engine tree.

The right architectural pressure is not repo-splitting. The right pressure is:

- cleaner internal module boundaries
- explicit ownership between layers
- profile-based build/install policy
- shared top-level docs and operational workflows

That preserves the product requirement that a user can install Ferrite once and
get the whole engine source tree locally.

## Installer Principle

The one-line installer is the canonical product entrypoint.

By design, the plain installer path should build the full engine by default.
Profiles still exist, but they are narrower opt-outs from the full engine build,
not the default meaning of the product.

So the supported model is:

- plain one-line install: build the whole engine
- explicit `--profile ...`: build a smaller supported slice on purpose

## Operational Principle

The repo should behave like a self-contained system:

- `install.sh` bootstraps the engine
- `ferrite-rt` provides admin/operator flows
- `ferrite-rt-runner` executes the heavy runtime path
- `doctor` validates the local engine state
- profiles express supported build scopes inside the same engine tree

## Non-Goals

Ferrite is not trying to be:

- a Linux replacement
- a hardware kernel
- a set of independent crates with unrelated release cycles
- a repo where users are expected to understand which subproject is the real
  product

It is a vertical runtime platform with OS-like layering over AI/GPU execution.

## Design Standard

Any new subsystem added under `ferrite/` should answer these questions clearly:

- does it belong to the unified engine tree?
- is it part of the default full-engine install?
- if not, which install profile owns it?
- how does `doctor` validate it?
- how does it relate to the operator CLI and runtime model?

If a subsystem cannot answer those questions, it is not integrated enough yet.
