# Ferrite Docs

This directory holds the engine-level documentation for Ferrite.

If the root [`README.md`](../README.md) is
the product landing page, this folder is the operator and architecture manual.

## Start Here

- [Architecture](ARCHITECTURE.md)
  What Ferrite is, how the stack is layered, and how execution flows through it.
- [Running Ferrite](RUNNING.md)
  Commands, runtime workflow, troubleshooting, and environment setup.
- [Engine Identity](ENGINE_IDENTITY.md)
  The repo-level design stance: one engine tree, one installer, one stack.
- [Installer Architecture](INSTALLER_ARCHITECTURE.md)
  How the installer is supposed to think about modules, profiles, and build scope.

## Subsystem Docs

- [ferrite-cli](../crates/ferrite-cli/README.md)
- [ferrite-core](../crates/ferrite-core/README.md)
- [ferrite-wasm-host](../crates/ferrite-wasm-host/README.md)
- [ferrite-sdk](../crates/ferrite-sdk/README.md)
- [ferrite-os](../ferrite-os/README.md)
- [ferrite-gpu-lang](../ferrite-gpu-lang/README.md)

## Read This Repo As

Ferrite is not just an app.

It is a vertical engine stack with:

- a lightweight operator CLI
- a heavy execution runner
- a WASM host/guest boundary
- an LLM inference engine
- a lower PTX/CUDA substrate
- a programmable GPU layer
- optional extended subsystems like graphics

Short version:

`Ferrite is a full engine tree, not a loose set of projects.`
