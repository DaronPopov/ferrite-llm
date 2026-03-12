# Ferrite Docs

This directory holds the engine-level documentation for Ferrite.

If the root [`README.md`](/home/daron/llm_engine/fer_llm/ferrite/README.md) is
the product landing page, this folder is the operator and architecture manual.

## Start Here

- [Architecture](/home/daron/llm_engine/fer_llm/ferrite/docs/ARCHITECTURE.md)
  What Ferrite is, how the stack is layered, and how execution flows through it.
- [Running Ferrite](/home/daron/llm_engine/fer_llm/ferrite/docs/RUNNING.md)
  Commands, runtime workflow, troubleshooting, and environment setup.
- [Engine Identity](/home/daron/llm_engine/fer_llm/ferrite/docs/ENGINE_IDENTITY.md)
  The repo-level design stance: one engine tree, one installer, one stack.
- [Installer Architecture](/home/daron/llm_engine/fer_llm/ferrite/docs/INSTALLER_ARCHITECTURE.md)
  How the installer is supposed to think about modules, profiles, and build scope.

## Subsystem Docs

- [ferrite-cli](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-cli/README.md)
- [ferrite-core](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-core/README.md)
- [ferrite-wasm-host](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-wasm-host/README.md)
- [ferrite-sdk](/home/daron/llm_engine/fer_llm/ferrite/crates/ferrite-sdk/README.md)
- [ferrite-os](/home/daron/llm_engine/fer_llm/ferrite/ferrite-os/README.md)
- [ferrite-gpu-lang](/home/daron/llm_engine/fer_llm/ferrite/ferrite-gpu-lang/README.md)

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
