# Installer Model

The `installer/` tree is the declarative control plane for Ferrite.

It splits the system into four layers:

1. Host bootstrap
2. Workspace dependency graph
3. Derived build artifacts
4. Generated runtime environment contract

The key invariant is:

Everything not committed to git must be reconstructable from committed declarations.

Current phase:

- manifests define desired intent
- profiles define machine/provider selection
- bootstrap invokes a small standalone installer engine
- the engine currently reads manifests and prints a plan

Planned next phases:

- host tool installation
- external source materialization into `.ferrite/store`
- binary asset fetch + checksum verification
- env generation
- validation execution

Binary assets are not generic downloads. For `libtorch`, the manifest is meant to
encode the wrapper ABI contract:

- exact archive family
- CUDA flavor
- C++ ABI mode
- `tch` compatibility
- `aten-ptx` / `ferrite-torch` compatibility
- machine constraints

`torch` should therefore be modeled as one runtime contract with multiple valid
provider realizations, for example:

- x86_64 CUDA prebuilt `libtorch`
- Jetson/system-provided `libtorch`

That keeps x86 CUDA and ARM CUDA as one logical feature without forcing them into
the same binary asset.

The installer engine should therefore resolve in two steps:

1. Select the runtime contract required by enabled features
2. Select the provider-backed asset(s) named by the active profile and verify
   they satisfy that contract

The first materialization phase should be non-destructive:

- create `.ferrite` state directories
- record manifests for source deps and runtime assets
- adopt already-present legacy `external/*` paths into the managed view
- only create compatibility symlinks when the repo target path is absent

The next bootstrap phases are:

- host tool detection with optional provider-backed installation
- pinned git fetch into `.ferrite/store/repos`
- subdir export into `.ferrite/store/materialized`
- later repointing compatibility paths from legacy trees to managed materialized trees
- runtime asset realization by provider:
  prebuilt download/unpack, or system detection
- env contract generation from templates
- build/apply execution for the selected profile
- validation execution for the selected profile
