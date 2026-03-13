# ferrite installer

This directory is the beginning of Ferrite's declarative installer control plane.

Current contents:

- `manifests/`: desired graph, feature bundles, host tools, and system package intent
- `manifests/products.toml`: installable product slices such as `ferrite-llm` and `ferrite-full`
- `profiles/`: machine-specific provider and validation selection
- `templates/`: generated runtime envelope templates
- `bootstrap/bootstrap.sh`: minimal bootstrap entrypoint
- `src/main.rs`: standalone installer engine scaffold

Current command:

```bash
cargo run --manifest-path installer/Cargo.toml -- plan --profile workstation-nvidia
```

Resolved asset view:

```bash
cargo run --manifest-path installer/Cargo.toml -- resolve --profile workstation-nvidia
```

Non-destructive materialization pass:

```bash
cargo run --manifest-path installer/Cargo.toml -- materialize --profile workstation-nvidia
```

Host bootstrap check:

```bash
cargo run --manifest-path installer/Cargo.toml -- bootstrap-host --profile workstation-nvidia
```

Apply missing host tools:

```bash
cargo run --manifest-path installer/Cargo.toml -- bootstrap-host --profile workstation-nvidia --apply
```

Fetch pinned source dependencies:

```bash
cargo run --manifest-path installer/Cargo.toml -- fetch-sources --profile workstation-nvidia
```

Fetch or detect runtime assets:

```bash
cargo run --manifest-path installer/Cargo.toml -- fetch-assets --profile workstation-nvidia
```

Generate the runtime env contract:

```bash
cargo run --manifest-path installer/Cargo.toml -- generate-env --profile workstation-nvidia
```

Build the selected profile:

```bash
cargo run --manifest-path installer/Cargo.toml -- build-profile --profile workstation-nvidia
```

Run validation checks:

```bash
cargo run --manifest-path installer/Cargo.toml -- validate-profile --profile workstation-nvidia
```

Run the whole installer pipeline:

```bash
cargo run --manifest-path installer/Cargo.toml -- bootstrap-all --profile workstation-nvidia
```

Bootstrap entrypoint:

```bash
./installer/bootstrap/bootstrap.sh --profile workstation-nvidia
```

If `--profile` is omitted, the installer auto-selects:

- `jetson` on detected NVIDIA Jetson `aarch64` hosts
- `workstation-nvidia` on `linux/x86_64`
- `cpu-only-dev` otherwise

Current product model:

- `ferrite-llm`: builds the LLM/runtime package slice only
- `ferrite-full`: builds the full monorepo workspace

Right now the engine can:

- print the declared plan
- resolve selected runtime assets for a profile
- create `.ferrite` state directories
- adopt existing `external/*` trees into `.ferrite/store/src` non-destructively
- detect missing host bootstrap tools and install them when `--apply` is used
- fetch pinned source repositories into `.ferrite/store/repos` and export declared subdirs into `.ferrite/store/materialized`
- realize selected runtime assets by adopting legacy paths, detecting system installs, or downloading archives
- generate `.ferrite/env/profile.sh` and `.ferrite/bin/ferrite-run`
- build the selected profile from the installer engine
- run profile validation checks from the installer engine

The bootstrap entrypoint now runs the full installer pipeline. The top-level repo
`install.sh` still has not been handed off to it yet.
