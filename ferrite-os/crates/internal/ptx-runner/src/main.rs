fn main() {
    if let Err(err) = ptx_runner::run_cli() {
        eprintln!("ptx-runner error: {err}");
        std::process::exit(1);
    }
}
