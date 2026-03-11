use std::io::{self, Read, Write};
use std::thread;
use std::time::Duration;

use ferrite_platform::ipc::{Endpoint, IpcStream};

pub fn connect_and_send(socket: &str, command: &str) -> io::Result<()> {
    let endpoint = Endpoint::new(socket);
    let mut stream = IpcStream::connect(&endpoint)?;
    stream.write_all(command.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    stream.shutdown(std::net::Shutdown::Write)?;

    let mut resp = String::new();
    stream.read_to_string(&mut resp)?;
    print!("{}", resp);
    Ok(())
}

pub fn run_watch_client(socket: &str, watch_ms: u64) -> io::Result<()> {
    let is_tty = ferrite_platform::tty::stdout_is_tty();
    let endpoint = Endpoint::new(socket);

    loop {
        let mut stream = IpcStream::connect(&endpoint)?;
        stream.write_all(b"metrics\n")?;
        stream.flush()?;
        stream.shutdown(std::net::Shutdown::Write)?;

        let mut resp = String::new();
        stream.read_to_string(&mut resp)?;

        if is_tty {
            print!("\r{}", resp.trim());
            io::stdout().flush()?;
        } else {
            println!("{}", resp.trim());
        }

        thread::sleep(Duration::from_millis(watch_ms));
    }
}
