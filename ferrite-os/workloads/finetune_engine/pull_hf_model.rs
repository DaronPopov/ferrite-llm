use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
struct Config {
    repo: String,
    dest: PathBuf,
    revision: String,
    token: Option<String>,
    include: Vec<String>,
    resume: bool,
}

fn print_usage() {
    eprintln!(
        "Usage: pull_hf_model.rs --repo ORG/MODEL --dest PATH [options]\n\
         Options:\n\
           --revision REV       Git revision/branch/tag (default: main)\n\
           --token TOKEN        Hugging Face token (or use HF_TOKEN env)\n\
           --include PATTERN    LFS include pattern (repeatable), e.g. *.safetensors\n\
           --resume             If dest exists, fetch latest + lfs pull instead of failing\n\
           -h, --help           Show help"
    );
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let mut repo: Option<String> = None;
    let mut dest: Option<PathBuf> = None;
    let mut revision = String::from("main");
    let mut token: Option<String> = env::var("HF_TOKEN").ok();
    let mut include = Vec::new();
    let mut resume = false;

    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--repo" => {
                let v = args.next().ok_or("missing value for --repo")?;
                repo = Some(v);
            }
            "--dest" => {
                let v = args.next().ok_or("missing value for --dest")?;
                dest = Some(PathBuf::from(v));
            }
            "--revision" => {
                revision = args.next().ok_or("missing value for --revision")?;
            }
            "--token" => {
                token = Some(args.next().ok_or("missing value for --token")?);
            }
            "--include" => {
                include.push(args.next().ok_or("missing value for --include")?);
            }
            "--resume" => {
                resume = true;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    let repo = repo.ok_or("--repo is required")?;
    let dest = dest.ok_or("--dest is required")?;

    Ok(Config {
        repo,
        dest,
        revision,
        token,
        include,
        resume,
    })
}

fn run(cmd: &mut Command, label: &str) -> Result<(), Box<dyn Error>> {
    let status = cmd.status()?;
    if !status.success() {
        return Err(format!("{label} failed with status {status}").into());
    }
    Ok(())
}

fn ensure_tool(name: &str, version_arg: &str) -> Result<(), Box<dyn Error>> {
    let status = Command::new(name).arg(version_arg).status();
    match status {
        Ok(s) if s.success() => Ok(()),
        _ => Err(format!("required tool not found or not working: {name}").into()),
    }
}

fn encode_token_for_url(token: &str) -> String {
    // Minimal percent-encode for URL userinfo safety.
    let mut out = String::with_capacity(token.len());
    for b in token.bytes() {
        let ch = b as char;
        let safe = ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '~');
        if safe {
            out.push(ch);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn repo_url(repo: &str, token: Option<&str>) -> String {
    match token {
        Some(t) => {
            let enc = encode_token_for_url(t);
            format!("https://user:{enc}@huggingface.co/{repo}")
        }
        None => format!("https://huggingface.co/{repo}"),
    }
}

fn is_non_empty_dir(path: &Path) -> Result<bool, Box<dyn Error>> {
    if !path.exists() {
        return Ok(false);
    }
    if !path.is_dir() {
        return Err(format!("destination exists and is not a directory: {}", path.display()).into());
    }
    Ok(path.read_dir()?.next().is_some())
}

fn redact_url(url: &str) -> String {
    if let Some(idx) = url.find("@huggingface.co") {
        return format!("https://***:***{}", &url[idx..]);
    }
    url.to_string()
}

fn git_clone(cfg: &Config, url: &str) -> Result<(), Box<dyn Error>> {
    let mut cmd = Command::new("git");
    cmd.arg("clone")
        .arg("--depth")
        .arg("1")
        .arg("--branch")
        .arg(&cfg.revision)
        .arg(url)
        .arg(&cfg.dest);
    run(&mut cmd, "git clone")
}

fn git_resume(cfg: &Config) -> Result<(), Box<dyn Error>> {
    run(
        Command::new("git")
            .arg("-C")
            .arg(&cfg.dest)
            .arg("fetch")
            .arg("--all")
            .arg("--tags"),
        "git fetch",
    )?;

    run(
        Command::new("git")
            .arg("-C")
            .arg(&cfg.dest)
            .arg("checkout")
            .arg(&cfg.revision),
        "git checkout",
    )?;

    run(
        Command::new("git")
            .arg("-C")
            .arg(&cfg.dest)
            .arg("pull")
            .arg("--ff-only"),
        "git pull",
    )
}

fn lfs_pull(cfg: &Config) -> Result<(), Box<dyn Error>> {
    run(
        Command::new("git")
            .arg("-C")
            .arg(&cfg.dest)
            .arg("lfs")
            .arg("install")
            .arg("--local"),
        "git lfs install",
    )?;

    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(&cfg.dest).arg("lfs").arg("pull");

    if !cfg.include.is_empty() {
        let joined = cfg.include.join(",");
        cmd.arg("-I").arg(joined);
    }

    run(&mut cmd, "git lfs pull")
}

fn count_files(path: &Path) -> Result<u64, Box<dyn Error>> {
    let mut stack: Vec<OsString> = vec![path.as_os_str().to_os_string()];
    let mut n = 0u64;
    while let Some(p) = stack.pop() {
        let pb = PathBuf::from(p);
        for ent in std::fs::read_dir(&pb)? {
            let ent = ent?;
            let meta = ent.metadata()?;
            if meta.is_dir() {
                stack.push(ent.path().into_os_string());
            } else if meta.is_file() {
                n += 1;
            }
        }
    }
    Ok(n)
}

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = parse_args()?;
    ensure_tool("git", "--version")?;
    ensure_tool("git-lfs", "version")?;

    let url = repo_url(&cfg.repo, cfg.token.as_deref());

    println!("[hf-pull] repo: {}", cfg.repo);
    println!("[hf-pull] revision: {}", cfg.revision);
    println!("[hf-pull] dest: {}", cfg.dest.display());
    println!("[hf-pull] auth: {}", if cfg.token.is_some() { "token" } else { "none" });
    println!("[hf-pull] clone-url: {}", redact_url(&url));

    let exists_non_empty = is_non_empty_dir(&cfg.dest)?;
    if exists_non_empty {
        if !cfg.resume {
            return Err(format!(
                "destination exists and is non-empty: {} (use --resume)",
                cfg.dest.display()
            )
            .into());
        }
        println!("[hf-pull] resume mode: fetching existing repo");
        git_resume(&cfg)?;
    } else {
        git_clone(&cfg, &url)?;
    }

    lfs_pull(&cfg)?;

    let file_count = count_files(&cfg.dest)?;
    println!("[hf-pull] done. files={file_count}");
    println!("RESULT repo={}", cfg.repo);
    println!("RESULT revision={}", cfg.revision);
    println!("RESULT dest={}", cfg.dest.display());
    println!("RESULT files={file_count}");

    Ok(())
}
