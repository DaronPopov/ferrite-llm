/// Attempt to acquire the agent lock.
pub fn acquire_lock(current: &Option<String>, owner: &str) -> Result<String, String> {
    match current {
        Some(existing) if existing != owner => {
            Err(format!("already locked by '{}'", existing))
        }
        Some(existing) => {
            Ok(format!("lock already held by '{}'", existing))
        }
        None => {
            Ok(format!("locked by '{}'", owner))
        }
    }
}

/// Release the agent lock.
pub fn release_lock(current: &Option<String>, requester: Option<&str>) -> Result<String, String> {
    match current {
        None => Err("no lock held".into()),
        Some(owner) => {
            if let Some(req) = requester {
                if req != owner {
                    return Err(format!(
                        "cannot unlock: held by '{}', requested by '{}'",
                        owner, req
                    ));
                }
            }
            Ok(format!("unlocked (was held by '{}')", owner))
        }
    }
}

/// Check whether human edits should be blocked due to an agent lock.
#[allow(dead_code)]
pub fn is_locked_for_human(lock: &Option<String>) -> bool {
    lock.is_some()
}
