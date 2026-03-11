use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// A single entry in the flattened tree view.
#[derive(Debug, Clone)]
pub struct TreeEntry {
    pub path: PathBuf,
    pub depth: usize,
    pub is_dir: bool,
    #[allow(dead_code)]
    pub child_count: usize,
}

/// Hierarchical file tree that can expand/collapse directories.
pub struct FileTree {
    pub visible: Vec<TreeEntry>,
    pub expanded: HashSet<PathBuf>,
    pub root: PathBuf,
}

impl FileTree {
    pub fn new(root: PathBuf) -> Self {
        let mut tree = Self {
            visible: Vec::new(),
            expanded: HashSet::new(),
            root,
        };
        tree.rebuild();
        tree
    }

    /// Rebuild the flat visible list by walking expanded directories.
    pub fn rebuild(&mut self) {
        self.visible.clear();
        self.walk(&self.root.clone(), 0);
        // Sort: directories first at each level, then alphabetical
        // (already handled by walk since we sort entries per-dir)
    }

    fn walk(&mut self, dir: &Path, depth: usize) {
        let Ok(read_dir) = std::fs::read_dir(dir) else {
            return;
        };

        let mut entries: Vec<(PathBuf, bool)> = Vec::new();
        for entry in read_dir.flatten() {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if name.starts_with(".git") || name == "target" || name.starts_with('.') {
                continue;
            }
            let is_dir = path.is_dir();
            entries.push((path, is_dir));
        }

        // Sort: directories first, then alphabetical
        entries.sort_by(|a, b| {
            b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
        });

        for (path, is_dir) in entries {
            let child_count = if is_dir {
                std::fs::read_dir(&path)
                    .map(|rd| rd.count())
                    .unwrap_or(0)
            } else {
                0
            };

            self.visible.push(TreeEntry {
                path: path.clone(),
                depth,
                is_dir,
                child_count,
            });

            if is_dir && self.expanded.contains(&path) {
                self.walk(&path, depth + 1);
            }
        }
    }

    /// Toggle expand/collapse of a directory at the given index.
    pub fn toggle(&mut self, index: usize) {
        if let Some(entry) = self.visible.get(index) {
            if entry.is_dir {
                let path = entry.path.clone();
                if self.expanded.contains(&path) {
                    self.expanded.remove(&path);
                } else {
                    self.expanded.insert(path);
                }
                self.rebuild();
            }
        }
    }

    /// Expand a directory at the given index.
    pub fn expand(&mut self, index: usize) {
        if let Some(entry) = self.visible.get(index) {
            if entry.is_dir && !self.expanded.contains(&entry.path) {
                self.expanded.insert(entry.path.clone());
                self.rebuild();
            }
        }
    }

    /// Collapse a directory at the given index.
    /// Returns the parent directory index if we should navigate up.
    pub fn collapse(&mut self, index: usize) -> Option<usize> {
        if let Some(entry) = self.visible.get(index) {
            if entry.is_dir && self.expanded.contains(&entry.path) {
                self.expanded.remove(&entry.path);
                self.rebuild();
                return None;
            }
            // If it's a file or already-collapsed dir, navigate to parent
            let parent = entry.path.parent()?;
            let parent_idx = self.visible.iter().position(|e| e.path == parent);
            return parent_idx;
        }
        None
    }

    /// Get the path at a given visible index.
    #[allow(dead_code)]
    pub fn path_at(&self, index: usize) -> Option<&Path> {
        self.visible.get(index).map(|e| e.path.as_path())
    }

    /// Number of visible entries.
    pub fn len(&self) -> usize {
        self.visible.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.visible.is_empty()
    }
}
