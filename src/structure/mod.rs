#[derive(Debug, Clone)]
pub struct Axes(pub Vec<String>);

#[derive(Debug, Clone)]
pub enum Topology {
    Sequence,
    Grid,
    Graph,
}

#[derive(Debug, Clone)]
pub struct Meaning {
    pub axes: Axes,
    pub topology: Topology,
}
