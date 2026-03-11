use std::collections::HashSet;

use super::*;

impl Graph {
    /// Get all tensor metadata.
    pub fn tensors(&self) -> &IndexMap<TensorId, TensorMeta> {
        &self.tensors
    }

    /// Get tensor metadata by ID.
    pub fn tensor(&self, id: TensorId) -> Option<&TensorMeta> {
        self.tensors.get(&id)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &IndexMap<NodeId, Node> {
        &self.nodes
    }

    /// Get a node by ID.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Get input tensor IDs.
    pub fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    /// Get output tensor IDs.
    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }

    /// Get the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of tensors.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    // ========================================================================
    // Graph analysis
    // ========================================================================

    /// Get nodes in topological order.
    pub fn topo_order(&self) -> Vec<NodeId> {
        // Simple topological sort based on insertion order
        // (since we add nodes in dependency order)
        self.nodes.keys().copied().collect()
    }

    /// Find which nodes produce tensors used by a given node.
    pub fn producers(&self, node_id: NodeId) -> Vec<NodeId> {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return Vec::new(),
        };

        let mut producers = Vec::new();
        for input_id in &node.inputs {
            for (nid, n) in &self.nodes {
                if n.output == *input_id {
                    producers.push(*nid);
                }
            }
        }
        producers
    }

    /// Find which nodes consume a tensor.
    pub fn consumers(&self, tensor_id: TensorId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, n)| n.inputs.contains(&tensor_id))
            .map(|(id, _)| *id)
            .collect()
    }

    // ========================================================================
    // Graph mutation (for optimization passes)
    // ========================================================================

    /// Replace a computation node with a constant. Converts the op to
    /// `Constant`, clears its inputs, and sets the scalar value on the
    /// output tensor metadata.
    pub(crate) fn replace_with_constant(&mut self, node_id: NodeId, value: f32) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.op = OpCode::Constant;
            node.inputs.clear();
            node.attrs = OpAttrs::default();
            if let Some(meta) = self.tensors.get_mut(&node.output) {
                meta.is_constant = true;
                meta.constant_value = Some(value);
            }
        }
    }

    /// Remove nodes whose outputs are not in `live_tensors`.
    /// Also removes orphaned tensor metadata. Graph inputs are always
    /// preserved regardless of the live set.
    pub(crate) fn remove_dead_nodes(&mut self, live_tensors: &HashSet<TensorId>) {
        // Keep Input nodes (graph inputs) and nodes with live outputs.
        self.nodes.retain(|_, node| {
            node.op == OpCode::Input || live_tensors.contains(&node.output)
        });

        // Collect all tensors still referenced by remaining nodes.
        let mut referenced: HashSet<TensorId> = HashSet::new();
        for node in self.nodes.values() {
            referenced.insert(node.output);
            for &input_id in &node.inputs {
                referenced.insert(input_id);
            }
        }

        // Keep tensors that are referenced or are graph inputs.
        self.tensors.retain(|id, _| referenced.contains(id));
    }
}
