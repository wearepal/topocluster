use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
mod clustering;

/// Rust module containing algorithms based on persistent-homology.
#[pymodule]
fn ph_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "merge_h0", text_signature = "graph, density_map, threshold")]
    fn merge_h0_py(
        neighbor_graph: Vec<Vec<usize>>,
        density_map: Vec<f32>,
        threshold: f32,
    ) -> PyResult<Vec<usize>> {
        Ok(clustering::zero_dim::merge_h0(
            &neighbor_graph,
            &density_map,
            threshold,
        ))
    }

    Ok(())
}
