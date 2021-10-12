pub mod clustering;

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn ph_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "merge_h0")]
    fn merge_h0_py(
        graph: Vec<Vec<usize>>,
        density_map: Vec<f32>,
        threshold: f32,
    ) -> PyResult<Vec<usize>> {
        Ok(clustering::zero_dim::merge_h0(
            &graph,
            &density_map,
            threshold,
        ))
    }

    Ok(())
}
