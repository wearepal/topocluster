use std::collections::HashMap;

fn merge_(
    ref_idx: usize,
    cmax_idx: usize,
    cnbd_idxs: &Vec<usize>,
    root_idxs: &mut Vec<usize>,
    cluster_map: &mut HashMap<usize, Vec<usize>>,
    density_map: &Vec<f32>,
    threshold: f32,
) -> () {
    for &cnbd_idx in cnbd_idxs.iter() {
        // exclude cmax: the other vertices will be merged into it if below
        // the persistence threshold
        if cnbd_idx != cmax_idx {
            // compute the persistence between each root vertex and the
            // current vertex
            let persistence = density_map[cnbd_idx] - density_map[ref_idx];
            // if the persistence is below the user-defined threshold,
            // then merge the root vertex into cmax.
            if persistence < threshold {
                let c = cluster_map.remove(&cnbd_idx);
                let cmax = cluster_map.get_mut(&cmax_idx).unwrap();
                if c.is_some() {
                    for &elem in c.as_ref().unwrap().iter() {
                        root_idxs[elem] = cmax_idx;
                        cmax.push(elem);
                    }
                    root_idxs[cnbd_idx] = cmax_idx;
                }
                root_idxs[cnbd_idx] = cmax_idx;
                cmax.push(cnbd_idx);
            }
        }
    }
}

/// Merges data based on their 0-dimensional persistence.
/// # Arguments
/// * `neighbor_graph` - Vector encoding the neighbourhood of each vertex.
/// * `density_map` - Vector containing the density associated with each vertex.
/// * `threshold` - Persistence threshold for merging.
pub fn cluster_h0(
    neighbor_graph: &Vec<Vec<usize>>,
    density_map: &Vec<f32>,
    threshold: f32,
    greedy: bool,
) -> Vec<usize> {
    assert!(
        neighbor_graph.len() == density_map.len(),
        "Neighbor graph and density map must have the same length."
    );
    let len = density_map.len();

    let indices: Vec<_> = (0..len).collect();
    // sort the vertices in descending order of density
    let mut pairs: Vec<_> = density_map.iter().zip(indices).collect();
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(a.0).unwrap());
    let (_, sort_idxs): (Vec<f32>, Vec<usize>) = pairs.iter().cloned().unzip();
    // indicates the root index to which each vertex is assigned
    let mut root_idxs: Vec<usize> = (0..len).collect();
    // mapping between root indexes and child indexes.
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();

    for &i in sort_idxs.iter() {
        let nbd_idxs = &neighbor_graph[i];
        // index of the neighboring vertex with the highest density
        let mut d_g: f32 = -f32::INFINITY;
        let mut g_i: Option<usize> = None;
        let mut cmax_idx: usize = 0;
        let mut d_cmax: f32 = -f32::INFINITY;
        // indexes of the clusters to which the neighboring vertices
        // currently belong.
        let mut cnbd_idxs = Vec::with_capacity(nbd_idxs.len());
        let d_i = density_map[i];

        for &j in nbd_idxs.iter() {
            if density_map[j] > d_i {
                let rj = root_idxs[j];
                cnbd_idxs.push(rj);
                let d_rj = density_map[rj];
                if d_rj > d_cmax {
                    d_cmax = d_rj;
                    cmax_idx = rj;
                }
                if !greedy {
                    let d_j = density_map[j];
                    if d_j > d_g {
                        g_i = Some(j);
                        d_g = d_j;
                    }
                }
            }
        }

        if cnbd_idxs.is_empty() {
            // v_i is a local maximum
            clusters.insert(i, vec![]);
        } else {
            let e_i_idx;
            match g_i {
                Some(k) => e_i_idx = root_idxs[k],
                None => e_i_idx = cmax_idx,
            }
            root_idxs[i] = e_i_idx;
            clusters.get_mut(&e_i_idx).unwrap().push(i);

            merge_(
                i,
                cmax_idx,
                &cnbd_idxs,
                &mut root_idxs,
                &mut clusters,
                density_map,
                threshold,
            );
        }
    }
    root_idxs
}

#[test]
fn test_greedy() {
    let graph = vec![vec![2], vec![0], vec![1], vec![1, 2]];
    let dm = vec![1.0, 2.0, 3.0, 2.5];
    let out = cluster_h0(&graph, &dm, 0.0, false);
    assert!(out.len() == dm.len());
}

#[test]
fn test_non_greedy() {
    let graph = vec![vec![2], vec![0], vec![1], vec![1, 2]];
    let dm = vec![1.0, 2.0, 3.0, 2.5];
    let out = cluster_h0(&graph, &dm, 0.0, true);
    assert!(out.len() == dm.len());
}
