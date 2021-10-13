use std;
use std::collections::{HashMap, HashSet};
use std::mem::drop;

pub fn merge_h0(
    neighbor_graph: &Vec<Vec<usize>>,
    density_map: &Vec<f32>,
    threshold: f32,
) -> Vec<usize> {
    assert!(
        neighbor_graph.len() == density_map.len(),
        "Neighbor graph and density map must have the same length."
    );

    let indices: Vec<_> = (0..density_map.len()).collect();
    // sort the vertices in descending order of density
    let mut pairs: Vec<_> = density_map.iter().zip(indices).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());
    let (_, sort_idxs): (Vec<f32>, Vec<usize>) = pairs.iter().cloned().unzip();
    // indicates the root index to which each vertex is assigned
    let mut root_idxs: Vec<usize> = (0..density_map.len()).collect();
    // mapping between root indexes and child indexes.
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();

    for &i in sort_idxs.iter() {
        let nbd_idxs: Vec<&usize> = neighbor_graph[i]
            .iter()
            .filter(|x| density_map[**x] > density_map[i])
            .collect::<Vec<&usize>>();

        if nbd_idxs.is_empty() {
            // v_i is a local maximum
            clusters.insert(i, vec![]);
        } else {
            // index of the neighboring vertex with the highest density
            let mut cmax_idx: usize = 0;
            let mut cmax_d: f32 = -f32::INFINITY;
            // indexes of the clusters to which the neighboring vertices
            // currently belong.
            let mut cnbd_idxs = HashSet::with_capacity(nbd_idxs.len());

            for &nbd_idx in nbd_idxs.iter() {
                let root_idx = root_idxs[*nbd_idx];
                cnbd_idxs.insert(root_idx);
                // If the density of the root vertex is greater than
                // the highest density up until this point, that vertex
                // becomes the new cmax.
                let root_d = density_map[root_idx];
                if root_d > cmax_d {
                    cmax_idx = root_idx;
                    cmax_d = root_d;
                }
            }
            // cmax_d is no longer needed
            drop(cmax_d);
            // exclude cmax from the subsequent interation
            cnbd_idxs.remove(&cmax_idx);

            for &cnbd_idx in cnbd_idxs.iter() {
                // compute the persistence between each root vertex and the
                // current vertex
                let persistence = density_map[cnbd_idx] - density_map[i];
                // if the persistence is below the user-defined threshold,
                // then merge the root vertex into cmax.
                if persistence < threshold {
                    if let Some(mut c) = clusters.remove(&cnbd_idx) {
                        // clusters.get_mut(&cmax_idx).unwrap().append(&mut c);
                        let cmax = clusters.get_mut(&cmax_idx);
                        match cmax {
                            None => print!("cmax: {:?}, cmax_idx: {:?}", cmax, cmax_idx),
                            Some(i) => i.append(&mut c),
                        }
                        for &elem in c.iter() {
                            root_idxs[elem] = cmax_idx;
                        }
                    }
                    root_idxs[cnbd_idx] = cmax_idx;
                }
            }
            root_idxs[i] = cmax_idx;
            clusters.get_mut(&cmax_idx).unwrap().push(i);
        }
    }
    root_idxs
}

#[test]
fn test_merge_h0() {
    let graph = vec![vec![2], vec![0], vec![1], vec![1, 2]];
    let dm = vec![1.0, 2.0, 3.0, 2.5];
    let out = merge_h0(&graph, &dm, 0.0);
    assert!(out.len() == dm.len());
}
