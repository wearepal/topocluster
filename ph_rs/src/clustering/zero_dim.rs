use std;
use std::collections::HashMap;
use std::mem::drop;

pub fn merge_h0(graph: &Vec<Vec<usize>>, dm: &Vec<f32>, threshold: f32) -> Vec<usize> {
    let indices: Vec<_> = (0..dm.len()).collect();
    let mut pairs: Vec<_> = dm.iter().zip(indices).collect();
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(a.0).unwrap());
    let (_, inds_s): (Vec<f32>, Vec<usize>) = pairs.iter().cloned().unzip();

    let mut root_inds: Vec<usize> = (0..dm.len()).collect();
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();

    for &i in inds_s.iter() {
        let nbd: Vec<&usize> = graph[i]
            .iter()
            .filter(|x| dm[**x] > dm[i])
            .collect::<Vec<&usize>>();
        if !nbd.is_empty() {
            let mut cmax: usize = 0;
            let cmax_d: f32 = -f32::INFINITY;
            let mut cnbd = vec![0; nbd.len()];
            for &j in nbd.iter() {
                let root_ind = root_inds[*j];
                cnbd.push(root_ind);
                if dm[root_ind] > cmax_d {
                    cmax = root_ind
                }
            }
            drop(cmax_d);

            for &k in cnbd.iter() {
                if !k == cmax {
                    let persistence = dm[k] - dm[i];
                    if persistence < threshold {
                        let mut c = clusters[&k].clone();
                        clusters.get_mut(&cmax).unwrap().append(&mut c);
                        clusters.remove_entry(&k);
                        root_inds[k] = cmax;
                    }
                }
            }
            root_inds[i] = cmax;
            clusters.entry(cmax).or_insert(vec![]).push(i);
        }
    }
    root_inds
}
