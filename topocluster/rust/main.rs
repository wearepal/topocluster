mod clustering;
use clustering::zero_dim::cluster_h0;

fn main() {
    let graph = vec![
        vec![1, 3],
        vec![0],
        vec![0, 1],
        vec![1, 3],
        vec![1, 2],
        vec![2, 3, 4],
        vec![5, 6, 0],
        vec![2, 1, 4],
        vec![5, 4, 0],
    ];
    let dm = vec![1.0, 2.0, 3.0, 2.5, 5.0, 3.5, 4.1, 0.5, 2.7];
    let out = cluster_h0(&graph, &dm, 10.0, false);
    println!("{:?}", out);
}
