mod clustering;
use clustering::zero_dim::merge_h0;

fn main() {
    let graph = vec![vec![2], vec![0], vec![1], vec![1, 2]];
    let dm = vec![1.0, 2.0, 3.0, 2.5];
    let out = merge_h0(graph, dm, 0.0);
    println!("{:?}", out);
}
