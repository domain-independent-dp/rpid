use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

pub fn fx_hash_assign_thread<K: Hash>(value: &K, threads: usize, seed: u32) -> usize {
    let mut hasher = FxHasher::default();
    hasher.write_u32(seed);
    value.hash(&mut hasher);
    (hasher.finish() as usize) % threads
}
