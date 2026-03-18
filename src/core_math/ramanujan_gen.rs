//! Graph generation and validation for the fixed `d = 3`, `n = 100` toy model.
//!
//! The generator intentionally avoids any opaque external graph library:
//! - candidates are created as the union of three edge-disjoint perfect matchings,
//! - each candidate is checked for simplicity, regularity, and connectivity,
//! - the adjacency spectrum is computed explicitly,
//! - the graph is accepted only when the second-largest eigenvalue by absolute value
//!   is within the Ramanujan bound `2 * sqrt(d - 1)`.
//!
//! This gives the project a concrete, inspectable topological certificate before any
//! learning logic is layered on top.

use std::collections::{HashSet, VecDeque};

use nalgebra::{DMatrix, SymmetricEigen};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use thiserror::Error;

use crate::{Edge, NODE_COUNT, REGULAR_DEGREE};

const LOCAL_MATCHING_RETRIES: usize = 512;
const DEFAULT_SEED_SEARCH_LIMIT: u64 = 50_000;
const SPECTRAL_EPSILON: f64 = 1.0e-9;

#[derive(Debug, Clone)]
pub struct SpectralCertificate {
    pub degree: usize,
    pub node_count: usize,
    pub search_seed: u64,
    pub largest_absolute_eigenvalue: f64,
    pub second_largest_absolute_eigenvalue: f64,
    pub ramanujan_bound: f64,
}

impl SpectralCertificate {
    pub fn is_ramanujan(&self) -> bool {
        self.second_largest_absolute_eigenvalue <= self.ramanujan_bound + SPECTRAL_EPSILON
    }
}

#[derive(Debug, Clone)]
pub struct VerifiedRamanujanGraph {
    pub edges: Vec<Edge>,
    pub certificate: SpectralCertificate,
}

#[derive(Debug, Error)]
pub enum GraphGenerationError {
    #[error("the current generator only supports even node counts; received {node_count}")]
    OddNodeCount { node_count: usize },
    #[error("the current generator is specialized for degree 3; received {degree}")]
    UnsupportedDegree { degree: usize },
    #[error("no verified Ramanujan graph was found in {attempts} deterministic seeds")]
    SearchExhausted { attempts: u64 },
}

#[derive(Debug, Error)]
pub enum GraphValidationError {
    #[error("self-loop detected at node {node}")]
    SelfLoop { node: usize },
    #[error("duplicate edge detected between nodes {u} and {v}")]
    DuplicateEdge { u: usize, v: usize },
    #[error("node {node} has degree {actual}, expected {expected}")]
    DegreeMismatch {
        node: usize,
        expected: usize,
        actual: usize,
    },
    #[error("graph is disconnected")]
    Disconnected,
}

/// Generate a verified `d = 3` Ramanujan candidate for the fixed 100-node MVP.
///
/// The search is deterministic even though candidate construction is randomized:
/// each seed maps to one candidate graph, and the lowest seed that satisfies the
/// validation rules is selected.
pub fn generate_cubic_ramanujan_graph_100() -> Result<VerifiedRamanujanGraph, GraphGenerationError>
{
    generate_verified_regular_graph(NODE_COUNT, REGULAR_DEGREE, DEFAULT_SEED_SEARCH_LIMIT)
}

/// Generate a verified `d = 3` regular graph and attach the spectral certificate
/// used to prove the topology satisfies the current Ramanujan constraint.
pub fn generate_verified_regular_graph(
    node_count: usize,
    degree: usize,
    search_limit: u64,
) -> Result<VerifiedRamanujanGraph, GraphGenerationError> {
    if node_count % 2 != 0 {
        return Err(GraphGenerationError::OddNodeCount { node_count });
    }

    if degree != REGULAR_DEGREE {
        return Err(GraphGenerationError::UnsupportedDegree { degree });
    }

    (0..search_limit)
        .into_par_iter()
        .filter_map(|seed| generate_candidate_from_seed(node_count, degree, seed))
        .find_first(|candidate| candidate.certificate.is_ramanujan())
        .ok_or(GraphGenerationError::SearchExhausted {
            attempts: search_limit,
        })
}

/// Build the adjacency matrix for a simple undirected graph.
pub fn adjacency_matrix(node_count: usize, edges: &[(usize, usize)]) -> DMatrix<f64> {
    let mut adjacency = DMatrix::zeros(node_count, node_count);

    for &(u, v) in edges {
        adjacency[(u, v)] = 1.0;
        adjacency[(v, u)] = 1.0;
    }

    adjacency
}

/// Validate a fixed-degree graph and compute the certificate that the topology
/// layer can retain as proof of the spectral constraint.
pub fn validate_graph(
    node_count: usize,
    degree: usize,
    edges: &[(usize, usize)],
    search_seed: u64,
) -> Result<SpectralCertificate, GraphValidationError> {
    let adjacency_lists = build_adjacency_lists(node_count, degree, edges)?;
    ensure_connected(&adjacency_lists)?;

    let spectrum = SymmetricEigen::new(adjacency_matrix(node_count, edges));
    let mut absolute_eigenvalues: Vec<f64> = spectrum
        .eigenvalues
        .iter()
        .map(|value| value.abs())
        .collect();

    absolute_eigenvalues.sort_by(|left, right| right.total_cmp(left));

    let largest_absolute_eigenvalue = absolute_eigenvalues[0];
    let second_largest_absolute_eigenvalue = absolute_eigenvalues[1];
    let ramanujan_bound = 2.0 * ((degree - 1) as f64).sqrt();

    Ok(SpectralCertificate {
        degree,
        node_count,
        search_seed,
        largest_absolute_eigenvalue,
        second_largest_absolute_eigenvalue,
        ramanujan_bound,
    })
}

fn generate_candidate_from_seed(
    node_count: usize,
    degree: usize,
    seed: u64,
) -> Option<VerifiedRamanujanGraph> {
    let edges = generate_edge_disjoint_matchings(node_count, degree, seed)?;
    let certificate = validate_graph(node_count, degree, &edges, seed).ok()?;

    Some(VerifiedRamanujanGraph { edges, certificate })
}

/// Candidate construction strategy:
/// each node must end with degree 3, so we repeatedly sample perfect matchings and
/// accept only matchings whose edges are new. The union of three such matchings is
/// automatically 3-regular if all edges are distinct.
fn generate_edge_disjoint_matchings(
    node_count: usize,
    degree: usize,
    seed: u64,
) -> Option<Vec<Edge>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut edges = Vec::with_capacity(node_count * degree / 2);
    let mut edge_set = HashSet::with_capacity(node_count * degree / 2);

    for _matching_index in 0..degree {
        let mut accepted = false;

        for _attempt in 0..LOCAL_MATCHING_RETRIES {
            let mut vertices: Vec<usize> = (0..node_count).collect();
            vertices.shuffle(&mut rng);

            let proposed_pairs: Vec<Edge> = vertices
                .chunks_exact(2)
                .map(|pair| normalized_edge(pair[0], pair[1]))
                .collect();

            if proposed_pairs
                .iter()
                .any(|candidate| edge_set.contains(candidate))
            {
                continue;
            }

            for edge in proposed_pairs {
                edge_set.insert(edge);
                edges.push(edge);
            }

            accepted = true;
            break;
        }

        if !accepted {
            return None;
        }
    }

    Some(edges)
}

fn build_adjacency_lists(
    node_count: usize,
    degree: usize,
    edges: &[(usize, usize)],
) -> Result<Vec<Vec<usize>>, GraphValidationError> {
    let mut adjacency_lists = vec![Vec::with_capacity(degree); node_count];
    let mut seen_edges = HashSet::with_capacity(edges.len());

    for &(u, v) in edges {
        if u == v {
            return Err(GraphValidationError::SelfLoop { node: u });
        }

        let edge = normalized_edge(u, v);
        if !seen_edges.insert(edge) {
            return Err(GraphValidationError::DuplicateEdge {
                u: edge.0,
                v: edge.1,
            });
        }

        adjacency_lists[u].push(v);
        adjacency_lists[v].push(u);
    }

    for (node, neighbors) in adjacency_lists.iter().enumerate() {
        if neighbors.len() != degree {
            return Err(GraphValidationError::DegreeMismatch {
                node,
                expected: degree,
                actual: neighbors.len(),
            });
        }
    }

    Ok(adjacency_lists)
}

fn ensure_connected(adjacency_lists: &[Vec<usize>]) -> Result<(), GraphValidationError> {
    let mut visited = vec![false; adjacency_lists.len()];
    let mut frontier = VecDeque::from([0usize]);
    visited[0] = true;

    while let Some(node) = frontier.pop_front() {
        for &neighbor in &adjacency_lists[node] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                frontier.push_back(neighbor);
            }
        }
    }

    if visited.into_iter().all(|flag| flag) {
        Ok(())
    } else {
        Err(GraphValidationError::Disconnected)
    }
}

fn normalized_edge(left: usize, right: usize) -> Edge {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_verified_cubic_graph_for_the_mvp_size() {
        let graph = generate_cubic_ramanujan_graph_100().expect("expected a verified graph");

        assert_eq!(graph.edges.len(), NODE_COUNT * REGULAR_DEGREE / 2);
        assert!(graph.certificate.is_ramanujan());
        assert!(
            (graph.certificate.largest_absolute_eigenvalue - REGULAR_DEGREE as f64).abs() < 1.0e-6
        );
    }
}
