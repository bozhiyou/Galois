/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2019, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <utility>

// Silence erroneous warnings from within Boost headers
// that show up with gcc 8.1.
#pragma GCC diagnostic ignored "-Wparentheses"
// This warning triggers with the assert(("explanation", check));
// syntax since the left hand argument has no side-effects.
// I prefer using the comma operator over && though because
// the parentheses are more readable, so I'm silencing
// the warning for this file.
#pragma GCC diagnostic ignored "-Wunused-value"

#include <galois/Galois.h>
#include <galois/graphs/FileGraph.h>
#include <galois/graphs/Graph.h>
#include <galois/graphs/LCGraph.h>

#include <galois/AtomicHelpers.h>

// Vendored from an old version of LLVM for Lonestar app command line handling.
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "galois/runtime/Profile.h"

static char const* name = "FMM";
static char const* desc =
    "fmm";
static char const* url = "";

using CoordTy = double;
using SlnTy = double;

enum Algo { serial = 0, parallel, partition, bipartiteSync };
enum Source { scatter = 0, analytical };

const char* const ALGO_NAMES[] = { "serial", "parallel", "partition", "bipartiteSync" };

static llvm::cl::opt<Algo>
    algo("algo", llvm::cl::desc("Choose an algorithm:"),
         llvm::cl::values(
           clEnumVal(serial, "serial"),
           clEnumVal(parallel, "parallel"),
           clEnumVal(partition, "partition"),
           clEnumVal(bipartiteSync, "bipartiteSync"),
           clEnumValEnd),
         llvm::cl::init(parallel));
static llvm::cl::opt<Source>
    source_type("source", llvm::cl::desc("Choose an sourceType:"),
         llvm::cl::values(
           clEnumVal(scatter, "a set of discretized points"),
           clEnumVal(analytical, "boundary in a analytical form"),
           clEnumValEnd),
         llvm::cl::init(analytical));
static llvm::cl::opt<unsigned long long> nh{
    "nh", llvm::cl::desc("number of cells in ALL direction"),
    llvm::cl::init(0u)};
static llvm::cl::opt<unsigned long long> nx{
    "nx", llvm::cl::desc("number of cells in x direction"),
    llvm::cl::init(10u)};
static llvm::cl::opt<unsigned long long> ny{
    "ny", llvm::cl::desc("number of cells in y direction"),
    llvm::cl::init(10u)};
static llvm::cl::opt<unsigned long long> nz{
    "nz", llvm::cl::desc("number of cells in z direction"),
    llvm::cl::init(10u)};
static llvm::cl::opt<unsigned> RF{
    "rf", llvm::cl::desc("round-off factor for OBIM"),
    llvm::cl::init(0u)};
// static llvm::cl::opt<CoordTy> num_groups{
//     "num_groups", llvm::cl::desc("number of frequency groups"),
//     llvm::cl::init(4u)};
// static llvm::cl::opt<CoordTy> num_vert_directions{
//     "num_vert_directions", llvm::cl::desc("number of vertical directions"),
//     llvm::cl::init(16u)};
// static llvm::cl::opt<CoordTy> num_horiz_directions{
//     "num_horiz_directions", llvm::cl::desc("number of horizontal directions."),
//     llvm::cl::init(32u)};
// static llvm::cl::opt<unsigned long long> num_iters{
//    "num_iters", llvm::cl::desc("number of iterations"), llvm::cl::init(10u)};
// static llvm::cl::opt<double> pulse_strength{
//    "pulse_strength", llvm::cl::desc("radiation pulse strength"),
//    llvm::cl::init(1.)};
// static llvm::cl::opt<double> absorption_coef{
//     "absorption_coef",
//     llvm::cl::desc("Absorption coefficient (between 0 and 1), absorption and "
//                    "scattering must sum to less than 1."),
//     llvm::cl::init(.01)};
// static llvm::cl::opt<double> scattering_coef{
//     "scattering_coef",
//     llvm::cl::desc("Scattering coefficient (between 0 and 1), absorption and "
//                    "scattering must sum to less than 1."),
//     llvm::cl::init(.25)};
// static llvm::cl::opt<bool> print_convergence{
//     "print_convergence",
//     llvm::cl::desc("Print the max change in amount of scattering at a given "
//                    "each iteration."),
//     llvm::cl::init(false)};
// static llvm::cl::opt<std::string> scattering_outfile{
//     "scattering_outfile",
//     llvm::cl::desc(
//         "Text file name to use to write final scattering term values "
//         "after each step."),
//     llvm::cl::init("")};


static constexpr CoordTy xa = -.5, xb = .5;
static constexpr CoordTy ya = -.5, yb = .5;
static constexpr CoordTy za = -.5, zb = .5;

static std::size_t NUM_CELLS;
static CoordTy dx, dy, dz;

///////////////////////////////////////////////////////////////////////////////

// Idk why this hasn't been standardized in C++ yet, but here it is.
static constexpr double PI =
    3.1415926535897932384626433832795028841971693993751;
constexpr SlnTy INF = std::numeric_limits<SlnTy>::max();
constexpr galois::MethodFlag no_lockable_flag = galois::MethodFlag::UNPROTECTED;

// TODO: In Galois, we need a graph type with dynamically sized
// node/edge data for this problem. For now, indexing into a
// separate data structure will have to be sufficient.

// Note: I'm going to use a CSR graph, so each node will already have a
// unique std::size_t id that can be used to index other data structures.
// I'll also use a std::size_t cutoff to distinguish between ghost cells
// that only exist to provide boundary condition data and actual cells.

// Each edge holds the unit normal pointing outward
// from the corresponding source cell in the graph.
// Note: this will be the negative of the vector stored
// on the edge coming the opposite direction.
// Note: In the regular grid case, this could be considered redundant,
// but this code hopefully will be adapted to handle irregular
// geometry at some point.
// Note: The sweeping direction for each direction along each edge
// could just be pre-computed, but that'd noticeably increase
// storage requirements.
// TODO: Try caching sweep directions and see if it's any better.

// Both these limitations could be lifted,
// but in the interest of keeping the buffer management
// code simple, I'm just going to assume them.
static_assert(sizeof(std::atomic<std::size_t>) <= sizeof(double),
              "Current buffer allocation code assumes atomic "
              "counters smaller than sizeof(double).");
static_assert(std::is_trivial_v<std::atomic<std::size_t>> &&
                  std::is_standard_layout_v<std::atomic<std::size_t>>,
              "Current buffer allocation code assumes no special "
              "construction/deletion code is needed for atomic counters.");
///////////////////////////////

// FMM
// enum Tag { KNOWN_FIX, KNOWN_OLD, KNOWN_NEW,
//            BAND_OLD, BAND_NEW, FAR };
enum Tag { KNOWN, BAND, FAR };

struct NonAtomicNodeData {
  bool is_ghost;
  Tag tag;
  double speed;
  double solution;
};

struct NodeData {
  bool is_ghost; // read only
  std::atomic<Tag> tag;
  double speed; // read only
  std::atomic<double> solution;
};
auto constexpr atomic_order = std::memory_order_relaxed;
// No fine-grained locks built into the graph.
// Use atomics for ALL THE THINGS!
using Graph = galois::graphs::LC_CSR_Graph<NodeData, void>
                ::with_no_lockable<true>::type
                ;
using GNode = Graph::GraphNode;

template <typename HeapTy>
struct FMMHeapWrapper {
  typedef typename HeapTy::key_type key_type;
  typedef typename HeapTy::value_type value_type;
  HeapTy wl;

  inline auto begin() { return wl.begin(); }
  inline auto end() { return wl.end(); }

  inline bool empty() { return wl.empty(); }

  void push(const value_type& p, const key_type& old_sln = 0) {
    auto [sln_temp, dst] = p;
    auto iter = wl.lower_bound(old_sln);
    for (; iter != wl.end(); std::advance(iter, 1)) {
      // if (dst == 274) {
      //   galois::gDebug("274: ", old_sln, " iter: ", iter->first, " ", iter->second);
      // }
      if (iter->second == dst) {
      // if (dst == 274) {
      //   galois::gDebug("274 catch: ", old_sln, " iter: ", iter->first, " ", iter->second);
      // }
        
        break;
      }
      if (iter->first != old_sln) {
        iter = wl.end();
        break;
      }
    }
      //  if (dst == 274) {
      //    galois::gDebug("274 finished: ", old_sln, " iter: ", iter->first, " ", iter->second);
      //  }
    if (iter == wl.end()) {
      // if (dst == 274) {
      //   galois::gDebug("dumping heap ...");
      //   for (auto i : wl) {
      //     galois::gDebug(i.first, " ", i.second);
      //     assert(i.second != 274);
      //   }
      //   galois::gDebug(dstData.tag);
      //   assert(dstData.tag == BAND);
      // }
      wl.insert({sln_temp, dst});
    } else {
      // if (dst == 274) assert(dstData.tag == BAND);
      auto nh = wl.extract(iter); // node handle
      nh.key() = sln_temp;
      wl.insert(std::move(nh));
    }
  }

  value_type pop() {
    auto pair = *(wl.begin());
    wl.erase(wl.begin()); // TODO serial only
    return pair;
  }
};
using BL = galois::InsertBag<GNode>;
using UpdateRequest = std::pair<SlnTy, GNode>;
using HeapTy = FMMHeapWrapper<std::multimap<UpdateRequest::first_type,
                                            UpdateRequest::second_type>>;
using WL = galois::InsertBag<UpdateRequest>;

/**
 * Convert node ID into grid coordinate tuple.
 */
std::array<std::size_t, 3> getPos(const GNode node) {
  if (node >= NUM_CELLS) return {};
  auto res = std::lldiv(node, nx * ny);
  CoordTy i = res.quot;
  res = std::lldiv(res.rem, ny);
  CoordTy j = res.quot;
  CoordTy k = res.rem;

  assert(i < nz && j < nx && k < ny);
  return std::array<std::size_t, 3>({j, k, i});
}
/**
 * Convert node ID into coordinate tuple.
 * `id = i * nx * ny + j * ny + k`
 * `(i, j, k) = (z, x, y)`
 * @param node Node ID
 * @return Coordinate tuple (x, y, z)
 */
std::array<CoordTy, 3> getCoord(const GNode node) {
  if (node >= NUM_CELLS) return {};
  auto res = std::lldiv(node, nx * ny);
  CoordTy i = res.quot;
  res = std::lldiv(res.rem, ny);
  CoordTy j = res.quot;
  CoordTy k = res.rem;

  CoordTy x = xa + dx * (j + 1);
  CoordTy y = ya + dy * (k + 1);
  CoordTy z = za + dz * (i + 1);
  // galois::gDebug(i, " ", j, " ", k);
  // galois::gDebug(dx, " - ", dy, " - ", dz);
  assert(x < xb && y < yb && z < zb);
  return std::array<CoordTy, 3>({x, y, z});
}

/**
 * Convert coordinate tuple into node ID.
 * See getCoord
 */
GNode getNodeID(const std::array<CoordTy, 3>& coords) {
  uint64_t i = std::round((coords[2] - za) / dz - 1.);
  uint64_t j = std::round((coords[0] - xa) / dx - 1.);
  uint64_t k = std::round((coords[1] - ya) / dy - 1.);

  return i * nx * ny + j * ny + k;
}

///////////////////////////////////////////////////////////////////////////////

// TODO make it external and customizable
// TODO multi-dimension configuration
template <typename CoordTy>
static double SpeedFunction(std::array<CoordTy, 3> coords) {
  const CoordTy& x = coords[0], y = coords[1], z = coords[2];

  return 1.;
  // return 1. + .50 * std::sin(20. * PI * x) * std::sin(20. * PI * y) * std::sin(20. * PI * z);
  // return 1. - .99 * std::sin(2. * PI * x) * std::sin(2. * PI * y) * std::sin(2. * PI * z);
}

static double BoundaryCondition(std::array<CoordTy, 3> coords = {}) {
  return 0.;
}

static bool NonNegativeRegion(const std::array<CoordTy, 3>& coords) {
  const CoordTy& x = coords[0], y = coords[1], z = coords[2];

  // Example 1: a spherical interface of radius 0.25 centered at the origin
  // return x * x + y * y + z * z >= .25 * .25;

  // Example 2: a plane past through the origin
  return 100. * x + y + 2. * z >= 0.;
}

template <typename Graph, typename BL,
          typename T = typename BL::value_type>
void AssignBoundary(Graph& graph, BL& boundary) {
  galois::do_all(
    galois::iterate(0ul, NUM_CELLS),
    [&](T node) noexcept {
      if (node > NUM_CELLS) return;

      if (NonNegativeRegion(getCoord(node))) {
        for (auto e : graph.edges(node, no_lockable_flag)) {
          GNode dst = graph.getEdgeDst(e);
          if (!NonNegativeRegion(getCoord(dst))) {
// #ifndef NDEBUG
//             auto c = getCoord(node);
//             galois::gDebug(node, " (", c[0], " ", c[1], " ", c[2], ")");
// #endif
            boundary.push(node);
            break;
          }
        }
      }
    },
    galois::loopname("assignBoundary"));
}

// template <typename WL>
// void AssignBoundary(WL& boundary) {
//   for (GNode i = 0; i < nx * ny; i++) {
//     boundary.push(i);
//   }
// }

template <typename BL>
void AssignBoundary(BL& boundary) {
// #ifndef NDEBUG
//   GNode n = getNodeID({0., 0., 0.});
//   auto c = getCoord(n);
//   galois::gDebug(n, " (", c[0], " ", c[1], " ", c[2], ")");
// #endif
  boundary.push(getNodeID({0., 0., 0.}));
}

/////////////////////////////////////////////

// Routine to initialize graph topology and face normals.
auto generate_grid(Graph& built_graph, std::size_t nx, std::size_t ny,
                   std::size_t nz) noexcept {

  galois::graphs::FileGraphWriter temp_graph;
  // Each node represents a grid cell.
  // Each edge represents a face.
  // Ghost nodes are added to represent the exterior
  // of the domain on the other side of each face.
  // This is for boundary condition handling.
  std::size_t num_outer_faces = (nx * ny + ny * nz + nx * nz) * 2;
  std::size_t num_cells       = nx * ny * nz;
  std::size_t num_nodes       = num_cells + num_outer_faces;
  std::size_t num_edges       = 6 * nx * ny * nz + num_outer_faces;
  temp_graph.setNumNodes(num_nodes);
  temp_graph.setNumEdges(num_edges);
  temp_graph.setSizeofEdgeData(0);

  // Interior cells will have degree 6
  // since they will have either other cells or
  // ghost cells on every side.
  // This condition isn't true in irregular meshes,
  // but that'd be a separate mesh generation routine.
  temp_graph.phase1();
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        std::size_t id = i * nx * ny + j * ny + k;
        for (std::size_t l = 0; l < 6; l++) {
          temp_graph.incrementDegree(id);
        }
      }
    }
  }
  // Set the degrees for all the ghost cells to 1.
  // No ghost cell should share a boundary with more
  // than one actual cell in the domain.
  for (std::size_t id = num_cells; id < num_nodes; id++) {
    temp_graph.incrementDegree(id);
  }

  // Now that the degree of each node is known,
  // fill in the actual topology.
  // Also fill in the node data with the vector
  // normal to the face, going out from the current cell.
  temp_graph.phase2();
  std::size_t xy_low_face_start  = num_cells;
  std::size_t xy_high_face_start = xy_low_face_start + nx * ny;
  std::size_t yz_low_face_start  = xy_high_face_start + nx * ny;
  std::size_t yz_high_face_start = yz_low_face_start + ny * nz;
  std::size_t xz_low_face_start  = yz_high_face_start + ny * nz;
  std::size_t xz_high_face_start = xz_low_face_start + nx * nz;
  assert(("Error in logic for dividing up node ids for exterior faces.",
          num_nodes == xz_high_face_start + nx * nz));
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        // (i, j, k) = (z, x, y)
        std::size_t id = i * nx * ny + j * ny + k;
        if (i > 0) {
          temp_graph.addNeighbor(id, id - ny * nz);
        } else {
          std::size_t ghost_id = yz_low_face_start + j * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (i < nx - 1) {
          temp_graph.addNeighbor(id, id + ny * nz);
        } else {
          std::size_t ghost_id = yz_high_face_start + j * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (j > 0) {
          temp_graph.addNeighbor(id, id - nz);
        } else {
          std::size_t ghost_id = xz_low_face_start + i * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (j < ny - 1) {
          temp_graph.addNeighbor(id, id + nz);
        } else {
          std::size_t ghost_id = xz_high_face_start + i * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (k > 0) {
          temp_graph.addNeighbor(id, id - 1);
        } else {
          std::size_t ghost_id = xy_low_face_start + i * ny + j;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (k < nz - 1) {
          temp_graph.addNeighbor(id, id + 1);
        } else {
          std::size_t ghost_id = xy_high_face_start + i * ny + j;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
      }
    }
  }

  // TODO: is it possible to set the edge data
  // during construction without copying here?
  temp_graph.finish<Graph::edge_data_type>();
  //auto* rawEdgeData = temp_graph.finish<Graph::edge_data_type>();
  //std::uninitialized_copy(std::make_move_iterator(edge_data.begin()),
  //                        std::make_move_iterator(edge_data.end()),
  //                        rawEdgeData);

  galois::graphs::readGraph(built_graph, temp_graph);
  return std::make_tuple(num_nodes, num_cells, num_outer_faces,
                         xy_low_face_start, xy_high_face_start,
                         yz_low_face_start, yz_high_face_start,
                         xz_low_face_start, xz_high_face_start);
}

///////////////////////////////////////////////////////////////////////////////

void initCells(Graph& graph, size_t num_cells) {
  galois::do_all(
    galois::iterate(graph.begin(), graph.end()),
    [&](GNode node) noexcept {
      auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      curData.is_ghost = (node >= num_cells);
      curData.tag = FAR;
      curData.speed = SpeedFunction(getCoord(node));
      curData.solution = INF; // TODO ghost init?
    },
    galois::no_stats(),
    galois::loopname("initializeCells"));
}

template <typename Graph, typename BL,
          typename T = typename BL::value_type>
void initBoundary(Graph& graph, BL& boundary) {
  galois::do_all(
    galois::iterate(boundary.begin(), boundary.end()),
    [&](const T& node) noexcept {
      auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      curData.tag = KNOWN;
      curData.solution = BoundaryCondition(getCoord(node));
    },
    galois::no_stats(),
    galois::loopname("initializeBoundary"));
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Solver

auto checkDirection(Graph& graph, GNode node, double center_sln, Graph::edge_iterator dir) {
#ifndef NDEBU
  if (dir >= graph.edge_end(node, no_lockable_flag)) {
    galois::gDebug(node, " ", (graph.getData(node, no_lockable_flag).is_ghost? "ghost":"non-ghost"), " ",
                     std::distance(graph.edge_begin(node, no_lockable_flag), graph.edge_end(node, no_lockable_flag)) );
    GALOIS_DIE("invalid direction");
  }
#endif
  SlnTy sln = center_sln;
  GNode upwind = node;
  GNode neighbor = graph.getEdgeDst(dir);
  auto& first_data = graph.getData(neighbor, no_lockable_flag);
  galois::gDebug("Check neighbor ", neighbor, (int)first_data.tag);
  // if (first_data.tag == KNOWN)
    if (first_data.solution < sln) {
    sln = first_data.solution;
    upwind = neighbor;
  }
  std::advance(dir, 1); // opposite direction of the same dimension
  if (dir != graph.edge_end(node, no_lockable_flag)) {
    neighbor = graph.getEdgeDst(dir);
    auto& second_data = graph.getData(neighbor, no_lockable_flag);
    galois::gDebug("Check neighbor ", neighbor, (int)second_data.tag);
    // if (second_data.tag == KNOWN)
      if (second_data.solution < sln) {
      sln = second_data.solution;
      upwind = neighbor;
    }
  }
  if (upwind == node)
    return std::make_pair(0., 0.);
  return std::make_pair(sln, dx);
}

double solveQuadratic(Graph& graph, GNode node, double sln, const double speed) {
  // TODO oarameterize dimension 3
  std::array<std::pair<double, double>, 3> sln_delta {
                                            std::make_pair(0., dx),
                                            std::make_pair(0., dy),
                                            std::make_pair(0., dz)
                                          };
  int non_zero_counter = 0;
  auto dir = graph.edge_begin(node, no_lockable_flag);
  for (auto& p : sln_delta) {
    if (dir == graph.edge_end(node, no_lockable_flag))
      break;
    double& s = p.first;
    double& d = p.second;
    auto [si, di] = checkDirection(graph, node, sln, dir);
    if (di) {
      s = si;
      non_zero_counter++;
    }
    else {
      // s = 0.; // already there
      d = 0.;
    }
    std::advance(dir, 2);
  }
  galois::gDebug("solveQuadratic: ", sln_delta[0].second, " ", sln_delta[1].second, " ", sln_delta[2].second,
    " #non_zero: ", non_zero_counter);
  if (non_zero_counter == 0)
    return INF;
  while (non_zero_counter) {
    auto max_s_d_it = std::max_element(sln_delta.begin(), sln_delta.end(),
        [&](std::pair<double, double>& a, std::pair<double, double>& b) {
          return a.first < b.first;
        });
    double a(0.), b(0.), c(0.);
    for (const auto& p : sln_delta) {
      const double& s = p.first, d = p.second;
      galois::gDebug(s, " ", d);
      double temp = (d == 0.? 0. : (1. / (d * d)));
      a += temp;
      temp *= s;
      b += temp;
      temp *= s;
      c += temp;
      galois::gDebug("tabs: ", temp, " ", a, " ", b, " ", c);
    }
    b *= -2.;
    c -= (1. / (speed * speed));
    double del = b * b - (4. * a * c);
    galois::gDebug(a, " ", b, " ", c, " del=", del);
    if (del >= 0) {
      double new_sln = (-b + std::sqrt(del)) / (2. * a);
      galois::gDebug("new solution: ", new_sln);
      if (new_sln > max_s_d_it->first)
        sln = std::min(sln, new_sln); // TODO atomic here?
    }
    max_s_d_it->first = 0.;
    max_s_d_it->second = 0.;
    non_zero_counter--;
  }
  return sln;
}

double SerialSolveQuadratic(Graph& graph, GNode node, double sln, const double speed) {
  // TODO oarameterize dimension 3
  std::array<std::pair<double, double>, 3> sln_delta {
                                            std::make_pair(0., dx),
                                            std::make_pair(0., dy),
                                            std::make_pair(0., dz)
                                          };
  int non_zero_counter = 0;
  auto dir = graph.edge_begin(node, no_lockable_flag);
  for (auto& p : sln_delta) {
    if (dir == graph.edge_end(node, no_lockable_flag))
      break;
    double& s = p.first;
    double& d = p.second;
    auto [si, di] = checkDirection(graph, node, sln, dir);
    if (di) {
      s = si;
      non_zero_counter++;
    }
    else {
      // s = 0.; // already there
      d = 0.;
    }
    std::advance(dir, 2);
  }
  if (non_zero_counter == 0)
    return INF;
  // galois::gDebug("SerialSolveQuadratic: ", sln_delta[0].second, " ", sln_delta[1].second, " ", sln_delta[2].second);
  // for (; non_zero_counter > 0; non_zero_counter--) {
    // auto max_sln_it = std::max_element(solutions.begin(), solutions.end());
    double a(0.), b(0.), c(0.);
    for (const auto& p : sln_delta) {
      const double& s = p.first, d = p.second;
      galois::gDebug(s, " ", d);
      double temp = (d == 0.? 0. : (1. / (d * d)));
      a += temp;
      temp *= s;
      b += temp;
      temp *= s;
      c += temp;
      // galois::gDebug(temp, " ", a, " ", b, " ", c);
    }
    b *= -2.;
    c -= (1. / (speed * speed));
    double del = b * b - (4. * a * c);
    galois::gDebug(a, " ", b, " ", c, " del=", del);
    if (del >= 0) {
      double new_sln = (-b + std::sqrt(del)) / (2. * a);
      galois::gDebug("new solution: ", new_sln);
      // if (new_sln > *max_sln_it)
      //  sln = std::min(sln, new_sln); // TODO atomic here?
      return new_sln;
    }
    // *max_sln_it = 0.;
  // }
  return INF;
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// first iteration

template <bool CONCURRENT,
          typename Graph, typename BL, typename WL,
          typename BT = typename BL::value_type,
          typename WT = typename WL::value_type>
void FirstIteration(Graph& graph, BL& boundary, WL& init_wl) {
  using Loop = typename
    std::conditional<CONCURRENT, galois::DoAll, galois::StdForEach>::type;
  Loop loop;

  loop(
    galois::iterate(boundary.begin(), boundary.end()),
    [&](BT node) noexcept {
      for (auto e : graph.edges(node, no_lockable_flag)) {
        GNode dst = graph.getEdgeDst(e);
        if (dst < NUM_CELLS) {
          auto& dstData = graph.getData(dst, no_lockable_flag);
          assert(!dstData.is_ghost);
          if (dstData.tag != KNOWN) {
            assert(dstData.solution.load(atomic_order) == INF || dstData.tag.load(atomic_order) != FAR);
            SlnTy old_sln = dstData.solution.load(atomic_order);
            SlnTy sln_temp = CONCURRENT? 
              solveQuadratic(graph, dst, old_sln, dstData.speed) :
              SerialSolveQuadratic(graph, dst, old_sln, dstData.speed);
            if (sln_temp < galois::atomicMin(dstData.solution, sln_temp)) {
              galois::gDebug("Hi! I'm ", dst, " I got ", sln_temp);
              if (auto old_tag = dstData.tag.load(atomic_order); old_tag != BAND) {
                while (!dstData.tag.compare_exchange_weak(
                                     old_tag, BAND, std::memory_order_relaxed));
                if constexpr (CONCURRENT)
                  init_wl.push(WT{sln_temp, dst});
                else
                  init_wl.push(WT{sln_temp, dst}, old_sln);
#ifndef NDEBUG
              } else {
                if constexpr (CONCURRENT) {
                  bool in_wl = false;
                  for (WT [_, i] : init_wl) {
                    if (i == node) {
                      in_wl = true;
                      break;
                    }
                  }
                  assert(in_wl);
                }
#endif
              }
            }
          }
        }
      }
    },
    galois::loopname("FirstIteration"));
}

template <typename Graph, typename BL, typename WL>
void BipartInitializeBag(Graph& graph, BL& boundary, WL& oddBag, WL& evenBag) {
  galois::do_all(
    galois::iterate(boundary.begin(), boundary.end()),
    [&](GNode node) noexcept {
      for (auto e : graph.edges(node, no_lockable_flag)) {
        GNode dst = graph.getEdgeDst(e);
        if (dst < NUM_CELLS) {
          auto& dstData = graph.getData(dst, no_lockable_flag);
          assert(!dstData.is_ghost);
          if (dstData.tag != KNOWN) {
            assert(dstData.solution == INF);
            assert(dstData.tag == FAR);
            SlnTy old_sln = dstData.solution;
            double sln_temp = solveQuadratic(graph, dst, old_sln, dstData.speed);
            if (sln_temp < old_sln) {
              auto [l, m, n] = getPos(dst);
              bool odd = (l+m+n) & 1u;
              galois::gDebug("Hi! I'm ", dst, " pos(", l, " ", m, " ", n, ")", odd?"1":"0", " I got ", sln_temp);
              WL& initBag = odd? oddBag : evenBag;
              dstData.solution = sln_temp;
              if (dstData.tag == BAND) {
#ifndef NDEBUG
                bool in_wl = false;
                for (GNode i : initBag) {
                  if (i == node) {
                    in_wl = true;
                    break;
                  }
                }
                assert(in_wl);
#endif
              } else {
                dstData.tag = BAND;
                initBag.push(dst);
              }
            }
          }
        }
      }
    },
    galois::loopname("BipartInitializeBag"));
}


////////////////////////////////////////////////////////////////////////////////

// TODO
template <typename RangeFunc, typename FunctionTy, typename... Args>
void while_wrapper(const RangeFunc& rangeMaker, FunctionTy &&fn,
    const Args&... args) {
      auto tpl = std::make_tuple(args...);
        // runtime::for_each_gen(rangeMaker(tpl), std::forward<FunctionTy>(fn), tpl);       
}


////////////////////////////////////////////////////////////////////////////////
// Operator

////////////////////////////////////////////////////////////////////////////////
// FMM

template <bool CONCURRENT,
          typename Graph, typename WL,
          typename T = typename WL::value_type>
void FastMarching(Graph& graph, WL& wl) {
  double max_error = 0;
  std::size_t num_iterations = 0;

auto SerialOp = [&]<typename ItemTy, typename UC>
  (const ItemTy& item, UC& wl) {
    auto [_old_sln, node] = item;
    assert(node < NUM_CELLS && "Ghost Point!");
    auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
#ifndef NDEBUG
    {
      assert(!curData.is_ghost && "impossible, asserted before");
      auto [x, y, z] = getCoord(node);
      galois::gDebug("Hi! I'm ", node, " (", x, " ", y, " ", z, " ) with ", curData.solution);
      assert(curData.solution != INF);
    }
if constexpr (CONCURRENT) {
    if (curData.tag == KNOWN) {
      galois::gDebug(node, " in bag as KNWON");
    }
    assert(curData.tag == BAND);
} else {
    assert(curData.tag != KNOWN);
    if (curData.tag != BAND) {
      galois::gDebug(node, " in heap with tag ", curData.tag);
      std::abort();
    }

    // if (curData.solution != _old_sln) {
    //   galois::gDebug("Wrong entry in the heap! ", node, " ", _old_sln, " but ", curData.solution);
    //   for(auto i : wl) {
    //     if (i.second == node)
    //       galois::gDebug(i.first);
    //   }
    //   std::abort();
    // }

    {
      auto [x, y, z] = getCoord(node);
      if (curData.solution - std::sqrt(x * x + y * y + z * z) > max_error)
        max_error = curData.solution - std::sqrt(x * x + y * y + z * z);
      if (curData.solution - std::sqrt(x * x + y * y + z * z) > 0.2) {
        galois::gDebug(curData.solution - std::sqrt(x * x + y * y + z * z),
          " - wrong distance, should be ", std::sqrt(x * x + y * y + z * z));
        assert(false);
      }
    }
}
#endif
    curData.tag.store(KNOWN, atomic_order);

    // UpdateNeighbors
    for (auto e : graph.edges(node, no_lockable_flag)) {
      GNode dst = graph.getEdgeDst(e);
      if (dst < NUM_CELLS) {
        auto& dstData = graph.getData(dst, no_lockable_flag);
        assert(!dstData.is_ghost);
        // chaotic execution: KNOWN neighbor doesn't suffice with smaller value
        //   reason: circle-back update
        //   ... but it holds with ordered serial execution
        // smaller value doesn't suffice to be KNOWN - by-pass to be active
        if (dstData.solution > curData.solution) {
#ifndef NDEBUG
          {
            auto [x, y, z] = getCoord(dst);
            galois::gDebug("Update ", dst, " (", x, " ", y, " ", z, " )");
          }
#endif
          // assert(dstData.solution == INF && dstData.tag == FAR);
          galois::gDebug("tag ", dstData.tag, ", sln ", dstData.solution);
          SlnTy old_sln = dstData.solution.load(atomic_order);
          assert(old_sln > curData.solution);
          double sln_temp = CONCURRENT?
            solveQuadratic(graph, dst, old_sln, dstData.speed) :
            SerialSolveQuadratic(graph, dst, old_sln, dstData.speed);
          if (sln_temp < galois::atomicMin(dstData.solution, sln_temp)) {
            galois::atomicMin(dstData.solution, sln_temp);
            auto old_tag = dstData.tag.load(atomic_order);
            if constexpr (CONCURRENT) {
              if (old_tag != BAND) {
                while (!dstData.tag.compare_exchange_weak(
                                     old_tag, BAND, std::memory_order_relaxed));
                wl.push(ItemTy{sln_temp, dst});
              }
            } else {
              while (old_tag != BAND && !dstData.tag.compare_exchange_weak(
                                     old_tag, BAND, std::memory_order_relaxed));
              wl.push(ItemTy{sln_temp, dst}, old_sln);
            }
          } else {
            galois::gDebug(dst, " solution not updated: ", sln_temp,
              " (currently ", dstData.solution, ")");
          }
        }
      }
    }
};

if constexpr (CONCURRENT) {
  galois::GReduceMax<double> max_error;
  auto Indexer = [&](const T& item) {
    unsigned t = std::round(item.first * RF);
    // galois::gDebug(item.first, "\t", t, "\n");
    return t;
  };
  using PSchunk = galois::worklists::PerSocketChunkLIFO<32>; // chunk size 16
  using OBIM    = galois::worklists::OrderedByIntegerMetric<decltype(Indexer), PSchunk>;

galois::runtime::profileVtune(
[&](){
  galois::for_each(
    galois::iterate(wl.begin(), wl.end()),
    SerialOp,
    galois::no_conflicts(),
    // galois::no_stats(),
    galois::wl<OBIM>(Indexer),
    galois::loopname("FMM")
  );
},
"FMM_VTune");
} else {
  while (!wl.empty()) {

    SerialOp(wl.pop(), wl);

    num_iterations++;
  }
  galois::gDebug("max error: ", max_error);
  galois::gPrint("#iterarions: ", num_iterations, "\n");
}
}

template <typename Graph, typename WL, typename Accum>
void BipartFastMarchingImpl(Graph& graph, WL& oldBag, WL& newBag, Accum& counter) {

  galois::GReduceMax<double> max_error;
  auto Indexer = [&](const double& sln) {
    unsigned t = std::round(sln * 2);
    return t;
  };
  using PSchunk = galois::worklists::PerSocketChunkLIFO<32>; // chunk size 16
  using OBIM    = galois::worklists::OrderedByIntegerMetric<decltype(Indexer), PSchunk>;

galois::runtime::profileVtune(
[&](){
  galois::for_each(
    galois::iterate(oldBag.begin(), oldBag.end()),
    [&](GNode node, auto& ctx) {
      // char n; std::cin>>n;
      assert(node < NUM_CELLS && "Ghost Point!");
      auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      // galois::gPrint("Hi! I'm ", node, "\n");
#ifndef NDEBUG
      {
        auto [x, y, z] = getCoord(node);
        galois::gDebug("Hi! I'm ", node, " (", x, " ", y, " ", z, " ) with ", curData.solution);
      }
      if (curData.tag == KNOWN) {
        galois::gDebug(node, " in bag as KNWON");
        assert(curData.solution != INF);
      }

      if (curData.is_ghost) {
        // should not get here, asserted before
        std::abort();
      }

      // {
      //   auto [x, y, z] = getCoord(node);
      //   galois::atomicMin(max_error, curData.solution - std::sqrt(x * x + y * y + z * z));
      //   if (curData.solution - std::sqrt(x * x + y * y + z * z) > 0.2) {
      //     galois::gDebug(curData.solution - std::sqrt(x * x + y * y + z * z),
      //       " - wrong distance, should be ", std::sqrt(x * x + y * y + z * z));
      //     assert(false);
      //   }
      // }
#endif
      assert(curData.solution != INF);
      assert(curData.tag == BAND);
      curData.tag.store(KNOWN, atomic_order);

      // UpdateNeighbors
      for (auto e : graph.edges(node, no_lockable_flag)) {
        GNode dst = graph.getEdgeDst(e);
        if (dst < NUM_CELLS) {
          auto& dstData = graph.getData(dst, no_lockable_flag);
          assert(!dstData.is_ghost);
          // if (dstData.tag != KNOWN) {
          if (dstData.solution > curData.solution) {
#ifndef NDEBUG
            {
              auto [x, y, z] = getCoord(dst);
              galois::gDebug("Update ", dst, " (", x, " ", y, " ", z, " )");
            }
#endif
            // assert(dstData.solution == INF && dstData.tag == FAR);
          galois::gDebug(dstData.tag, dstData.solution);
            SlnTy old_sln = dstData.solution;
            double sln_temp = solveQuadratic(graph, dst, old_sln, dstData.speed);
            if (sln_temp < galois::atomicMin(dstData.solution, sln_temp)) {
              if (dstData.tag != BAND) {
                dstData.tag = BAND;
                newBag.push(dst);
                counter += 1;
              }
            } else {
              galois::gDebug(dst, " solution not updated: ", sln_temp,
                " (currently ", dstData.solution, ")");
            }
          }
        }
      }
    },
    galois::no_pushes(),
    galois::no_conflicts(),
    // galois::no_stats(),
    galois::wl<OBIM>(Indexer),
    galois::loopname("BipartFMM")
  );
},
"FMM_VTune");
}

template <typename Graph, typename WL>
void BipartFastMarching(Graph& graph, WL& oddBag, WL& evenBag) {
  galois::GAccumulator<std::size_t> more_work;

  do {
    galois::gDebug("more work ", more_work.reduce());
    more_work.reset();
    WL new_odd, new_even;
    BipartFastMarchingImpl(graph, oddBag, new_odd, more_work);
    oddBag.clear();
    std::swap(oddBag, new_odd);
    BipartFastMarchingImpl(graph, evenBag, new_even, more_work);
    evenBag.clear();
    std::swap(evenBag, new_even);
  } while (more_work.reduce());
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Algo

template <typename WL,
          typename Graph, typename BL>
void serialAlgo(Graph& graph, BL& boundary) {
  //  using HeapElemTy = std::pair<GNode, double>
  //  auto heapCmp = [&](HeapElemTy a, HeapElemTy b) { return a.second < b.second; }
  //  using HeapTy = galois::MinHeap<std::pair<GNode, double>>
  WL wl;

  FirstIteration<false>(graph, boundary, wl);

  if (wl.empty()) {
    galois::gDebug("No cell to be processed!");
    std::abort();
  } else {
#ifndef NDEBUG
  galois::gDebug("vvvvvvvv init band vvvvvvvv");
  for (auto i : wl) {
    auto coords = getCoord(i.second);
    galois::gDebug(i.second, " (", coords[0], " ", coords[1], " ", coords[2], ") with ", i.first);
  }
  galois::gDebug("^^^^^^^^ init band ^^^^^^^^");
#endif
  }

#ifndef NDEBUG
// initBag sanity check
  galois::do_all(
    galois::iterate(wl),
    [&](auto pair) {
      galois::gDebug(pair.first, " : ", pair.second);
      GNode node = pair.second;
      auto& curData = graph.getData(node, no_lockable_flag);
      // if (curData.tag != BAND) {
      //   GALOIS_DIE("Problem with initBag");
      // }
      if (curData.tag != BAND) {
        galois::gDebug("non-BAND");
      }
    },
    //galois::no_stats(),
    galois::loopname("sanity_check_initBag")
  );
#endif // end of initBag sanity check;

  FastMarching<false>(graph, wl);
}

template<typename Graph, typename WL>
void partitionAlgo(Graph& graph, WL& boundary) {}

template<typename Graph, typename WL>
void bipartAlgo(Graph& graph, WL& boundary) {
  //  using HeapElemTy = std::pair<GNode, double>
  //  auto heapCmp = [&](HeapElemTy a, HeapElemTy b) { return a.second < b.second; }
  //  using HeapTy = galois::MinHeap<std::pair<GNode, double>>
  WL oddBag, evenBag;

  BipartInitializeBag(graph, boundary, oddBag, evenBag);

  if (oddBag.empty() && evenBag.empty()) {
    galois::gDebug("No cell to be processed!");
    std::abort();
  } else {
#ifndef NDEBUG
  galois::gDebug("vvvvvvvv init band vvvvvvvv");
  for (auto i : oddBag) {
    auto coords = getCoord(i);
    galois::gDebug(i, " (", coords[0], " ", coords[1], " ", coords[2], ") with ",
      graph.getData(i, no_lockable_flag).solution);
  }
  galois::gDebug("=======");
  for (auto i : evenBag) {
    auto coords = getCoord(i);
    galois::gDebug(i, " (", coords[0], " ", coords[1], " ", coords[2], ") with ",
      graph.getData(i, no_lockable_flag).solution);
  }
  galois::gDebug("^^^^^^^^ init band ^^^^^^^^");
#endif
  }
  BipartFastMarching(graph, oddBag, evenBag);
}

template <typename WL,
          typename Graph, typename BL>
void parallelAlgo(Graph& graph, BL& boundary) {
  WL initBag;

  FirstIteration<true>(graph, boundary, initBag);

  if (initBag.empty()) {
    galois::gDebug("No cell to be processed!");
    std::abort();
  } else {
#ifndef NDEBUG
  galois::gDebug("vvvvvvvv init band vvvvvvvv");
  for (auto i : initBag) {
    auto coords = getCoord(i);
    galois::gDebug(i, " (", coords[0], " ", coords[1], " ", coords[2], ") with ",
      graph.getData(i, no_lockable_flag).solution);
  }
  galois::gDebug("^^^^^^^^ init band ^^^^^^^^");
#endif
  }

#ifndef NDEBUG
// initBag sanity check
  galois::do_all(
    galois::iterate(initBag),
    [&](auto pair) {
      auto [_, node] = pair;
      auto& curData = graph.getData(node, no_lockable_flag);
      // if (curData.tag != BAND) {
      //   GALOIS_DIE("Problem with initBag");
      // }
      if (curData.tag != BAND) {
        galois::gDebug("non-BAND");
      }
    },
    galois::no_stats(),
    galois::loopname("sanity_check_initBag")
  );
#endif // end of initBag sanity check;

  FastMarching<true>(graph, initBag);
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Sanity check

void SanityCheck(Graph& graph) {
  galois::GReduceMax<double> max_error;

  galois::do_all(
    galois::iterate(0ul, NUM_CELLS),
    [&](GNode node) noexcept {
      if (node >= NUM_CELLS)
        return;
      auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      if (curData.solution == INF) {
        galois::gPrint("Untouched cell: ", node, "\n");
        assert(curData.solution != INF);
      }

      SlnTy val = 0.;
      std::array<double, 3> dims { dx, dy, dz }; // TODO not exactly x y z order
      auto dir = graph.edge_begin(node, no_lockable_flag);
      for (double& d : dims) {
        if (dir == graph.edge_end(node, no_lockable_flag))
          break;
        GNode neighbor = graph.getEdgeDst(dir);
        auto& first_data = graph.getData(neighbor, no_lockable_flag);
        // assert(first_data.is_ghost || first_data.tag == KNOWN);
        std::advance(dir, 1); // opposite direction of the same dimension
        assert(dir != graph.edge_end(node, no_lockable_flag));
        neighbor = graph.getEdgeDst(dir);
        auto& second_data = graph.getData(neighbor, no_lockable_flag);
        // assert(second_data.is_ghost || second_data.tag == KNOWN);
        SlnTy
          s1 = (curData.solution - first_data.solution) / d,
          s2 = (curData.solution - second_data.solution) / d;
        val += std::pow(std::max(0., std::max(s1, s2)), 2);
        std::advance(dir, 1);
      }
      auto tolerance = 2.e-8;
      SlnTy error = std::sqrt(val) * curData.speed - 1.;
      max_error.update(error);
      if (error > tolerance) {
        auto [x, y, z] = getCoord(node);
        galois::gPrint("Upwind structure violated at cell: ", node,
                        " (", x, " ", y, " ", z, ")",
                        " with ", curData.solution.load(std::memory_order_relaxed), " of error ", error,
                        " (", std::sqrt(x * x + y * y + z * z), ")\n");
        return;
      }
    },
    galois::no_stats(),
    galois::loopname("sanityCheck"));

  galois::gPrint("max err: ", max_error.reduce(), "\n");
}

void SanityCheck2(Graph& graph) {
  galois::do_all(
    galois::iterate(0ul, NUM_CELLS),
    [&](GNode node) noexcept {
      auto [x, y, z] = getCoord(node);
      auto &solution = graph.getData(node).solution;
      assert(std::abs(solution - std::sqrt(x * x + y * y + z * z)));
    },
    galois::no_stats(),
    galois::loopname("sanityCheck2"));
}

void _debug_print() {
  for (GNode i = 0; i < NUM_CELLS; i++) {
    auto [x, y, z] = getCoord(i);
    galois::gDebug(x, " ", y, " ", z, " ", getNodeID({x, y, z}));
  }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url);

  galois::gDebug(ALGO_NAMES[algo]);

  // configure global variables
  if (nh) nx = ny = nz = nh;
  NUM_CELLS = nh? nh * nh * nh : nx * ny * nz;
  dx = (xb - xa) / CoordTy(nx + 1);
  dy = (yb - ya) / CoordTy(ny + 1);
  dz = (zb - za) / CoordTy(nz + 1);
  if (!RF) RF = 1 / std::min({dx, dy, dz}, std::less<CoordTy>{});
  galois::gDebug(nx, " - ", ny, " - ", nz);
  galois::gDebug(dx, " - ", dy, " - ", dz);
  galois::gDebug("RF: ", RF);

  // generate grids
  Graph graph;
  auto [num_nodes, num_cells, num_outer_faces, xy_low, xy_high, yz_low, yz_high,
        xz_low, xz_high] = generate_grid(graph, nx, ny, nz);

  // _debug_print();

  // initialize all cells
  initCells(graph, num_cells);

  // TODO better way for boundary settings?
  BL boundary;
  if (source_type == scatter)
    AssignBoundary(boundary);
  else
    AssignBoundary(graph, boundary);
  assert(!boundary.empty() && "Boundary not defined!");

#ifndef NDEBUG
  // print boundary
  galois::gDebug("vvvvvvvv boundary vvvvvvvv");
  for (GNode b : boundary) {
    auto [x, y, z] = getCoord(b);
    galois::gDebug(b, " (", x, ", ", y, ", ", z, ")");
  }
  galois::gDebug("^^^^^^^^ boundary ^^^^^^^^");
#endif

  initBoundary(graph, boundary);

  galois::StatTimer Tmain;
  Tmain.start();

  switch (algo) {
  case serial:
    serialAlgo<HeapTy>(graph, boundary);
    break;
  case parallel:
    parallelAlgo<WL>(graph, boundary);
    break;
  case partition:
    partitionAlgo(graph, boundary);
    break;
  case bipartiteSync:
    bipartAlgo(graph, boundary);
    break;
  default:
    std::abort();
  }

  Tmain.stop();

  SanityCheck(graph);
  // SanityCheck2(graph);

  return 0;
}

