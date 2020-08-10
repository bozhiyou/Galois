/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2020, The University of Texas at Austin. All rights reserved.
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
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <utility>

// Silence erroneous warnings from within Boost headers
// that show up with gcc 8.1.
// #pragma GCC diagnostic ignored "-Wparentheses"
// This warning triggers with the assert(("explanation", check));
// syntax since the left hand argument has no side-effects.
// I prefer using the comma operator over && though because
// the parentheses are more readable, so I'm silencing
// the warning for this file.
// #pragma GCC diagnostic ignored "-Wunused-value"

#include <galois/Galois.h>
#include <galois/graphs/FileGraph.h>
#include <galois/graphs/Graph.h>
#include <galois/graphs/LCGraph.h>

#include <galois/AtomicHelpers.h>

// Vendored from an old version of LLVM for Lonestar app command line handling.
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#ifdef GALOIS_ENABLE_VTUNE
#include "galois/runtime/Profile.h"
#endif

static char const* name = "Fast Marching Method";
static char const* desc =
    "Eikonal equation solver "
    "(https://en.wikipedia.org/wiki/Fast_marching_method)";
static char const* url = "";

#define DIM_LIMIT 2 // 2-D specific
using data_t         = double;
constexpr data_t INF = std::numeric_limits<data_t>::max();

enum Algo { serial = 0, parallel };
enum SourceType { scatter = 0, analytical };

const char* const ALGO_NAMES[] = {"serial", "parallel"};

static llvm::cl::OptionCategory catAlgo("1. Algorithmic Options");
static llvm::cl::opt<Algo>
    algo("algo", llvm::cl::value_desc("algo"),
         llvm::cl::desc("Choose an algorithm (default parallel):"),
         llvm::cl::values(clEnumVal(serial, "serial heap implementation"),
                          clEnumVal(parallel, "parallel implementation")),
         llvm::cl::init(parallel), llvm::cl::cat(catAlgo));
static llvm::cl::opt<unsigned> RF{"rf",
                                  llvm::cl::desc("round-off factor for OBIM"),
                                  llvm::cl::init(0u), llvm::cl::cat(catAlgo)};
static llvm::cl::opt<double> tolerance("e", llvm::cl::desc("Final error bound"),
                                       llvm::cl::init(2.e-6),
                                       llvm::cl::cat(catAlgo));

static llvm::cl::OptionCategory catInput("2. Input Options");
static llvm::cl::opt<SourceType> source_type(
    "sourceFormat", llvm::cl::desc("Choose an source format:"),
    llvm::cl::values(clEnumVal(scatter, "a set of discretized points"),
                     clEnumVal(analytical, "boundary in a analytical form")),
    llvm::cl::init(scatter), llvm::cl::cat(catInput));
static llvm::cl::opt<std::string> input_segy(
    "segy", llvm::cl::value_desc("path-to-file"),
    llvm::cl::desc("Use SEG-Y (rev 1) file as input speed map. NOTE: This will "
                   "determine the size on each dimensions"),
    llvm::cl::init(""), llvm::cl::cat(catInput));
static llvm::cl::opt<std::string> input_npy(
    "inpy", llvm::cl::value_desc("path-to-file"),
    llvm::cl::desc(
        "Use npy file (dtype=float32) as input speed map. NOTE: This will "
        "determine the size on each dimensions"),
    llvm::cl::init(""), llvm::cl::cat(catInput));
static llvm::cl::opt<std::string> input_csv(
    "icsv", llvm::cl::value_desc("path-to-file"),
    llvm::cl::desc(
        "Use csv file as input speed map. NOTE: Current implementation "
        "requires explicit definition of the size on each dimensions (see -d)"),
    llvm::cl::init(""), llvm::cl::cat(catInput));
// TODO parameterize the following
static data_t xa = -.5, xb = .5;
static data_t ya = -.5, yb = .5;

static llvm::cl::OptionCategory catOutput("3. Output Options");
static llvm::cl::opt<std::string>
    output_csv("ocsv", llvm::cl::desc("Export results to a csv format file"),
               llvm::cl::init(""), llvm::cl::cat(catOutput));
static llvm::cl::opt<std::string>
    output_npy("onpy", llvm::cl::desc("Export results to a npy format file"),
               llvm::cl::init(""), llvm::cl::cat(catOutput));

static llvm::cl::OptionCategory catDisc("4. Discretization options");
namespace internal {
template <typename T>
struct StrConv;
template <>
struct StrConv<std::size_t> {
  auto operator()(const char* str, char** endptr, int base) {
    return strtoul(str, endptr, base);
  }
};
template <>
struct StrConv<double> {
  auto operator()(const char* str, char** endptr, int) {
    return strtod(str, endptr);
  }
};
} // namespace internal
template <typename NumTy, int MAX_SIZE = 0>
struct NumVecParser : public llvm::cl::parser<std::vector<NumTy>> {
  template <typename... Args>
  NumVecParser(Args&... args) : llvm::cl::parser<std::vector<NumTy>>(args...) {}
  internal::StrConv<NumTy> strconv;
  // parse - Return true on error.
  bool parse(llvm::cl::Option& O, llvm::StringRef ArgName,
             const std::string& ArgValue, std::vector<NumTy>& Val) {
    const char* beg = ArgValue.c_str();
    char* end;

    do {
      NumTy d = strconv(Val.empty() ? beg : end + 1, &end, 0);
      if (!d)
        return O.error("Invalid option value '" + ArgName + "=" + ArgValue +
                       "': should be comma-separated unsigned integers");
      Val.push_back(d);
    } while (*end == ',');
    if (*end != '\0')
      return O.error("Invalid option value '" + ArgName + "=" + ArgValue +
                     "': should be comma-separated unsigned integers");
    if (MAX_SIZE && Val.size() > MAX_SIZE)
      return O.error(ArgName + "=" + ArgValue + ": expect no more than " +
                     std::to_string(MAX_SIZE) + " numbers but get " +
                     std::to_string(Val.size()));
    return false;
  }
};
static llvm::cl::opt<std::vector<std::size_t>, false,
                     NumVecParser<std::size_t, DIM_LIMIT>>
    dims("d", llvm::cl::value_desc("d1,d2"),
         llvm::cl::desc("Size of each dimensions as a comma-separated array "
                        "(support up to 2-D)"),
         llvm::cl::cat(catDisc));
static llvm::cl::opt<std::vector<double>, false,
                     NumVecParser<double, DIM_LIMIT>>
    intervals(
        "dx", llvm::cl::value_desc("dx,dy"),
        llvm::cl::desc("Interval of each dimensions as a comma-separated array "
                       "(support up to 2-D)"),
        llvm::cl::init(std::vector<double>{1., 1.}), llvm::cl::cat(catDisc));

static uint64_t nx, ny;
static std::size_t NUM_CELLS;
static data_t dx, dy;

///////////////////////////////////////////////////////////////////////////////

constexpr galois::MethodFlag no_lockable_flag = galois::MethodFlag::UNPROTECTED;

static_assert(sizeof(std::atomic<std::size_t>) <= sizeof(double),
              "Current buffer allocation code assumes atomic "
              "counters smaller than sizeof(double).");
static_assert(std::is_trivial_v<std::atomic<std::size_t>> &&
                  std::is_standard_layout_v<std::atomic<std::size_t>>,
              "Current buffer allocation code assumes no special "
              "construction/deletion code is needed for atomic counters.");
///////////////////////////////

struct NodeData {
  double speed; // read only
  std::atomic<double> solution;
};

///////////////////////////////////////////////////////////////////////////////

#include "fastmarchingmethod.h"
#include "structured/grids.h"
#include "structured/utils.h"

using Graph =
    galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<true>::type;
using GNode         = Graph::GraphNode;
using BL            = galois::InsertBag<GNode>;
using UpdateRequest = std::pair<data_t, GNode>;
using HeapTy        = FMMHeapWrapper<
    std::multimap<UpdateRequest::first_type, UpdateRequest::second_type>>;
using WL = galois::InsertBag<UpdateRequest>;

#include "util/input.hh"
#include "util/output.hh"

struct PushWrap {
  template <typename C, typename T = typename C::value_type>
  void operator()(C& cont, T&& item) {
    cont.push(std::forward<T>(item));
  }

  template <typename C, typename T = typename C::value_type,
            typename K = typename C::key_type>
  void operator()(C& cont, T&& item, K& hint) {
    cont.push(std::forward<T>(item), hint);
  }
};

class Solver {
  Graph& graph;
  BL boundary;
  PushWrap pushWrap;
  galois::substrate::PerThreadStorage<
      std::array<std::pair<data_t, data_t>, DIM_LIMIT>>
      local_pairs;

  void AssignBoundary() {
    if (source_type == scatter) {
      std::size_t n = xy2id({0., 0.});
      boundary.push(n);
    } else {
      galois::do_all(
          galois::iterate(0ul, NUM_CELLS),
          [&](GNode node) noexcept {
            if (node > NUM_CELLS)
              return;

            if (NonNegativeRegion(id2xy(node))) {
              for (auto e : graph.edges(node, no_lockable_flag)) {
                GNode dst = graph.getEdgeDst(e);
                if (!NonNegativeRegion(id2xy(dst))) {
                  // #ifndef NDEBUG
                  //             auto c = id2xyz(node);
                  //             galois::gDebug(node, " (", c[0], " ", c[1], "
                  //             ", c[2], ")");
                  // #endif
                  boundary.push(node);
                  break;
                }
              }
            }
          },
          galois::loopname("assignBoundary"));
    }
  }

  /////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////////

  void initBoundary() {
    galois::do_all(
        galois::iterate(boundary.begin(), boundary.end()),
        [&](const BL::value_type& node) noexcept {
          auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
          curData.solution.store(BoundaryCondition(id2xy(node)),
                                 std::memory_order_relaxed);
        },
        galois::no_stats(), galois::loopname("initializeBoundary"));
  }

  ////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////////
  // Solver

  template <typename It = typename Graph::edge_iterator>
  bool checkDirection(data_t& up_sln, It dir) {
    bool upwind = false;

    auto FindUpwind = [&](It ei) {
      GNode neighbor = graph.getEdgeDst(ei);
      auto& n_data   = graph.getData(neighbor, no_lockable_flag);
      if (data_t n_sln = n_data.solution.load(std::memory_order_relaxed);
          n_sln < up_sln) {
        up_sln = n_sln;
        upwind = true;
      }
    };

    // check one neighbor
    FindUpwind(dir++);
    // check the other neighbor
    FindUpwind(dir);
    return upwind;
  }

  template <bool CONCURRENT, typename NodeData, typename It>
  double solveQuadratic(NodeData& my_data, It edge_begin, It edge_end) {
    data_t sln         = my_data.solution.load(std::memory_order_relaxed);
    const data_t speed = my_data.speed;

    auto& sln_delta = *(local_pairs.getLocal());
    sln_delta       = {std::make_pair(sln, dx), std::make_pair(sln, dy)};

    int non_zero_counter = 0;
    auto dir             = edge_begin;
    for (auto& [s, d] : sln_delta) {
      if (dir >= edge_end)
        GALOIS_DIE("Dimension exceeded");

      if (checkDirection(s, dir)) {
        non_zero_counter++;
      }
      std::advance(dir, 2);
    }
    // local computation, nothing about graph
    if (non_zero_counter == 0)
      return sln;
    galois::gDebug("solveQuadratic: #upwind_dirs: ", non_zero_counter);

    std::sort(sln_delta.begin(), sln_delta.end(),
              [&](std::pair<data_t, data_t>& a, std::pair<data_t, data_t>& b) {
                return a.first < b.first;
              });
    auto p = sln_delta.begin();
    data_t a(0.), b_(0.), c_(0.), f(1. / (speed * speed));
    do {
      const auto& [s, d] = *p++;
      galois::gDebug(s, " ", d);
      // Arrival time may be updated in previous rounds
      // in which case remaining upwind dirs become invalid
      if (sln < s)
        break;

      double temp = 1. / (d * d);
      a += temp;
      temp *= s;
      b_ += temp;
      temp *= s;
      c_ += temp;
      if (CONCURRENT || non_zero_counter == 1) {
        data_t b = -2. * b_, c = c_ - f;
        galois::gDebug("tabc: ", temp, " ", a, " ", b, " ", c);

        double del = b * b - (4. * a * c);
        galois::gDebug(a, " ", b, " ", c, " del=", del);
        if (del >= 0) {
          double new_sln = (-b + std::sqrt(del)) / (2. * a);
          galois::gDebug("new solution: ", new_sln);
          if constexpr (CONCURRENT) { // actually also apply to serial
            if (new_sln > s) {        // conform causality
              sln = std::min(sln, new_sln);
            }
          } else {
            return new_sln;
          }
        }
      }
    } while (--non_zero_counter);
    return sln;
  }

  ////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////////
  // Operator

  ////////////////////////////////////////////////////////////////////////////////
  // FMM

  template <bool CONCURRENT, typename Graph, typename WL,
            typename GNode = typename Graph::GraphNode,
            typename T     = typename WL::value_type>
  void FastMarching(Graph& graph, WL& wl) {
    galois::GAccumulator<std::size_t> emptyWork;
    galois::GAccumulator<std::size_t> nonEmptyWork;
    emptyWork.reset();
    nonEmptyWork.reset();

    auto PushOp = [&](const auto& item, auto& wl) {
      using ItemTy = std::remove_cv_t<std::remove_reference_t<decltype(item)>>;
      // typename ItemTy::second_type node;
      // std::tie(std::ignore, node) = item;
      auto [old_sln, node] = item;
      assert(node < NUM_CELLS && "Ghost Point!");
#ifndef NDEBUG
      {
        auto [i, j] = id2ij(node);
        galois::gDebug("Pop ", old_sln, " - ", node, " (", i, " ", j, ")");
      }
#endif
      auto& curData      = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      const data_t s_sln = curData.solution.load(std::memory_order_relaxed);

      // Ignore stale repetitive work items
      if (s_sln < old_sln) {
        return;
      }

      // UpdateNeighbors
      for (auto e : graph.edges(node, no_lockable_flag)) {
        GNode dst = graph.getEdgeDst(e);

        if (dst >= NUM_CELLS)
          continue;
        auto& dstData = graph.getData(dst, no_lockable_flag);
#ifndef NDEBUG
        {
          auto [i, j] = id2ij(dst);
          galois::gDebug("Processing ", dst, " (", i, " ", j, ") ",
                         dstData.solution.load(std::memory_order_relaxed));
        }
#endif

        // Given that the arrival time propagation is non-descending
        if (dstData.solution.load(std::memory_order_relaxed) <= s_sln) {
          continue;
        }

        data_t sln_temp = solveQuadratic<CONCURRENT>(
            dstData, graph.edge_begin(dst, no_lockable_flag),
            graph.edge_end(dst, no_lockable_flag));
#ifndef NDEBUG
        {
          auto [i, j] = id2ij(dst);
          galois::gDebug("New val ", dst, " (", i, " ", j, ") ", sln_temp);
        }
#endif
        if (data_t old_sln = galois::atomicMin(dstData.solution, sln_temp);
            sln_temp < old_sln) {
          // nonEmptyWork += 1;
          if constexpr (CONCURRENT) {
            wl.push(ItemTy{sln_temp, dst});
          } else {
#ifndef NDEBUG
            {
              galois::gDebug("-- update heap");
              bool in_wl = false;
              for (auto [_, n] : wl) {
                auto [i, j] = id2ij(n);
                galois::gDebug(_, " - ", n, " (", i, " ", j, ")");
                if (n == dst) {
                  in_wl = true;
                  break;
                }
              }
              if (in_wl)
                galois::gDebug("repetitive item!");
              galois::gDebug("-- updated heap");
            }
#endif
            wl.push(ItemTy{sln_temp, dst}, old_sln);
          }
        } else {
          // emptyWork += 1;
        }
      }
    };

    if constexpr (CONCURRENT) {
      auto Indexer = [&](const T& item) {
        unsigned t = std::round(item.first * RF);
        // galois::gDebug(item.first, "\t", t, "\n");
        return t;
      };
      using PSchunk =
          galois::worklists::PerSocketChunkLIFO<32>; // chunk size 16
      using OBIM =
          galois::worklists::OrderedByIntegerMetric<decltype(Indexer), PSchunk>;
#ifdef GALOIS_ENABLE_VTUNE
      galois::runtime::profileVtune(
          [&]() {
#endif
            galois::for_each(galois::iterate(wl.begin(), wl.end()), PushOp,
                             galois::disable_conflict_detection(),
                             // galois::no_stats(),  // stat iterations
                             galois::wl<OBIM>(Indexer),
                             galois::loopname("FMM"));
#ifdef GALOIS_ENABLE_VTUNE
          },
          "FMM_VTune");
#endif
    } else {
      std::size_t num_iterations = 0;

      while (!wl.empty()) {

        PushOp(wl.pop(), wl);

        num_iterations++;

#ifndef NDEBUG
        sleep(1); // Debug pause
        galois::gDebug("\n********\n");
#endif
      }

      galois::gPrint("#iterarions: ", num_iterations, "\n");
    }

    // galois::runtime::reportParam("Statistics", "EmptyWork",
    // emptyWork.reduce()); galois::runtime::reportParam("Statistics",
    // "NonEmptyWork",
    //                              nonEmptyWork.reduce());
  }

  ////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////////
  // Algo

  template <bool CONCURRENT, typename WL>
  void runAlgo() {
    WL initBag;

    using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                           galois::StdForEach>::type;
    Loop loop;

    loop(
        galois::iterate(boundary.begin(), boundary.end()),
        [&](GNode node) noexcept {
          auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
          pushWrap(initBag,
                   {curData.solution.load(std::memory_order_relaxed), node});
        },
        galois::loopname("FirstIteration"));

    if (initBag.empty()) {
      GALOIS_DIE("No cell to be processed!");
#ifndef NDEBUG
    } else {
      galois::gDebug("vvvvvvvv init band vvvvvvvv");
      for (auto [k, i] : initBag) {
        auto [x, y] = id2xy(i);
        galois::gDebug(k, " - ", i, " (", x, " ", y, "): arrival_time=",
                       graph.getData(i, no_lockable_flag)
                           .solution.load(std::memory_order_relaxed));
      }
      galois::gDebug("^^^^^^^^ init band ^^^^^^^^");
#endif // end of initBag sanity check;
    }

    FastMarching<CONCURRENT>(graph, initBag);
  }

  ////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////////
  // Sanity check

  void SanityCheck() {
    galois::GReduceMax<double> max_error;

    galois::do_all(
        galois::iterate(graph.begin(), graph.end()),
        [&](GNode node) noexcept {
          if (node >= NUM_CELLS)
            return;

          auto& my_data = graph.getData(node, no_lockable_flag);
          if (my_data.solution.load(std::memory_order_relaxed) == INF) {
            galois::gPrint("Untouched cell: ", node, "\n");
          }
          data_t new_val = solveQuadratic<false>(
              my_data, graph.edge_begin(node, no_lockable_flag),
              graph.edge_end(node, no_lockable_flag));
          if (data_t old_val = my_data.solution.load(std::memory_order_relaxed);
              new_val != old_val) {
            data_t error = std::abs(new_val - old_val) / std::abs(old_val);
            max_error.update(error);
            if (error > tolerance) {
              auto [x, y] = id2xy(node);
              galois::gPrint("Error bound violated at cell ", node, " (", x,
                             " ", y, "): old_val=", old_val,
                             " new_val=", new_val, " error=", error, "\n");
            }
          }
        },
        galois::no_stats(), galois::loopname("sanityCheck"));

    galois::gPrint("MAX ERROR: ", max_error.reduce(), "\n");
  }

  void SanityCheck2() {
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

          data_t val = 0.;
          std::array<double, 2> dims{dx, dy}; // TODO not exactly x y z order
          auto dir = graph.edge_begin(node, no_lockable_flag);
          for (double& d : dims) {
            if (dir == graph.edge_end(node, no_lockable_flag))
              break;
            GNode neighbor   = graph.getEdgeDst(dir);
            auto& first_data = graph.getData(neighbor, no_lockable_flag);
            std::advance(dir, 1); // opposite direction of the same dimension
            assert(dir != graph.edge_end(node, no_lockable_flag));
            neighbor          = graph.getEdgeDst(dir);
            auto& second_data = graph.getData(neighbor, no_lockable_flag);
            data_t s1         = (curData.solution - first_data.solution) / d,
                   s2         = (curData.solution - second_data.solution) / d;
            val += std::pow(std::max(0., std::max(s1, s2)), 2);
            std::advance(dir, 1);
          }
          data_t error =
              (std::sqrt(val) - (1. / curData.speed)) / (1. / curData.speed);
          max_error.update(error);
          if (error > tolerance) {
            auto [x, y] = id2xy(node);
            galois::gPrint(
                "Error bound violated at cell: ", node, " (", x, " ", y, ")",
                " with ", curData.solution.load(std::memory_order_relaxed),
                " of error ", error, " (", std::sqrt(x * x + y * y), ")\n");
            return;
          }
        },
        galois::no_stats(), galois::loopname("sanityCheck2"));

    galois::gPrint("MAX ERROR: ", max_error.reduce(), "\n");
  }

public:
  explicit Solver(Graph& g) : graph(g) {
    //! Boundary assignment and initialization
    // TODO better way for boundary settings?
    AssignBoundary();
    if (boundary.empty())
      GALOIS_DIE("Boundary not found!");

#ifndef NDEBUG
    // print boundary
    galois::gDebug("vvvvvvvv boundary vvvvvvvv");
    for (GNode b : boundary) {
      auto [x, y] = id2xy(b);
      galois::gDebug(b, " (", x, ", ", y, ")");
    }
    galois::gDebug("^^^^^^^^ boundary ^^^^^^^^");
#endif

    initBoundary();

    //! run algo
    galois::StatTimer Tmain;
    Tmain.start();

    switch (algo) {
    case serial:
      runAlgo<false, HeapTy>();
      break;
    case parallel:
      runAlgo<true, WL>();
      break;
    default:
      std::abort();
    }

    Tmain.stop();

    SanityCheck();

    if (!output_npy.empty())
      DumpToNpy(graph);
    else if (!output_csv.empty())
      DumpToCsv(graph);
  }
};

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url, nullptr);

  galois::gDebug(ALGO_NAMES[algo]);

  Graph graph;

  //! Grids generation and cell initialization
  // generate grids
  SetupGrids(graph);

  Solver solver(graph);

  return 0;
}
