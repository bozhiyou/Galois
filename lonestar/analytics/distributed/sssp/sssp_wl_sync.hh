/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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

// #include "galois/runtime/SyncStructures.h"
/**
 * Creates a Galois reduction sync structure that does a min reduction.
 */
#ifdef GALOIS_ENABLE_GPU
// TODO GPU code included
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_MIN_WL(fieldname, fieldtype)              \
  struct Reduce_min_##fieldname {                                              \
    typedef fieldtype ValTy;                                                   \
                                                                               \
    static ValTy extract(uint32_t, const struct NodeData& node) {              \
      return node.fieldname;                                                   \
    }                                                                          \
                                                                               \
    static bool extract_batch(unsigned, uint8_t*, size_t*, DataCommMode*) {    \
      return false;                                                            \
    }                                                                          \
                                                                               \
    static bool extract_batch(unsigned, uint8_t*) { return false; }            \
                                                                               \
    static bool extract_reset_batch(unsigned, uint8_t*, size_t*,               \
                                    DataCommMode*) {                           \
      return false;                                                            \
    }                                                                          \
                                                                               \
    static bool extract_reset_batch(unsigned, uint8_t*) { return false; }      \
                                                                               \
    static bool reset_batch(size_t, size_t) { return true; }                   \
                                                                               \
    static bool reduce(uint32_t lid, struct NodeData& node, ValTy y) {         \
      if (y < galois::min(node.fieldname, y)) {                                \
        initBag->push(UpdateRequest{lid, y});                                  \
        return true;                                                           \
      } else {                                                                 \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
                                                                               \
    static bool reduce_batch(unsigned, uint8_t*, DataCommMode) {               \
      return false;                                                            \
    }                                                                          \
                                                                               \
    static bool reduce_mirror_batch(unsigned, uint8_t*, DataCommMode) {        \
      return false;                                                            \
    }                                                                          \
                                                                               \
    static void reset(uint32_t, struct NodeData&) {}                           \
                                                                               \
    static void setVal(uint32_t, struct NodeData& node, ValTy y) {             \
      node.fieldname = y;                                                      \
    }                                                                          \
                                                                               \
    static bool setVal_batch(unsigned, uint8_t*, DataCommMode) {               \
      return false;                                                            \
    }                                                                          \
  }
#endif
GALOIS_SYNC_STRUCTURE_REDUCE_MIN_WL(dist_current, unsigned int);
GALOIS_SYNC_STRUCTURE_BITSET(dist_current);
