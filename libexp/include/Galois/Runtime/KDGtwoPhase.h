/** KDG two phase executor -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_KDGTWOPHASE_H
#define GALOIS_RUNTIME_KDGTWOPHASE_H

#include "Galois/GaloisForwardDecl.h"
#include "Galois/Accumulator.h"
#include "Galois/Atomic.h"
#include "Galois/BoundedVector.h"
#include "Galois/gdeque.h"
#include "Galois/PriorityQueue.h"
#include "Galois/Timer.h"
#include "Galois/DoAllWrap.h"
#include "Galois/PerThreadContainer.h"
#include "Galois/optional.h"

#include "Galois/Substrate/Barrier.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Executor_DoAll.h"
#include "Galois/Runtime/Executor_ParaMeter.h"
#include "Galois/Runtime/ForEachTraits.h"
#include "Galois/Runtime/Range.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Substrate/Termination.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Runtime/IKDGbase.h"
#include "Galois/Runtime/WindowWorkList.h"
#include "Galois/Runtime/UserContextAccess.h"
#include "Galois/Substrate/gio.h"
#include "Galois/Runtime/ThreadRWlock.h"
#include "Galois/Substrate/CompilerSpecific.h"
#include "Galois/Runtime/Mem.h"

#include <boost/iterator/transform_iterator.hpp>

#include <iostream>
#include <memory>


namespace Galois {
namespace Runtime {


namespace {

template <typename T, typename Cmp>
class TwoPhaseContext: public OrderedContextBase<T> {

public:

  using Base = OrderedContextBase<T>;
  // using NhoodList =  Galois::gdeque<Lockable*, 4>;
  using CtxtCmp = ContextComparator<TwoPhaseContext, Cmp>;

  CtxtCmp ctxtCmp;
  bool source = true;


  using value_type = T;

  explicit TwoPhaseContext (const T& x, const Cmp& cmp)
    : 
      Base (x),  // pass true so that Base::acquire invokes virtual subAcquire
      ctxtCmp (cmp),
      source (true) 
  {}

  bool isSrc (void) const {
    return source;
  }

  void disableSrc (void) {
    source = false;
  }

  void reset () { 
    source = true;
  }

  virtual void subAcquire (Lockable* l, Galois::MethodFlag) {


    if (Base::tryLock (l)) {
      Base::addToNhood (l);
    }

    TwoPhaseContext* other = nullptr;

    do {
      other = static_cast<TwoPhaseContext*> (Base::getOwner (l));

      if (other == this) {
        return;
      }

      if (other) {
        bool conflict = ctxtCmp (other, this); // *other < *this
        if (conflict) {
          // A lock that I want but can't get
          this->source = false;
          return; 
        }
      }
    } while (!this->stealByCAS(l, other));

    // Disable loser
    if (other) {
      other->source = false; // Only need atomic write
    }

    return;


    // bool succ = false;
    // if (Base::tryAcquire (l) == Base::NEW_OWNER) {
      // Base::addToNhood (l);
      // succ = true;
    // }
// 
    // assert (Base::getOwner (l) != NULL);
// 
    // if (!succ) {
      // while (true) {
        // TwoPhaseContext* that = static_cast<TwoPhaseContext*> (Base::getOwner (l));
// 
        // assert (that != NULL);
        // assert (this != that);
// 
        // if (PtrComparator::compare (this, that)) { // this < that
          // if (Base::stealByCAS (that, this)) {
            // that->source = false;
            // break;
          // }
// 
        // } else { // this >= that
          // this->source = false; 
          // break;
        // }
      // }
    // } // end outer if
  } // end subAcquire



};

template <typename T, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename ArgsTuple>
class IKDGtwoPhaseExecutor: public IKDGbase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, TwoPhaseContext<T, Cmp> > {

public:
  using Ctxt = TwoPhaseContext<T, Cmp>;
  using Base = IKDGbase <T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, Ctxt>;

  using WindowWL = typename std::conditional<Base::NEEDS_PUSH, PQwindowWL<T, Cmp>, SortedRangeWindowWL<T, Cmp> >::type;
  using CtxtWL = typename Base::CtxtWL;



protected:

  static const bool DETAILED_STATS = false;

  struct CtxtMaker {
    IKDGtwoPhaseExecutor& outer;

    Ctxt* operator () (const T& x) {

      Ctxt* ctxt = outer.ctxtAlloc.allocate (1);
      assert (ctxt);
      outer.ctxtAlloc.construct (ctxt, x, outer.cmp);

      return ctxt;
    }
  };

  struct WindowWLwrapper: public WindowWL {
    IKDGtwoPhaseExecutor& outer;

    WindowWLwrapper (IKDGtwoPhaseExecutor& outer, const Cmp& cmp):
      WindowWL (cmp), outer (outer) {}

    void push (const T& x) {
      WindowWL::push (x);
    }

    // TODO: complete this class
    void push (Ctxt* c) {
      assert (c);

      WindowWL::push (c->getActive ());

      // destroy and deallocate c
      outer.ctxtAlloc.destroy (c);
      outer.ctxtAlloc.deallocate (c, 1);
    }

    void poll (CtxtWL& wl, size_t newSize, size_t origSize) {
      WindowWL::poll (wl, newSize, origSize, outer.ctxtMaker);
    }
  };

  WindowWLwrapper winWLwrapper;
  CtxtMaker ctxtMaker;


public:
  IKDGtwoPhaseExecutor (
      const Cmp& cmp, 
      const NhFunc& nhFunc,
      const ExFunc& exFunc,
      const OpFunc& opFunc,
      const ArgsTuple& argsTuple)
    :
      Base (cmp, nhFunc, exFunc, opFunc, argsTuple),
      winWLwrapper (*this, cmp),
      ctxtMaker {*this}
  {
  }

  ~IKDGtwoPhaseExecutor () {

    dumpStats ();

    if (Base::ENABLE_PARAMETER) {
      ParaMeter::closeStatsFile ();
    }
  }

  void dumpStats (void) {
    reportStat (Base::loopname, "efficiency %", double (100.0 * Base::totalCommits) / Base::totalTasks);
    reportStat (Base::loopname, "avg. parallelism", double (Base::totalCommits) / Base::rounds);
  }

  template <typename R>
  void push_initial (const R& range) {
    if (Base::targetCommitRatio == 0.0) {

      Galois::do_all_choice (range,
          [this] (const T& x) {
            Base::getNextWL ().push_back (ctxtMaker (x));
          }, 
          std::make_tuple (
            Galois::loopname ("init-fill"),
            chunk_size<NhFunc::CHUNK_SIZE> ()));


    } else {
      winWLwrapper.initfill (range);
          
    }
  }

  void execute () {
    execute_impl ();
  }

protected:

  GALOIS_ATTRIBUTE_PROF_NOINLINE void endRound () {
    Base::endRound ();

    if (Base::ENABLE_PARAMETER) {
      ParaMeter::StepStats s (Base::rounds, Base::roundCommits.reduceRO (), Base::roundTasks.reduceRO ());
      s.dump (ParaMeter::getStatsFile (), Base::loopname);
    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhoodImpl (HIDDEN::DummyExecFunc*) {
    // for stable case

    Galois::do_all_choice (makeLocalRange (Base::getCurrWL ()),
        [this] (Ctxt* c) {
          typename Base::UserCtxt& uhand = *Base::userHandles.getLocal ();
          uhand.reset ();

          // nhFunc (c, uhand);
          runCatching (Base::nhFunc, c, uhand);

          Base::roundTasks += 1;
        },
        std::make_tuple (
          Galois::loopname ("expandNhood"),
          chunk_size<NhFunc::CHUNK_SIZE> ()));
  }

  struct GetActive: public std::unary_function<Ctxt*, const T&> {
    const T& operator () (const Ctxt* c) const {
      assert (c != nullptr);
      return c->getActive ();
    }
  };

  template <typename F>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhoodImpl (F*) {
    // for unstable case
    auto m_beg = boost::make_transform_iterator (Base::getCurrWL ().begin_all (), GetActive ());
    auto m_end = boost::make_transform_iterator (Base::getCurrWL ().end_all (), GetActive ());

    Galois::do_all_choice (makeLocalRange (Base::getCurrWL ()),
        [m_beg, m_end, this] (Ctxt* c) {
          typename Base::UserCtxt& uhand = *Base::userHandles.getLocal ();
          uhand.reset ();

          runCatching (Base::nhFunc, c, uhand, m_beg, m_end);

          Base::roundTasks += 1;
        },
        std::make_tuple (
          Galois::loopname ("expandNhoodUnstable"),
          chunk_size<NhFunc::CHUNK_SIZE> ()));
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhood () {
    // using ptr to exFunc to choose the right impl. 
    // relying on the fact that for stable case, the exFunc is DummyExecFunc. 
    expandNhoodImpl (&this->exFunc); 
  }

  inline void executeSourcesImpl (HIDDEN::DummyExecFunc*) {
  }

  template <typename F>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void executeSourcesImpl (F*) {
    assert (Base::HAS_EXEC_FUNC);

    Galois::do_all_choice (makeLocalRange (Base::getCurrWL ()),
      [this] (Ctxt* ctxt) {

        typename Base::UserCtxt& uhand = *Base::userHandles.getLocal ();
        uhand.reset ();

        if (ctxt->isSrc ()) {
          this->exFunc (ctxt->getActive (), uhand);
        }
      },
      std::make_tuple (
        Galois::loopname ("exec-sources"),
        Galois::chunk_size<ExFunc::CHUNK_SIZE> ()));

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void executeSources (void) {
    // using ptr to exFunc to choose the right impl. 
    // relying on the fact that for stable case, the exFunc is DummyExecFunc. 
    executeSourcesImpl (&this->exFunc);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void applyOperator () {
    Galois::optional<T> minElem;

    if (Base::NEEDS_PUSH) {
      if (Base::targetCommitRatio != 0.0 && !winWLwrapper.empty ()) {
        minElem = *winWLwrapper.getMin();
      }
    }


    Galois::do_all_choice (makeLocalRange (Base::getCurrWL ()),
        [this, &minElem] (Ctxt* c) {
          bool commit = false;

          typename Base::UserCtxt& uhand = *Base::userHandles.getLocal ();
          uhand.reset ();

          if (c->isSrc ()) {
            // opFunc (c->active, uhand);
            if (Base::OPERATOR_CAN_ABORT) {
              runCatching (Base::opFunc, c, uhand);
              commit = c->isSrc (); // in case opFunc signalled abort

            } else {
              Base::opFunc (c->getActive (), uhand);
              assert (c->isSrc ());
              commit = true;
            }
          } else {
            commit = false;
          }

          if (commit) {
            Base::roundCommits += 1;
            if (Base::NEEDS_PUSH) { 
              for (auto i = uhand.getPushBuffer ().begin ()
                  , endi = uhand.getPushBuffer ().end (); i != endi; ++i) {

                if ((Base::targetCommitRatio == 0.0) || !minElem || !Base::cmp (*minElem, *i)) {
                  // if *i >= *minElem
                  Base::getNextWL ().push_back (ctxtMaker (*i));
                } else {
                  winWLwrapper.push (*i);
                } 
              }
            } else {
              assert (uhand.getPushBuffer ().begin () == uhand.getPushBuffer ().end ());
            }

            c->commitIteration ();
            c->~Ctxt ();
            Base::ctxtAlloc.deallocate (c, 1);
          } else {
            c->cancelIteration ();
            c->reset ();
            Base::getNextWL ().push_back (c);
          }
        },
        std::make_tuple (
          Galois::loopname ("applyOperator"),
          chunk_size<OpFunc::CHUNK_SIZE> ()));
  }


  void execute_impl () {

    while (true) {
      Base::beginRound (winWLwrapper);

      if (Base::getCurrWL ().empty_all ()) {
        break;
      }

      Timer t;

      if (DETAILED_STATS) {
        std::printf ("trying to execute %zd elements\n", Base::getCurrWL ().size_all ());
        t.start ();
      }

      expandNhood ();

      executeSources ();

      applyOperator ();

      endRound ();

      if (DETAILED_STATS) {
        t.stop ();
        std::printf ("Time taken: %ld\n", t.get ());
      }
      
    }
  }
  
};


} // end anonymous namespace

template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename ExFunc, typename _ArgsTuple>
void for_each_ordered_ikdg (const R& range, const Cmp& cmp, const NhFunc& nhFunc, 
    const ExFunc& exFunc,  const OpFunc& opFunc, const _ArgsTuple& argsTuple) {

  auto argsT = std::tuple_cat (argsTuple, 
      get_default_trait_values (argsTuple,
        std::make_tuple (loopname_tag {}, enable_parameter_tag {}),
        std::make_tuple (default_loopname {}, enable_parameter<false> {})));
  using ArgsT = decltype (argsT);

  using T = typename R::value_type;
  

  using Exec = IKDGtwoPhaseExecutor<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsT>;
  
  Exec e (cmp, nhFunc, exFunc, opFunc, argsT);

  const bool wakeupThreadPool = true;

  if (wakeupThreadPool) {
    Substrate::getThreadPool().burnPower(Galois::getActiveThreads ());
  }

  e.push_initial (range);
  e.execute ();

  if (wakeupThreadPool) {
    Substrate::getThreadPool().beKind ();
  }

}


template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename ArgsTuple>
void for_each_ordered_ikdg (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const ArgsTuple& argsTuple) {

  for_each_ordered_ikdg (range, cmp, nhFunc, HIDDEN::DummyExecFunc (), opFunc, argsTuple);
}

} // end namespace Runtime
} // end namespace Galois

#endif //  GALOIS_RUNTIME_KDG_TWO_PHASE_H
