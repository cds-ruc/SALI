#ifndef __SALI_H__
#define __SALI_H__

#include <random>
#include "concurrency.h"
#include "sali_base.h"
#include "omp.h"
#include "tbb/combinable.h"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/spin_mutex.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <math.h>
#include <sstream>
#include <stack>
#include <stdint.h>
#include <thread>
#include <vector>
#include <unordered_set>
#include <future>
#include <unistd.h>
#include "piecewise_linear_model.h"


#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace sali {

#define READ_EVOLVE 0
#define COMPRESS 0
static thread_local int skip_counter = 0;
static std::atomic_int64_t allocated = 0;
static std::atomic_int64_t num_compressed_node = 0;
static std::atomic_int64_t freed_size = 0;
static int max_ratio = 0;

// runtime assert
#define RT_ASSERT(expr)                                                        \
  {                                                                            \
    if (!(expr)) {                                                             \
      fprintf(stderr, "Thread %d: RT_ASSERT Error at %s:%d, `%s` not hold!\n", \
              omp_get_thread_num(), __FILE__, __LINE__, #expr);                \
      exit(0);                                                                 \
    }                                                                          \
  }

typedef void (*dealloc_func)(void *ptr);

// runtime debug
#define PRINT_DEBUG 0

#if PRINT_DEBUG
#define RESET "\033[0m"
#define RED "\033[31m"     /* Red */
#define GREEN "\033[32m"   /* Green */
#define YELLOW "\033[33m"  /* Yellow */
#define BLUE "\033[34m"    /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m"    /* Cyan */
#define WHITE "\033[37m"   /* White */

#define RT_DEBUG(msg, ...)                                                     \
  if (omp_get_thread_num() == 0) {                                             \
    printf(GREEN "T%d: " msg RESET "\n", omp_get_thread_num(), __VA_ARGS__);   \
  } else if (omp_get_thread_num() == 1) {                                      \
    printf(YELLOW "\t\t\tT%d: " msg RESET "\n", omp_get_thread_num(),          \
           __VA_ARGS__);                                                       \
  } else {                                                                     \
    printf(BLUE "\t\t\t\t\t\tT%d: " msg RESET "\n", omp_get_thread_num(),      \
           __VA_ARGS__);                                                       \
  }
#else
#define RT_DEBUG(msg, ...)
#endif

#define COLLECT_TIME 0

#if COLLECT_TIME
#include <chrono>
#endif

template<class T, class P, bool USE_FMCD = true>
class SALI {
    static_assert(std::is_arithmetic<T>::value,
    "SALI key type must be numeric.");

    inline double compute_gap_count(int size) {
      if (size >= 1000000)
        return 1;
      if (size >= 100000)
        return 2;
      return 5;
    }

    inline double compute_max_ratio(int size) {
      if (size >= 1500000)
        return 2;
      if (size >= 1000000)
        return 1;
      if (size >= 100000)
        return 0;
      if (size >= 10000)
        return 0;
      if (size >= 1000)
        return 0;
      if (size >= 100)
        return 0;
      return 0;
    }

    struct Node;

    inline int PREDICT_POS(Node *node, T key) const {
      double v = node->model.predict_double(key);
      if (v > std::numeric_limits<int>::max() / 2) {
        return node->num_items - 1;
      }
      if (v < 0) {
        return 0;
      }
      return std::min(node->num_items - 1, static_cast<int>(v));
      //return node->model.predict_pos(key);
    }


    const double BUILD_LR_REMAIN;
    const bool QUIET;

    struct {
        long long fmcd_success_times = 0;
        long long fmcd_broken_times = 0;
#if COLLECT_TIME
        double time_scan_and_destory_tree = 0;
        double time_build_tree_bulk = 0;
#endif
    } stats;

public:
    // Epoch based Memory Reclaim
    class ThreadSpecificEpochBasedReclamationInformation {

        std::array<std::vector<Node *>, 3> mFreeLists;
        std::atomic <uint32_t> mLocalEpoch;
        uint32_t mPreviouslyAccessedEpoch;
        bool mThreadWantsToAdvance;
        SALI<T, P> *tree;

    public:
        ThreadSpecificEpochBasedReclamationInformation(SALI<T, P> *index)
            : mFreeLists(), mLocalEpoch(3), mPreviouslyAccessedEpoch(3),
              mThreadWantsToAdvance(false), tree(index) {}

        ThreadSpecificEpochBasedReclamationInformation(
            ThreadSpecificEpochBasedReclamationInformation const &other) = delete;

        ThreadSpecificEpochBasedReclamationInformation(
            ThreadSpecificEpochBasedReclamationInformation &&other) = delete;

        ~ThreadSpecificEpochBasedReclamationInformation() {
          for (uint32_t i = 0; i < 3; ++i) {
            freeForEpoch(i);
          }
        }

        void scheduleForDeletion(Node *pointer) {
          assert(mLocalEpoch != 3);
          std::vector < Node * > &currentFreeList =
              mFreeLists[mLocalEpoch];
          currentFreeList.emplace_back(pointer);
          mThreadWantsToAdvance = (currentFreeList.size() % 64u) == 0;
        }

        void scheduleForDeletionX(Node *pointer) {
          assert(mLocalEpoch != 3);
          std::vector < Node * > &currentFreeList =
              mFreeLists[mLocalEpoch];
          currentFreeList.emplace_back(pointer);
          mThreadWantsToAdvance = (currentFreeList.size() % 64u) == 0;
        }

        uint32_t getLocalEpoch() const {
          return mLocalEpoch.load(std::memory_order_acquire);
        }

        void enter(uint32_t newEpoch) {
          assert(mLocalEpoch == 3);
          if (mPreviouslyAccessedEpoch != newEpoch) {
            freeForEpoch(newEpoch);
            mThreadWantsToAdvance = false;
            mPreviouslyAccessedEpoch = newEpoch;
          }
          mLocalEpoch.store(newEpoch, std::memory_order_release);
        }

        void leave() { mLocalEpoch.store(3, std::memory_order_release); }

        bool doesThreadWantToAdvanceEpoch() { return (mThreadWantsToAdvance); }

    private:
        void freeForEpoch(uint32_t epoch) {
          std::vector < Node * > &previousFreeList =
              mFreeLists[epoch];
          for (Node *node: previousFreeList) {
            if (node->is_two) {
              RT_ASSERT(node->build_size == 2);
              RT_ASSERT(node->num_items == 8);
              node->size = 2;
              //node->num_inserts = node->num_insert_to_data = 0;
              for (int i = 0; i < node->num_items; i++) node->items[i].typeVersionLockObsolete.store(0b100);
              for (int i = 0; i < node->num_items; i++) node->items[i].entry_type = 0;
              tree->pending_two[omp_get_thread_num()].push(node);
            } else {
              tree->delete_items(node->items, node->num_items);
              tree->delete_nodes(node, 1);
            }
          }
          previousFreeList.resize(0u);
        }
    };

    class EpochBasedMemoryReclamationStrategy {
    public:
        uint32_t NEXT_EPOCH[3] = {1, 2, 0};
        uint32_t PREVIOUS_EPOCH[3] = {2, 0, 1};

        SALI<T, P> *tree;
        std::array<std::vector<Node *>, 3> gFreeLists;
        std::atomic <uint32_t> mCurrentEpoch;
        tbb::enumerable_thread_specific <
        ThreadSpecificEpochBasedReclamationInformation,
        tbb::cache_aligned_allocator<
            ThreadSpecificEpochBasedReclamationInformation>,
        tbb::ets_key_per_instance>
            mThreadSpecificInformations;

        EpochBasedMemoryReclamationStrategy(SALI<T, P> *index)
            : mCurrentEpoch(0), tree(index), mThreadSpecificInformations(index) {}
        ~EpochBasedMemoryReclamationStrategy(){
          for (uint32_t i = 0; i < 3; ++i) {
            freeForEpochG(i);
          }
        }

    public:
        static EpochBasedMemoryReclamationStrategy *getInstance(SALI<T, P> *index) {

          return index->ebr;
        }

        void enterCriticalSection() {
          ThreadSpecificEpochBasedReclamationInformation &currentMemoryInformation =
              mThreadSpecificInformations.local();
          uint32_t currentEpoch = mCurrentEpoch.load(std::memory_order_acquire);
          currentMemoryInformation.enter(currentEpoch);
          if (currentMemoryInformation.doesThreadWantToAdvanceEpoch() &&
              canAdvance(currentEpoch)) {
            bool ret = mCurrentEpoch.compare_exchange_strong(currentEpoch,
                                                             NEXT_EPOCH[currentEpoch]);
            if (ret) {
              freeForEpochG(NEXT_EPOCH[mCurrentEpoch.load(std::memory_order_acquire)]);
            }
          }
        }

        void addG(Node *pointer) {
          uint32_t currentEpoch = mCurrentEpoch.load(std::memory_order_acquire);
          std::vector < Node * > &currentFreeList = gFreeLists[mCurrentEpoch];
          currentFreeList.emplace_back(pointer);
        }

        void freeForEpochG(uint32_t epoch) {
          std::vector < Node * > &previousFreeList =
              gFreeLists[epoch];
          for (Node *node: previousFreeList) {
            //std::cout<<"freeForEpochG"<<node<<std::endl;
            if (node->is_two) {
              RT_ASSERT(node->build_size == 2);
              RT_ASSERT(node->num_items == 8);
              node->size = 2;
              //node->num_inserts = node->num_insert_to_data = 0;
              for (int i = 0; i < node->num_items; i++) node->items[i].typeVersionLockObsolete.store(0b100);
              for (int i = 0; i < node->num_items; i++) node->items[i].entry_type = 0;
              tree->pending_two[omp_get_thread_num()].push(node);
            } else {
              freed_size += node->num_items * sizeof(Item);
              freed_size += sizeof(Node);
              tree->delete_items(node->items, node->num_items);
              tree->delete_nodes(node, 1);
            }
          }
          previousFreeList.resize(0u);
        }

        bool canAdvance(uint32_t currentEpoch) {
          uint32_t previousEpoch = PREVIOUS_EPOCH[currentEpoch];
          return !std::any_of(
              mThreadSpecificInformations.begin(),
              mThreadSpecificInformations.end(),
              [previousEpoch](ThreadSpecificEpochBasedReclamationInformation const
                              &threadInformation) {
                return (threadInformation.getLocalEpoch() == previousEpoch);
              });
        }

        void leaveCriticialSection() {
          ThreadSpecificEpochBasedReclamationInformation &currentMemoryInformation =
              mThreadSpecificInformations.local();
          currentMemoryInformation.leave();
        }

        void scheduleForDeletion(Node *pointer) {
          mThreadSpecificInformations.local().scheduleForDeletion(pointer);
        }
    };

    class EpochGuard {
        EpochBasedMemoryReclamationStrategy *instance;

    public:
        EpochGuard(SALI<T, P> *index) {
          instance = EpochBasedMemoryReclamationStrategy::getInstance(index);
          instance->enterCriticalSection();
        }

        ~EpochGuard() { instance->leaveCriticialSection(); }
    };

    EpochBasedMemoryReclamationStrategy *ebr;

    typedef std::pair <T, P> V;

    SALI(double BUILD_LR_REMAIN = 0, bool QUIET = false, long long memory_budget = 0)
        : BUILD_LR_REMAIN(BUILD_LR_REMAIN), QUIET(QUIET), memory_budget_(memory_budget) {
      {
        std::cout << "SALI" << std::endl;
        std::cout << sizeof(Node) << std::endl;
        std::cout << sizeof(Item) << std::endl;
        allocated = 0;
        std::vector < Node * > nodes;
        for (int _ = 0; _ < 1e2; _++) {
          Node *node = build_tree_two(T(0), P(), T(1), P(), 0, 0, 1);
          nodes.push_back(node);
        }
        std::cout << std::to_string(allocated.load()) << std::endl;
        for (auto node: nodes) {
          destroy_tree(node);
        }
        std::cout << std::to_string(allocated.load()) << std::endl;
        std::cout << pending_two[0].size() << std::endl;

        if (!QUIET) {
          printf("initial memory pool size = %lu\n",
                 pending_two[omp_get_thread_num()].size());
        }
      }
      if (USE_FMCD && !QUIET) {
        printf("enable FMCD\n");
      }

      root = build_tree_none();
      int temp = 100000000;

      ebr = initEbrInstance(this);

      dummy_head_.next_ = &dummy_tail_;
      dummy_tail_.prev_ = &dummy_head_;
      pool_size_ = 0;

#if COMPRESS
      if (memory_budget_ == 0) {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        memory_budget_ = pages * page_size;
      }
      std::cout << "memory_budget: " << memory_budget_ << std::endl;
      th = std::thread(&sali::SALI<T, P>::threadFunction, this);
#endif
    }

    ~SALI() {
      std::cout << "SALI Destruct." << std::endl;

#if COMPRESS
      exitSignal = 1;
      compress_cv_.notify_one();
      th.join();
#endif
      delete ebr;
      std::cout << "pool " << pool_size_ << std::endl;
      std::cout << "num_compressed_node " << num_compressed_node << std::endl;
      std::cout << "freed_size " << freed_size << std::endl;
      std::cout << "total_size " << total_size() << std::endl;
      destroy_tree(root);
      root = NULL;
      destory_pending();
      ListNode *cur = dummy_head_.next_;
      while (cur != &dummy_tail_) {
        ListNode *temp = cur->next_;
        delete cur;
        cur = temp;
      }
    }

    void insert(const V &v) { insert(v.first, v.second); }

    void show() {
      for (int i = 0; i < root->num_items; i++) {
        if (root->items[i].entry_type != 1)
          std::cout << "root: " << root->items[i].comp.data.key << std::endl;
        else {
          for (int j = 0; j < root->items[i].comp.child->num_items; j++) {
            if (root->items[i].comp.child->items[j].entry_type == 2)
              std::cout << "child: " << root->items[i].comp.child->items[j].comp.data.key
                        << root->items[i].comp.child->items[j].comp.data.value << std::endl;
          }
        }
      }
    }

    void insert(const T &key, const P &value) {
      EpochGuard guard(this);
      // root = insert_tree(root, key, value);
      bool state = insert_tree(key, value);
      RT_DEBUG("Insert_tree(%d): success/fail? %d", key, state);
    }

    int get_first_key_from_node(Node *node, T *key) {
      //std::cout<<"get_first_key_from_node"<<node<< std::endl;
      //std::cout<<"get_first_key_from_node"<<node->num_items<< std::endl;
      bool needRestart = false;
      uint64_t versionItem;
      if (needRestart) {
        return -1;
      }
      bool ret = 0;
      for (int i = 0; i < node->num_items; i++) {
        versionItem = node->items[i].readLockOrRestart(needRestart);
        if (needRestart) {
          return -1;
        }
        if (node->items[i].entry_type == 0) {
          node->items[i].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart) {
            return -1;
          }
          continue;
        } else if (node->items[i].entry_type == 1) {
          Node *child = node->items[i].comp.child;
          node->items[i].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart) {
            return -1;
          }
          ret = get_first_key_from_node(child, key);
          break;
        } else if (node->items[i].entry_type == 2) {
          *key = node->items[i].comp.data.key;
          node->items[i].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart) {
            return -1;
          }
          break;
        }
      }
      //std::cout<<"get_first_key_from_node"<<node<< std::endl;
      //std::cout<<"get_first_key_from_node  ret "<<*key<< std::endl;
      return 0;
    }

    int get_real_pos_from_compressed_node(Node *node, int tmp_pos, T key) {
      int lo = (tmp_pos) <= (compress_epsilon) ? 0 : ((tmp_pos) - (compress_epsilon));
      int hi = (tmp_pos) + (compress_epsilon) + 2 >= (node->num_items - 1) ? (node->num_items - 1) : (tmp_pos) +
                                                                                                     (compress_epsilon) +
                                                                                                     2;
      while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        bool needRestart = false;
        uint64_t versionItem = node->items[mid].readLockOrRestart(needRestart);
        if (needRestart) {
          return -1;
        }
        T mid_key;
        if (node->items[mid].entry_type == 2) {
          mid_key = node->items[mid].comp.data.key;
          node->items[mid].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart) {
            return -1;
          }
        } else if (node->items[mid].entry_type == 1) {
          Node *child = node->items[mid].comp.child;
          node->items[mid].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart) {
            return -1;
          }
          int ret = get_first_key_from_node(child, &mid_key);
          if (ret < 0) {
            return -1;
          }
        } else {
          RT_ASSERT(node->items[mid].entry_type == 1 || node->items[mid].entry_type == 2);
        }
        if (key > mid_key) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      return hi;
    }

    bool at(const T &key, P &value) {
      EpochGuard guard(this);
      int restartCount = 0;
      restart:
      if (restartCount++)
        yield(restartCount);
      bool needRestart = false;

      // for lock coupling
      uint64_t versionItem;
      Node *parent = nullptr;

      constexpr int MAX_DEPTH = 128;
      Node *path[MAX_DEPTH];
      int path_size = 0;
      //int num_fixed = 0;
      bool ret = false;
      for (Node *node = root;;) {
        path[path_size++] = node;
        int pos = PREDICT_POS(node, key);
        versionItem = node->items[pos].readLockOrRestart(needRestart);
        if (needRestart)
          goto restart;

        if (node->items[pos].entry_type == 1) { // 1 means child
          if (node->last_adjust_type == 3) {
            node->items[pos].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart)
              goto restart;
            int compress_pos = get_real_pos_from_compressed_node(node, pos, key);
            if (compress_pos < 0) {
              goto restart;
            }
            versionItem = node->items[compress_pos].readLockOrRestart(needRestart);
            if (needRestart)
              goto restart;
            if (node->items[compress_pos].entry_type == 2) {
              if (node->items[compress_pos].comp.data.key != key) {
                ret = false;
              } else {
                value = node->items[compress_pos].comp.data.value;
                ret = true;
              }
              node->items[compress_pos].readUnlockOrRestart(versionItem, needRestart);
              if (needRestart)
                goto restart;
              break;
            } else {
              pos = compress_pos;
            }
          }
          parent = node;
          node = node->items[pos].comp.child;

          parent->items[pos].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart)
            goto restart;
        } else { // the entry is a data or empty
          if (node->items[pos].entry_type == 0) { // 0 means empty
            ret = false;
            break;
          } else { // 2 means data
            if (node->items[pos].comp.data.key != key) {
              if (node->last_adjust_type == 3) {
                node->items[pos].readUnlockOrRestart(versionItem, needRestart);
                if (needRestart)
                  goto restart;
                int compress_pos = get_real_pos_from_compressed_node(node, pos, key);
                if (compress_pos < 0) {
                  goto restart;
                }
                versionItem = node->items[compress_pos].readLockOrRestart(needRestart);
                if (needRestart)
                  goto restart;
                if (node->items[compress_pos].entry_type == 1) {
                  parent = node;
                  node = node->items[compress_pos].comp.child;
                  parent->items[compress_pos].readUnlockOrRestart(versionItem, needRestart);
                  if (needRestart)
                    goto restart;
                  continue;
                } else {
                  if (node->items[compress_pos].comp.data.key != key) {
                    ret = false;
                  } else {
                    value = node->items[compress_pos].comp.data.value;
                    ret = true;
                  }
                  node->items[compress_pos].readUnlockOrRestart(versionItem, needRestart);
                  if (needRestart)
                    goto restart;
                  break;
                }
              } else {
                RT_ASSERT(node->items[pos].comp.data.key == key);
              }
            } else {
              value = node->items[pos].comp.data.value;
              ret = true;
              node->items[pos].readUnlockOrRestart(versionItem, needRestart);
              if (needRestart)
                goto restart;
              break;
            }
          }
        }
      }


#if READ_EVOLVE
      if (likely(path_size <= 3 || !pq_distribution(getGen()) || path[1]->last_adjust_type == 2)) {
        return ret;
      }

      Node *node_prob_cur = path[1];
      int prev_size = node_prob_cur->build_size;
      uint64_t prev_build_time = node_prob_cur->build_time;
      long double prev_speed = node_prob_cur->speed;
      // const int ESIZE = node->size; //race here
      // T *keys = new T[ESIZE];
      // P *values = new P[ESIZE];
      std::vector <T> *keys;   // make it be a ptr here because we will let scan_and_destroy
      // to decide the size after getting the locks
      std::vector <P> *values; // scan_and_destroy will fill up the keys/values

      if (prev_size < 64) {
        int t_size = count_tree_size(node_prob_cur);
        if (t_size < 64) {
          return ret;
        }
      }
      num_read_probability_trigger++;

      int numKeysCollected = scan_and_destory_tree(
          node_prob_cur, &keys, &values); // pass the (address) of the ptr
      if (numKeysCollected < 0) {
        return true;
      }

      uint64_t cur_time = timeSinceEpochNanosec();
      long double speed = (long double) (numKeysCollected - prev_size) / (cur_time - prev_build_time + 1);
      bulk_args args = {numKeysCollected, speed, cur_time, 2, speed / prev_speed};

      Node *new_node = build_tree_bulk_two_model(keys, values, args);


      delete keys;
      delete values;
      Node *node_prob_prev = path[0];

      int retryLockCount = 0;
      retryLock:
      if (retryLockCount++)
        yield(retryLockCount);

      int pos = PREDICT_POS(node_prob_prev, key);

      bool needRetry = false;

      node_prob_prev->items[pos].writeLockOrRestart(needRetry);
      if (needRetry) {
        RT_DEBUG("Final step of adjust, obtain parent %p lock FAIL, retry",
                 path[i - 1]);
        goto retryLock;
      }
      node_prob_prev->items[pos].comp.child = new_node;
      node_prob_prev->items[pos].writeUnlock();
#endif
      return ret;

    }

    bool exists(const T &key) const {
      // EpochGuard guard;
      Node *node = root;
      while (true) {
        int pos = PREDICT_POS(node, key);
        if (node->items[pos].entry_type == 0) {
          return false;
        } else if (node->items[pos].entry_type == 2) {
          return node->items[pos].comp.data.key == key;
        } else {
          node = node->items[pos].comp.child;
        }
      }
    }

    void yield(int count) {
      if (count > 3)
        sched_yield();
      else
        _mm_pause();
    }

    void bulk_load(const V *vs, int num_keys) {
      if (num_keys == 0) {
        destroy_tree(root);
        root = build_tree_none();
        return;
      }
      if (num_keys == 1) {
        destroy_tree(root);
        root = build_tree_none();
        insert(vs[0]);
        return;
      }
      if (num_keys == 2) {
        destroy_tree(root);
        root =
            build_tree_two(vs[0].first, vs[0].second, vs[1].first, vs[1].second, 0.001, timeSinceEpochNanosec(), 1);
        return;
      }

      RT_ASSERT(num_keys > 2);
      for (int i = 1; i < num_keys; i++) {
        RT_ASSERT(vs[i].first > vs[i - 1].first);
      }

      std::vector <T> *keys = new std::vector<T>();
      std::vector <P> *values = new std::vector<P>();
      keys->reserve(num_keys);
      values->reserve(num_keys);
      for (int i = 0; i < num_keys; i++) {
        (*keys)[i] = vs[i].first;
        (*values)[i] = vs[i].second;
      }
      destroy_tree(root);
      bulk_args args = {num_keys, 0.001, timeSinceEpochNanosec(), 1, 1};
      root = build_tree_bulk(keys, values, args);
#if COMPRESS
      for (int i = 0; i < root->num_items; i++) {
        if (root->items[i].entry_type == 1
            && root->items[i].comp.child->num_items >= 64) {
          //std::cout << "push_to_cooling_pool, new_node: " << root->items[i].comp.child << std::endl;
          push_to_cooling_pool(root, root->items[i].comp.child);
        }
      }
#endif
      delete keys;
      delete values;
    }

    bool remove(const T &key) {
      EpochGuard guard(this);
      int restartCount = 0;
      restart:
      if (restartCount++)
        yield(restartCount);
      bool needRestart = false;

      constexpr int MAX_DEPTH = 128;
      Node *path[MAX_DEPTH];
      int path_size = 0;

      // for lock coupling
      uint64_t versionItem;
      Node *parent;
      Node *node = root;

      while (true) {
        // R-lock this node

        RT_ASSERT(path_size < MAX_DEPTH);
        path[path_size++] = node;

        int pos = PREDICT_POS(node, key);
        versionItem = node->items[pos].readLockOrRestart(needRestart);
        if (needRestart)
          goto restart;
        if (node->items[pos].entry_type == 0) // 0 means empty entry
        {
          return false;
        } else if (node->items[pos].entry_type == 2) // 2 means existing entry has data already
        {
          RT_DEBUG("Existed %p pos %d, locking.", node, pos);
          node->items[pos].upgradeToWriteLockOrRestart(versionItem, needRestart);
          if (needRestart) {
            goto restart;
          }

          node->items[pos].entry_type = 0;

          node->items[pos].writeUnlock();

          /*for (int i = 0; i < path_size; i++) {
            path[i]->size--;
          }*/
          /*if (node->size == 0) {
            int parent_pos = PREDICT_POS(parent, key);
            restartCount = 0;
            deleteNodeRemove:
            bool deleteNodeRestart = false;
            if (restartCount++)
              yield(restartCount);
            parent->items[parent_pos].writeLockOrRestart(deleteNodeRestart);
            if (deleteNodeRestart) goto deleteNodeRemove;

            parent->items[parent_pos].entry_type = 0;

            parent->items[parent_pos].writeUnlock();

            safe_delete_nodes(node, 1);

          }*/
          return true;
        } else // 1 means has a child, need to go down and see
        {
          parent = node;
          node = node->items[pos].comp.child;           // now: node is the child

          parent->items[pos].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart)
            goto restart;
        }
      }

    }

    bool update(const T &key, const P &value) {
      EpochGuard guard(this);
      int restartCount = 0;
      restart:
      if (restartCount++)
        yield(restartCount);
      bool needRestart = false;

      // for lock coupling
      uint64_t versionItem;
      Node *parent;

      for (Node *node = root;;) {
        // R-lock this node

        int pos = PREDICT_POS(node, key);
        versionItem = node->items[pos].readLockOrRestart(needRestart);
        if (needRestart)
          goto restart;
        if (node->items[pos].entry_type == 0) // 0 means empty entry
        {
          return false;
        } else if (node->items[pos].entry_type == 2) // 2 means existing entry has data already
        {
          RT_DEBUG("Existed %p pos %d, locking.", node, pos);
          node->items[pos].upgradeToWriteLockOrRestart(versionItem, needRestart);
          if (needRestart) {
            goto restart;
          }

          node->items[pos].comp.data.value = value;

          node->items[pos].writeUnlock();

          break;
        } else // 1 means has a child, need to go down and see
        {
          parent = node;
          node = node->items[pos].comp.child;           // now: node is the child

          parent->items[pos].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart)
            goto restart;
        }
      }

      return true;
    }

    // Find the minimum `len` keys which are no less than `lower`, returns the number of found keys.
    int range_query_len(std::pair <T, P> *results, const T &lower, int len) {
      return range_core_len<false>(results, 0, root, lower, len);
    }

    size_t total_size() const {
      std::stack < Node * > s;
      s.push(root);

      size_t size = 0;
      while (!s.empty()) {
        Node *node = s.top();
        s.pop();
        size += sizeof(*node);
        for (int i = 0; i < node->num_items; i++) {
          size += sizeof(Item);
          if (node->items[i].entry_type == 1) {
            s.push(node->items[i].comp.child);
          }
        }
      }
      return size;
    }

    void print_depth() const {
      std::stack < Node * > s;
      std::stack<int> d;
      s.push(root);
      d.push(1);

      int max_depth = 1;
      long sum_depth = 0, sum_nodes = 0;
      while (!s.empty()) {
        Node *node = s.top();
        s.pop();
        int depth = d.top();
        d.pop();
        for (int i = 0; i < node->num_items; i++) {
          if (node->items[i].entry_type == 1) {
            s.push(node->items[i].comp.child);
            d.push(depth + 1);
          } else if (node->items[i].entry_type == 2) {
            max_depth = std::max(max_depth, depth);
            sum_depth += depth;
            sum_nodes++;
          }
        }
      }

      std::cout << "==============================================" << std::endl;
      printf("max_depth = %d, avg_depth = %.2lf\n", max_depth, double(sum_depth) / double(sum_nodes));
      std::cout << "root build_size = " << std::to_string(root->build_size) << std::endl;
      std::cout << "root num_items = " << std::to_string(root->num_items) << std::endl;
      std::cout << "num_rebuild = " << std::to_string(num_rebuild) << std::endl;
      std::cout << "num_write_probability_trigger = " << std::to_string(num_write_probability_trigger) << std::endl;
      std::cout << "num_read_probability_trigger = " << std::to_string(num_read_probability_trigger) << std::endl;
      std::cout << "num_lower_64 = " << std::to_string(num_lower_64) << std::endl;
      std::cout << "==============================================" << std::endl;
    }

    long adjust_num() {
      return num_write_probability_trigger;
    }

    std::tuple<long, double, long> depth() {
      std::stack < Node * > s;
      std::stack<int> d;
      s.push(root);
      d.push(1);

      int max_depth = 1;
      long sum_depth = 0, sum_nodes = 0;
      while (!s.empty()) {
        Node *node = s.top();
        s.pop();
        int depth = d.top();
        d.pop();
        for (int i = 0; i < node->num_items; i++) {
          if (node->items[i].entry_type == 1) {
            s.push(node->items[i].comp.child);
            d.push(depth + 1);
          } else if (node->items[i].entry_type == 2) {
            max_depth = std::max(max_depth, depth);
            sum_depth += depth;
            sum_nodes++;
          }
        }
      }
      return std::tuple<long, double, long>(max_depth, double(sum_depth) / double(sum_nodes), sum_nodes);
    }

private:
    struct Node;
    struct Item : OptLock {
        union {
            struct {
                T key;
                P value;
            } data;
            Node *child;
        } comp;
        uint8_t entry_type; //0 means empty, 1 means child, 2 means data
    };
    struct Node {
        int is_two;            // is special node for only two keys
        int build_size;        // tree size (include sub nodes) when node created
        //std::atomic<int> size; // current subtree size
        int size;
        int fixed; // fixed node will not trigger rebuild
        //std::atomic<int> num_inserts, num_insert_to_data;
        int num_items; // number of slots
        Item *items;
        //prob
        long double p_conflict;
        std::bernoulli_distribution conflict_distribution;
        //long double p_acc;
        uint64_t build_time;
        long double speed;
        int last_adjust_type = 1;
        int cooling = 0;
        TwoLinearModel<T> model;
    };


    Node *root;
    int adjustsuccess = 0;
    std::stack<Node *> pending_two[1024];


    template<class Type>
    class accounting_allocator {
    public:
        // type definitions
        typedef Type value_type;
        typedef Type *pointer;
        typedef const Type *const_pointer;
        typedef Type &reference;
        typedef const Type &const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        //static size_t allocated;

        // rebind allocator to type U
        template<class U>
        struct rebind {
            typedef accounting_allocator<U> other;
        };

        // return address of values
        pointer address(reference value) const {
          return &value;
        }

        const_pointer address(const_reference value) const {
          return &value;
        }

        /* constructors and destructor
         * - nothing to do because the allocator has no state
         */
        accounting_allocator() throw() {
        }

        accounting_allocator(const accounting_allocator &) throw() {
        }

        template<class U>
        accounting_allocator(const accounting_allocator<U> &) throw() {
        }

        ~accounting_allocator() throw() {
        }

        // return maximum number of elements that can be allocated
        size_type max_size() const throw() {
          //  std::cout << "max_size()" << std::endl;
          return std::numeric_limits<std::size_t>::max() / sizeof(Type);
        }

        // allocate but don't initialize num elements of type Type
        pointer allocate(size_type num, const void * = 0) {
          // print message and allocate memory with global new
          //std::cerr << "allocate " << num << " element(s)" << " of size " << sizeof(Type) << std::endl;
          pointer ret = (pointer)(::operator new(num * sizeof(Type)));
          //std::cerr << " allocated at: " << (void*)ret << std::endl;
          allocated += num * sizeof(Type);
          //std::cerr << "allocated: " << allocated/(1024*1024) << " MB" << endl;
          return ret;
        }

        // initialize elements of allocated storage p with value value
        void construct(pointer p, const Type &value) {
          // initialize memory with placement new
          new((void *) p)Type(value);
        }

        void construct(pointer p) {
          // initialize memory with placement new
          new((void *) p)Type();
        }

        // destroy elements of initialized storage p
        void destroy(pointer p) {
          // destroy objects by calling their destructor
          p->~Type();
        }

        // deallocate storage p of deleted elements
        void deallocate(pointer p, size_type num) {
          // print message and deallocate memory with global delete
#if 0
          std::cerr << "deallocate " << num << " element(s)"
                     << " of size " << sizeof(Type)
                     << " at: " << (void*)p << std::endl;
#endif
          ::operator delete((void *) p);
          allocated -= num * sizeof(Type);
        }
    };

    /*template<>
    class accounting_allocator<void> {
    public:
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef void *pointer;
        typedef const void *const_pointer;
        typedef void value_type;

        template<typename _Tp1>
        struct rebind {
            typedef allocator <_Tp1> other;
        };
    };


// return that all specializations of this allocator are interchangeable
    template<class T1, class T2>
    bool operator==(const accounting_allocator<T1> &,
                    const accounting_allocator<T2> &) throw() {
      return true;
    }

    template<class T1, class T2>
    bool operator!=(const accounting_allocator<T1> &,
                    const accounting_allocator<T2> &) throw() {
      return false;
    }*/

    std::allocator <Node> node_allocator;

    std::atomic<long long> num_read_probability_trigger = 0;
    std::atomic<long long> num_write_probability_trigger = 0;
    std::atomic<long long> num_lower_64 = 0;
    std::atomic<long long> num_rebuild = 0;
    uint32_t temp = 0xfffff;
    std::vector <std::vector<uint32_t>> p_array;
    std::vector <std::discrete_distribution<int>> d_array;
    std::bernoulli_distribution pq_distribution{0.00001};
    std::bernoulli_distribution rq_distribution{0.00001};

    std::atomic<int> exitSignal = 0;
    std::thread th;
    struct alignas(64) AlignedStruct {
      int field = 0;
    };
    std::array<AlignedStruct, 1024> al;

    long long memory_budget_ = 0;

    struct NodePair {
        Node *parent_{nullptr};
        Node *child_{nullptr};

        NodePair(Node *parent, Node *child) : parent_(parent), child_(child) {}

        NodePair() = default;
    };

    struct ListNode {
        NodePair val_;
        ListNode *prev_{nullptr};
        ListNode *next_{nullptr};

        ListNode(NodePair val, ListNode *prev, ListNode *next) : val_(val), prev_(prev), next_(next) {}

        explicit ListNode(NodePair val) : val_(val) {}

        ListNode() = default;
    };

    std::unordered_map<Node *, ListNode *> map_;
    ListNode dummy_head_;
    ListNode dummy_tail_;
    tbb::spin_mutex mu_;
    size_t pool_size_;
    std::bernoulli_distribution cool_select_distribution{0.1};

    void delete_from_cooling_pool(Node *ptr) {
      //std::cout << "delete_from_cooling_pool, new_node: " << ptr << std::endl;

      mu_.lock();
      auto it = map_.find(ptr);
      if (it == map_.end()) {
        mu_.unlock();
        return;
      }
      ListNode *node = it->second;
      node->prev_->next_ = node->next_;
      node->next_->prev_ = node->prev_;
      map_.erase(it);
      pool_size_--;
      node->val_.child_->cooling = 0;
      mu_.unlock();
      delete node;
    }

    void push_to_cooling_pool(Node *parent, Node *ptr) {
      mu_.lock();
      if (map_.find(ptr) != map_.end()) {
        mu_.unlock();
        return;
      }
      ListNode *node = new ListNode(NodePair(parent, ptr));
      node->next_ = &dummy_tail_;
      node->prev_ = dummy_tail_.prev_;
      dummy_tail_.prev_->next_ = node;
      dummy_tail_.prev_ = node;
      map_.insert(std::pair<Node *, ListNode *>(ptr, node));
      pool_size_++;
      ptr->cooling = 1;
      mu_.unlock();
    }

    bool victim_from_cooling_pool(NodePair *ptr) {
      mu_.lock();
      if (pool_size_ == 0) {
        mu_.unlock();
        return false;
      }
      pool_size_--;
      ListNode *node = dummy_head_.next_;
      dummy_head_.next_ = node->next_;
      dummy_head_.next_->prev_ = &dummy_head_;
      map_.erase(node->val_.child_);
      node->val_.child_->cooling = 0;
      mu_.unlock();
      *ptr = node->val_;
      delete node;
      return true;
    }

    /*uint32_t p_array[17][17] = {
        { 0 },
        { 0xffffffff },
        { 0xffffffff, 0xffffffff },
        { 0x3fff, 0x3fff, 0xffffffff },
        { 0x1fff, 0x1fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0xfff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0xfff, 0xfff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0xfff, 0x7ff, 0xfff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0xfff, 0x7ff, 0x7ff, 0xfff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0xfff, 0x3ff, 0x7ff, 0x7ff, 0xfff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0x7ff, 0x1ff, 0x3ff, 0x7ff, 0x7ff, 0xfff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0x7ff, 0xff, 0x1ff, 0x3ff, 0x7ff, 0x7ff, 0xfff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0x7ff, 0xff, 0x7f, 0x1ff, 0x3ff, 0x7ff, 0x7ff, 0xfff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0x7ff, 0xff, 0x3f, 0x7f, 0x1ff, 0x3ff, 0x7ff, 0x7ff, 0xfff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff },
        { 0x3fff, 0x1fff, 0x7ff, 0xff, 0x3f, 0, 0x7f, 0x1ff, 0x3ff, 0x7ff, 0x7ff, 0xfff, 0x1fff, 0x3fff, 0x3fff, 0xffffffff }
    };*/

    uint32_t uintRand() {
      static thread_local std::mt19937
      generator(time(0));
      return generator();
    }

    std::minstd_rand &getGen() {
      static thread_local std::minstd_rand
      generator(time(0));
      return generator;
    }

    uint64_t timeSinceEpochNanosec() {
      using namespace std::chrono;
      return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    }

    std::mutex compress_mu_;
    std::condition_variable compress_cv_;
    int compress_epsilon = 4096;

    void threadFunction() {
      std::cout << "Compress Thread Start" << std::endl;

      std::unique_lock <std::mutex> latch(compress_mu_);
      while (exitSignal == 0) {
        compress_cv_.wait(latch);
        //std::cout << "Compress Thread Start2" << std::endl;
        NodePair nodePair;
        if (exitSignal == 1) {
          while (victim_from_cooling_pool(&nodePair)) {
            //std::cout << "Doing Compress" << std::endl;
            Node *child = nodePair.child_;
            std::vector <T> *keys;
            std::vector <P> *values;
            int numKeysCollected = scan_and_destory_tree2(
                child, &keys, &values);
            if (numKeysCollected < 0) {
              continue;
            }

            int prev_size = child->build_size;
            uint64_t prev_build_time = child->build_time;
            uint64_t cur_time = timeSinceEpochNanosec();
            long double speed = (long double) (numKeysCollected - prev_size) / (cur_time - prev_build_time + 1);
            //bulk_args args={numKeysCollected, speed, cur_time, 1,speed/prev_speed};


            Node *new_node = new_nodes(1);
            OptimalPiecewiseLinearModel<T, P> opt(std::numeric_limits<int>::max());
            opt.add_point((*keys)[0], 0);
            int start = 0;
            //int segment_idx=0;
            for (int i = 1; i < numKeysCollected; ++i) {
              bool add_success = opt.add_point((*keys)[i], i - start);  // alwayse start from 0 for each segment
              RT_ASSERT(add_success);
            }
            typename OptimalPiecewiseLinearModel<T, P>::CanonicalSegment cs = opt.get_segment();
            //pgm auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(cs.get_first_x());
            auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(0);
            //new_node->model.params[segment_idx]=model_param(cs_slope,cs_intercept);
            new_node->model.type = 0;
            new_node->model.a1 = cs_slope;
            new_node->model.b1 = cs_intercept;
            //segment_idx++;
            //start = i;
            //opt.add_point((*keys)[i], i-start);

            new_node->is_two = 0;
            new_node->build_size = numKeysCollected;
            new_node->size = numKeysCollected;
            new_node->fixed = 0;
            //new_node->num_inserts = child->num_insert_to_data = 0;
            new_node->num_items = numKeysCollected;
            if (numKeysCollected > 1e6) {
              new_node->fixed = 1;
            }
            new_node->items = new_items(new_node->num_items);
            if (new_node->build_size < 64) {
              new_node->p_conflict = 1 / (0.1 * 2 * 64);
            } else {
              new_node->p_conflict = 1 / (0.1 * 2 * new_node->build_size);
            }
            new_node->conflict_distribution = std::bernoulli_distribution(new_node->p_conflict);
            new_node->build_time = cur_time;
            new_node->speed = speed;
            new_node->last_adjust_type = 3;
            for (int i = 0; i < numKeysCollected; ++i) {
              new_node->items[i].entry_type = 2;
              new_node->items[i].comp.data.key = (*keys)[i];
              new_node->items[i].comp.data.value = (*values)[i];
            }
            int retryLockCount = 0;
            retryLock1:
            if (retryLockCount++)
              yield(retryLockCount);

            int pos = PREDICT_POS(nodePair.parent_, (*keys)[0]);

            bool needRetry = false;

            nodePair.parent_->items[pos].writeLockOrRestart(needRetry);
            if (needRetry) {
              goto retryLock1;
            }
            nodePair.parent_->items[pos].comp.child = new_node;
            nodePair.parent_->items[pos].writeUnlock();
            num_compressed_node += numKeysCollected;
          }
          return;
        }

        auto ret = victim_from_cooling_pool(&nodePair);
        if (ret == false) {
          //std::cout << "Nothing Compress" << std::endl;
          continue;
        }
        if (nodePair.parent_ == nullptr) {
          std::cout << "Doing Compress Root" << std::endl;
        }
        Node *child = nodePair.child_;
        std::vector <T> *keys;
        std::vector <P> *values;
        int numKeysCollected = scan_and_destory_tree2(
            child, &keys, &values);
        if (numKeysCollected < 0) {
          continue;
        }

        int prev_size = child->build_size;
        uint64_t prev_build_time = child->build_time;
        uint64_t cur_time = timeSinceEpochNanosec();
        long double speed = (long double) (numKeysCollected - prev_size) / (cur_time - prev_build_time + 1);
        //bulk_args args={numKeysCollected, speed, cur_time, 1,speed/prev_speed};


        Node *new_node = new_nodes(1);
        OptimalPiecewiseLinearModel<T, P> opt(std::numeric_limits<int>::max());
        opt.add_point((*keys)[0], 0);
        int start = 0;
        //int segment_idx=0;
        for (int i = 1; i < numKeysCollected; ++i) {
          bool add_success = opt.add_point((*keys)[i], i - start);  // alwayse start from 0 for each segment
          RT_ASSERT(add_success);
        }
        typename OptimalPiecewiseLinearModel<T, P>::CanonicalSegment cs = opt.get_segment();
        //pgm auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(cs.get_first_x());
        auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(0);
        //new_node->model.params[segment_idx]=model_param(cs_slope,cs_intercept);
        new_node->model.type = 0;
        new_node->model.a1 = cs_slope;
        new_node->model.b1 = cs_intercept;
        //segment_idx++;
        //start = i;
        //opt.add_point((*keys)[i], i-start);

        new_node->is_two = 0;
        new_node->build_size = numKeysCollected;
        new_node->size = numKeysCollected;
        new_node->fixed = 0;
        //new_node->num_inserts = child->num_insert_to_data = 0;
        new_node->num_items = numKeysCollected;
        if (numKeysCollected > 1e6) {
          new_node->fixed = 1;
        }
        new_node->items = new_items(new_node->num_items);
        if (new_node->build_size < 64) {
          new_node->p_conflict = 1 / (0.1 * 2 * 64);
        } else {
          new_node->p_conflict = 1 / (0.1 * 2 * new_node->build_size);
        }
        new_node->conflict_distribution = std::bernoulli_distribution(new_node->p_conflict);
        new_node->build_time = cur_time;
        new_node->speed = speed;
        new_node->last_adjust_type = 3;
        for (int i = 0; i < numKeysCollected; ++i) {
          new_node->items[i].entry_type = 2;
          new_node->items[i].comp.data.key = (*keys)[i];
          new_node->items[i].comp.data.value = (*values)[i];
        }
        int retryLockCount = 0;
        retryLock:
        if (retryLockCount++)
          yield(retryLockCount);

        if (nodePair.parent_ == nullptr) {
          root = new_node;
          std::cout << "Doing Compress Root" << std::endl;
          continue;
        }
        int pos = PREDICT_POS(nodePair.parent_, (*keys)[0]);

        bool needRetry = false;

        nodePair.parent_->items[pos].writeLockOrRestart(needRetry);
        if (needRetry) {
          goto retryLock;
        }
        nodePair.parent_->items[pos].comp.child = new_node;
        nodePair.parent_->items[pos].writeUnlock();
        num_compressed_node += numKeysCollected;
        //std::cout << "numKeysCollected: " << numKeysCollected << std::endl;
        //std::cout << "numKeysCollected: " << new_node << std::endl;
        //std::cout << "Compress Thread Done" << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
      std::cout << "Compress Thread End" << std::endl;
    }

    void compress() {
      std::cout << "Begin Compress" << std::endl;
      NodePair nodePair;
      auto ret = victim_from_cooling_pool(&nodePair);
      if (ret == false) {
        return;
      }
      Node *child = nodePair.child_;
      std::vector <T> *keys;
      std::vector <P> *values;
      int numKeysCollected = scan_and_destory_tree(
          child, &keys, &values);
      if (numKeysCollected < 0) {
        return;
      }
      int prev_size = child->build_size;
      uint64_t prev_build_time = child->build_time;
      uint64_t cur_time = timeSinceEpochNanosec();
      long double speed = (long double) (numKeysCollected - prev_size) / (cur_time - prev_build_time + 1);
      //bulk_args args={numKeysCollected, speed, cur_time, 1,speed/prev_speed};


      Node *new_node = new_nodes(1);
      OptimalPiecewiseLinearModel<T, P> opt(compress_epsilon);
      opt.add_point((*keys)[0], 0);
      int start = 0;
      //int segment_idx=0;
      for (int i = 1; i < numKeysCollected; ++i) {
        bool add_success = opt.add_point((*keys)[i], i - start);  // alwayse start from 0 for each segment
        RT_ASSERT(add_success);
      }
      typename OptimalPiecewiseLinearModel<T, P>::CanonicalSegment cs = opt.get_segment();
      //pgm auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(cs.get_first_x());
      auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(0);
      //new_node->model.params[segment_idx]=model_param(cs_slope,cs_intercept);
      new_node->model.a = cs_slope;
      new_node->model.b = cs_intercept;
      //segment_idx++;
      //start = i;
      //opt.add_point((*keys)[i], i-start);

      new_node->is_two = 0;
      new_node->build_size = numKeysCollected;
      new_node->size = numKeysCollected;
      new_node->fixed = 0;
      //new_node->num_inserts = child->num_insert_to_data = 0;
      new_node->num_items = numKeysCollected;
      if (numKeysCollected > 1e6) {
        new_node->fixed = 1;
      }
      new_node->items = new_items(new_node->num_items);
      if (new_node->build_size < 64) {
        new_node->p_conflict = 1 / (0.1 * 2 * 64);
      } else {
        new_node->p_conflict = 1 / (0.1 * 2 * new_node->build_size);
      }
      new_node->conflict_distribution = std::bernoulli_distribution(new_node->p_conflict);
      new_node->build_time = cur_time;
      new_node->speed = speed;
      new_node->last_adjust_type = 3;
      for (int i = 1; i < numKeysCollected; ++i) {
        new_node->items[i].entry_type = 2;
        new_node->items[i].comp.data.key = (*keys)[i];
        new_node->items[i].comp.data.value = (*values)[i];
      }
      int retryLockCount = 0;
      retryLock:
      if (retryLockCount++)
        yield(retryLockCount);

      int pos = PREDICT_POS(nodePair.parent_, (*keys)[0]);

      bool needRetry = false;

      nodePair.parent_->items[pos].writeLockOrRestart(needRetry);
      if (needRetry) {
        goto retryLock;
      }
      nodePair.parent_->items[pos].comp.child = new_node;
      nodePair.parent_->items[pos].writeUnlock();
      std::cout << "End Compress" << std::endl;

    }

    EpochBasedMemoryReclamationStrategy *initEbrInstance(SALI<T, P> *index) {
      ebr = new EpochBasedMemoryReclamationStrategy(index);
      return ebr;
    }

    Node *new_nodes(int n) {
      Node *p = node_allocator.allocate(n);
      al[omp_get_thread_num()].field += n * sizeof(Node);
      //node_allocator.construct(p);
      RT_ASSERT(p != NULL && p != (Node *) (-1));
      return p;
    }

    void delete_nodes(Node *p, int n) {
      node_allocator.deallocate(p, n);
      al[omp_get_thread_num()].field -= n * sizeof(Node);
    }

    void safe_delete_nodes(Node *p, int n) {
      for (int i = 0; i < n; ++i) {
        ebr->scheduleForDeletion(p + i);
      }
    }

    std::allocator <Item> item_allocator;

    Item *new_items(int n) {
      Item *p = item_allocator.allocate(n);
      al[omp_get_thread_num()].field += n * sizeof(Item);
      for (int i = 0; i < n; ++i) {
        p[i].typeVersionLockObsolete.store(0b100);
        p[i].entry_type = 0;
      }
      RT_ASSERT(p != NULL && p != (Item * )(-1));
      return p;
    }

    void delete_items(Item *p, int n) {
      item_allocator.deallocate(p, n);
      al[omp_get_thread_num()].field -= n * sizeof(Item);
    }

    /// build an empty tree
    Node *build_tree_none() {
      Node *node = new_nodes(1);
      node->is_two = 0;
      node->build_size = 0;
      node->size = 0;
      node->fixed = 0;
      //node->num_inserts = node->num_insert_to_data = 0;
      node->num_items = 1;
      node->model.clear();
      node->items = new_items(1);
      node->items[0].entry_type = 0;

      node->p_conflict = 1 / (0.1 * 2 * 64);
      node->conflict_distribution = std::bernoulli_distribution(node->p_conflict);
      node->build_time = timeSinceEpochNanosec();
      node->speed = 0.001;
      node->last_adjust_type = 1;
      return node;
    }

    /// build a tree with two keys
    Node *build_tree_two(T key1, P value1, T key2, P value2, long double _speed, long _time, int _type) {
      if (key1 > key2) {
        std::swap(key1, key2);
        std::swap(value1, value2);
      }
      // printf("%d, %d\n", key1, key2);
      RT_ASSERT(key1 < key2);

      Node *node = NULL;
      if (pending_two[omp_get_thread_num()].empty()) {
        node = new_nodes(1);
        node->is_two = 1;
        node->build_size = 2;
        node->size = 2;
        node->fixed = 0;
        //node->num_inserts = node->num_insert_to_data = 0;

        node->num_items = 8;
        node->items = new_items(node->num_items);
      } else {
        node = pending_two[omp_get_thread_num()].top();
        pending_two[omp_get_thread_num()].pop();

      }
      RT_ASSERT(node->build_size == 2);
      RT_ASSERT(node->num_items == 8);

      const long double mid1_key = key1;
      const long double mid2_key = key2;

      const long double mid1_target = static_cast<long double>(node->num_items) / 3;
      const long double mid2_target = static_cast<long double>(node->num_items) * 2 / 3;

      node->model.train_two(mid1_key, mid2_key, mid1_target, mid2_target);

      /*node->model.a1= node->model.a2= (mid2_target - mid1_target) / (mid2_key - mid1_key);
      node->model.b1= node->model.b2= mid1_target - node->model.a1 * mid1_key;
      node->model.mid=(mid1_key+mid2_key)/2;*/
      /*node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
      node->model.b = mid1_target - node->model.a * mid1_key;*/
      //RT_ASSERT(isfinite(node->model.a));
      //RT_ASSERT(isfinite(node->model.b));

      { // insert key1&value1
        int pos = PREDICT_POS(node, key1);
        //std::cout<<"insert key1&value1 "<<pos<<" "<<key1<<std::endl;
        if (node->items[pos].entry_type != 0) {
          std::cout << "node->items[pos].entry_type " << pos << std::endl;
          std::cout << "node->items[pos].entry_type " << node << std::endl;
          std::cout << "node->items[pos].entry_type " << node->items[pos].entry_type << std::endl;
        }
        RT_ASSERT(node->items[pos].entry_type == 0);
        node->items[pos].entry_type = 2;
        node->items[pos].comp.data.key = key1;
        node->items[pos].comp.data.value = value1;
      }
      { // insert key2&value2
        int pos = PREDICT_POS(node, key2);
        //std::cout<<"insert key2&value2 "<<pos<<" "<<key2<<std::endl;
        //node->model.print();
        /* if(node->items[pos].entry_type != 0){
           std::cout<<"insert key1&value1 "<<pos<<" "<<key1<<std::endl;
           std::cout<<"insert key2&value2 "<<pos<<" "<<key2<<std::endl;
           node->model.print();
         }*/
        RT_ASSERT(node->items[pos].entry_type == 0);
        node->items[pos].entry_type = 2;
        node->items[pos].comp.data.key = key2;
        node->items[pos].comp.data.value = value2;
      }
      //prob
      node->p_conflict = 1 / (0.1 * 2 * 64);
      node->conflict_distribution = std::bernoulli_distribution(node->p_conflict);
      node->build_time = _time;
      node->speed = _speed;
      node->last_adjust_type = _type;

      return node;
    }

    struct bulk_args {
        int _size;
        long double _speed;
        uint64_t _time;
        int _type;
        long double ratio;
    };

    /// bulk build, _keys must be sorted in asc order.
    Node *
    build_tree_bulk(std::vector <T> *_keys, std::vector <P> *_values, bulk_args &args) {
#if READ_EVOLVE
      args._type=2;
      return build_tree_bulk_two_model(_keys, _values, args);
#else
      return build_tree_bulk_fmcd(_keys, _values, args);
#endif
      /*if (USE_FMCD) {
        return build_tree_bulk_fmcd(_keys, _values, _size);
      } else {
        return build_tree_bulk_fast(_keys, _values, _size);
      }*/
    }

    /// bulk build, _keys must be sorted in asc order.
    /// split keys into three parts at each node.
    Node *build_tree_bulk_fast(T *_keys, P *_values, int _size) {
      RT_ASSERT(_size > 1);

      typedef struct {
          int begin;
          int end;
          int level; // top level = 1
          Node *node;
      } Segment;
      std::stack <Segment> s;

      Node *ret = new_nodes(1);
      s.push((Segment) {0, _size, 1, ret});

      while (!s.empty()) {
        const int begin = s.top().begin;
        const int end = s.top().end;
        const int level = s.top().level;
        Node *node = s.top().node;
        s.pop();

        RT_ASSERT(end - begin >= 2);
        if (end - begin == 2) {
          Node *_ = build_tree_two(_keys[begin], _values[begin], _keys[begin + 1],
                                   _values[begin + 1]);
          memcpy(node, _, sizeof(Node));
          delete_nodes(_, 1);
        } else {
          T *keys = _keys + begin;
          P *values = _values + begin;
          const int size = end - begin;
          const int BUILD_GAP_CNT = compute_gap_count(size);

          node->is_two = 0;
          node->build_size = size;
          node->size = size;
          node->fixed = 0;
          //node->num_inserts = node->num_insert_to_data = 0;

          int mid1_pos = (size - 1) / 3;
          int mid2_pos = (size - 1) * 2 / 3;

          RT_ASSERT(0 <= mid1_pos);
          RT_ASSERT(mid1_pos < mid2_pos);
          RT_ASSERT(mid2_pos < size - 1);

          const long double mid1_key =
              (static_cast<long double>(keys[mid1_pos]) +
               static_cast<long double>(keys[mid1_pos + 1])) /
              2;
          const long double mid2_key =
              (static_cast<long double>(keys[mid2_pos]) +
               static_cast<long double>(keys[mid2_pos + 1])) /
              2;

          node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
          const double mid1_target =
              mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
              static_cast<int>(BUILD_GAP_CNT + 1) / 2;
          const double mid2_target =
              mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
              static_cast<int>(BUILD_GAP_CNT + 1) / 2;

          node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
          node->model.b = mid1_target - node->model.a * mid1_key;
          RT_ASSERT(isfinite(node->model.a));
          RT_ASSERT(isfinite(node->model.b));

          const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
          node->model.b += lr_remains;
          node->num_items += lr_remains * 2;

          if (size > 1e6) {
            node->fixed = 1;
          }

          node->items = new_items(node->num_items);

          for (int item_i = PREDICT_POS(node, keys[0]), offset = 0;
               offset < size;) {
            int next = offset + 1, next_i = -1;
            while (next < size) {
              next_i = PREDICT_POS(node, keys[next]);
              if (next_i == item_i) {
                next++;
              } else {
                break;
              }
            }
            if (next == offset + 1) {
              node->items[item_i].entry_type = 2;
              node->items[item_i].comp.data.key = keys[offset];
              node->items[item_i].comp.data.value = values[offset];
            } else {
              // ASSERT(next - offset <= (size+2) / 3);
              node->items[item_i].entry_type = 1;
              node->items[item_i].comp.child = new_nodes(1);
              s.push((Segment) {begin + offset, begin + next, level + 1,
                                node->items[item_i].comp.child});
            }
            if (next >= size) {
              break;
            } else {
              item_i = next_i;
              offset = next;
            }
          }
        }
      }

      return ret;
    }

    void get_fmcd_output(int L, int N, T *keys, long double *a, long double *b) {
      int i = 0;
      int D = 1;
      RT_ASSERT(D <= N - 1 - D);
      long double Ut = (static_cast<long double>(keys[N - 1 - D]) -
                        static_cast<long double>(keys[D])) /
                       (static_cast<long double>(L - 2)) +
                       1e-6;
      while (i < N - 1 - D) {
        while (i + D < N && keys[i + D] - keys[i] >= Ut) {
          i++;
        }
        if (i + D >= N) {
          break;
        }
        D = D + 1;
        if (D * 3 > N)
          break;
        RT_ASSERT(D <= N - 1 - D);
        Ut = (static_cast<long double>(keys[N - 1 - D]) -
              static_cast<long double>(keys[D])) /
             (static_cast<double>(L - 2)) +
             1e-6;
      }
      if (D * 3 <= N) {
        //stats.fmcd_success_times++;

        *a = 1.0 / Ut;
        *b =
            (L -
             (*a) * (static_cast<long double>(keys[N - 1 - D]) +
                     static_cast<long double>(keys[D]))) /
            2;
        RT_ASSERT(isfinite(*a));
        RT_ASSERT(isfinite(*b));
      } else {
        //stats.fmcd_broken_times++;

        int mid1_pos = (N - 1) / 3;
        int mid2_pos = (N - 1) * 2 / 3;

        RT_ASSERT(0 <= mid1_pos);
        RT_ASSERT(mid1_pos < mid2_pos);
        RT_ASSERT(mid2_pos < N - 1);

        const long double mid1_key =
            (static_cast<long double>(keys[mid1_pos]) +
             static_cast<long double>(keys[mid1_pos + 1])) /
            2;
        const long double mid2_key =
            (static_cast<long double>(keys[mid2_pos]) +
             static_cast<long double>(keys[mid2_pos + 1])) /
            2;

        const double mid1_target = (L - 1) / 3;
        const double mid2_target = (L - 1) * 2 / 3;

        *a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
        *b = mid1_target - (*a) * mid1_key;
        RT_ASSERT(isfinite(*a));
        RT_ASSERT(isfinite(*b));
      }
    }

    Node *
    build_tree_bulk_ml(std::vector <T> *_keys, std::vector <P> *_values, bulk_args &args) {
      int _size = args._size;
      long double _speed = args._speed;
      uint64_t _time = args._time;
      int _type = args._type;
      //std::cout<<"build_tree_bulk_fmcd "<<std::to_string(args.ratio)<<std::endl;
      long double ratio = args.ratio < 1 ? 1 : args.ratio;
      ratio = ratio > 8 ? 8 : ratio;
      ratio = 1;
      RT_ASSERT(_size > 1);


      typedef struct {
          int begin;
          int end;
          int level; // top level = 1
          Node *node;
          long double speed;
      } Segment;
      std::stack <Segment> s;

      Node *ret = new_nodes(1);
      s.push((Segment) {0, _size, 1, ret, _speed});

      while (!s.empty()) {
        const int begin = s.top().begin;
        const int end = s.top().end;
        const int level = s.top().level;
        long double speed = s.top().speed;
        //std::cout<<"fmcd _speed "<<speed<<std::endl;
        Node *node = s.top().node;
        s.pop();
        RT_ASSERT(end - begin >= 2);
        if (end - begin == 2) {
          Node *_ = build_tree_two((*_keys)[begin], (*_values)[begin], (*_keys)[begin + 1],
                                   (*_values)[begin + 1], speed, _time, _type);
          memcpy(node, _, sizeof(Node));
          delete_nodes(_, 1);
        } else {
          T *keys = &((*_keys)[begin]);
          P *values = &((*_values)[begin]);
          const int size = end - begin;
          const int BUILD_GAP_CNT = compute_gap_count(size);

          node->is_two = 0;
          node->build_size = size;
          node->size = size;
          node->fixed = 0;
          //node->num_inserts = node->num_insert_to_data = 0;

          //prob
          if (node->build_size < 64) {
            node->p_conflict = 1 / (0.1 * 2 * 64);
          } else {
            node->p_conflict = 1 / (0.1 * 2 * node->build_size);
          }
          node->conflict_distribution = std::bernoulli_distribution(node->p_conflict);
          node->build_time = _time;
          node->speed = speed;
          node->last_adjust_type = _type;
          node->num_items = 0;

          int init_segment_count = 8;
          if (size < init_segment_count * 1024 * 1024) {
            //std::cout<<"??????????????"<<std::endl;
            //std::cout<<size<<std::endl;
            /*if(size==3){
              for(int i=0;i<size;i++){
                std::cout<<keys[i]<<std::endl;
              }
            }*/
            node->model.top_param.a = 0;
            node->model.top_param.b = 0;
            node->model.segment_count = 1;
            long double tmp_a;
            long double tmp_b;
            int segment_size = size * static_cast<int>(BUILD_GAP_CNT + 1);
            get_fmcd_output(segment_size, size, keys, &tmp_a, &tmp_b);
            /*node->model.params.emplace_back(tmp_a,tmp_b);
            node->model.segment_size.push_back(segment_size);
            node->model.segment_offset.push_back(0);*/
            node->model.params[0] = model_param(tmp_a, tmp_b);
            node->model.segment_size[0] = segment_size;
            node->model.segment_offset[0] = 0;
            node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
            /*if(size==3){
              node->model.print();
              for(int i=0;i<size;i++){
                std::cout<<node->model.predict_pos(keys[i])<<std::endl;
                std::cout<<PREDICT_POS(node,keys[i])<<std::endl;
              }
            }*/
            //node->model.print();
            //std::cout<<"!!!!!!!!!!!!!!"<<std::endl;
          } else {
            //std::cout<<"??????????????"<<init_segment_count<<std::endl;
            node->model.segment_count = init_segment_count;
            int segment_count = node->model.segment_count;
            int N = size;
            long double a = 0;
            long double b = 0;
            RT_ASSERT(N > 2);
            get_fmcd_output(segment_count, N, keys, &a, &b);
            node->model.top_param.a = a;
            node->model.top_param.b = b;
            std::vector<int> segments(segment_count, 0);
            for (int i = 0; i < size; i++) {
              double v = a * static_cast<long double>(keys[i]) + b;
              if (v > std::numeric_limits<int>::max() / 2) {
                segments[segment_count - 1]++;
              } else if (v < 0) {
                segments[0]++;
              } else {
                segments[std::min(segment_count - 1, static_cast<int>(v))]++;
              }
            }
            /*if(segments[segment_count-2]<=2||segments[segment_count-1]<=2){
              segment_count--;
              node->model.segment_count--;
            }*/
            /*for(int i=0;i<segment_count;i++){
              std::cout<<N<<" "<<segments[i]<<std::endl;
            }*/
            int offset = 0;
            int segment_offset = 0;
            for (int i = 0; i < segment_count; i++) {
              if (segments[i] < 2) {
                long double left_key =
                    (static_cast<long double>(i) - node->model.top_param.b) / node->model.top_param.a;
                long double right_key =
                    (static_cast<long double>(i + 1) - node->model.top_param.b) / node->model.top_param.a;

                long double tmp_a = (8) / (right_key - left_key);
                long double tmp_b = -(tmp_a) * left_key;
                /*node->model.params.emplace_back(tmp_a, tmp_b);
                node->model.segment_size.push_back(8);
                node->model.segment_offset.push_back(segment_offset);*/
                node->model.params[i] = model_param(tmp_a, tmp_b);
                node->model.segment_size[i] = 8;
                node->model.segment_offset[i] = segment_offset;
                offset += segments[i];
                segment_offset += 8;
                node->num_items += 8;
              } else if (segments[i] == 2) {
                long double mid1_key = keys[offset];
                long double mid2_key = keys[offset + 1];
                long double mid1_target = static_cast<long double>(8) / 3;
                long double mid2_target = static_cast<long double>(8) * 2 / 3;
                long double tmp_a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
                long double tmp_b = mid1_target - (tmp_a) * mid1_key;
                /*node->model.params.emplace_back(tmp_a, tmp_b);
                node->model.segment_size.push_back(8);
                node->model.segment_offset.push_back(segment_offset);*/
                node->model.params[i] = model_param(tmp_a, tmp_b);
                node->model.segment_size[i] = 8;
                node->model.segment_offset[i] = segment_offset;
                offset += segments[i];
                segment_offset += 8;
                node->num_items += 8;
              } else {
                long double tmp_a;
                long double tmp_b;
                int segment_size = segments[i] * static_cast<int>(BUILD_GAP_CNT + 1);
                RT_ASSERT(segments[i] > 2);
                get_fmcd_output(segment_size, segments[i], keys + offset, &tmp_a, &tmp_b);
                /*node->model.params.emplace_back(tmp_a, tmp_b);
                node->model.segment_size.push_back(segment_size);
                node->model.segment_offset.push_back(segment_offset);*/
                node->model.params[i] = model_param(tmp_a, tmp_b);
                node->model.segment_size[i] = segment_size;
                node->model.segment_offset[i] = segment_offset;
                offset += segments[i];
                segment_offset += segment_size;
                node->num_items += segment_size;
              }

            }
          }


          const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
          node->num_items += lr_remains * 2;
          for (int i = 0; i < node->model.segment_count; i++) {
            node->model.params[i].b += lr_remains;
          }

          if (size > 1e6) {
            node->fixed = 1;
          }

          //std::cout<<"!!!!!!!!!!!!!!"<<node->num_items<<std::endl;
          node->items = new_items(node->num_items);

          for (int item_i = PREDICT_POS(node, keys[0]), offset = 0;
               offset < size;) {
            int next = offset + 1, next_i = -1;
            while (next < size) {
              next_i = PREDICT_POS(node, keys[next]);
              //std::cout<<std::to_string(keys[next])<<" next_i: "<<next_i<<"\n";
              if (next_i == item_i) {
                next++;
              } else {
                break;
              }
            }
            if (next == offset + 1) {
              node->items[item_i].entry_type = 2;
              node->items[item_i].comp.data.key = keys[offset];
              node->items[item_i].comp.data.value = values[offset];
            } else {
              // RT_ASSERT(next - offset <= (size+2) / 3);
              node->items[item_i].entry_type = 1;
              node->items[item_i].comp.child = new_nodes(1);
              s.push((Segment) {begin + offset, begin + next, level + 1, node->items[item_i].comp.child,
                                speed / node->num_items});
              /*if(speed / node->num_items ==0){
                std::cout<<"speed == 0 parent_speed: "<<speed<<"num_items: "<<std::to_string(node->num_items)<<std::endl;
              }*/
            }
            if (next >= size) {
              break;
            } else {
              item_i = next_i;
              offset = next;
            }
          }
        }
      }

      return ret;
    }

    Node *
    build_tree_bulk_two_model(std::vector <T> *_keys, std::vector <P> *_values, bulk_args &args) {
      int _size = args._size;
      long double _speed = args._speed;
      uint64_t _time = args._time;
      int _type = args._type;
      //std::cout<<"build_tree_bulk_fmcd "<<std::to_string(args.ratio)<<std::endl;
      long double ratio = args.ratio < 1 ? 1 : args.ratio;
      ratio = ratio > 8 ? 8 : ratio;
      ratio = 1;
      RT_ASSERT(_size > 1);


      typedef struct {
          int begin;
          int end;
          int level; // top level = 1
          Node *node;
          long double speed;
      } Segment;
      std::stack <Segment> s;

      Node *ret = new_nodes(1);
      s.push((Segment) {0, _size, 1, ret, _speed});

      while (!s.empty()) {
        const int begin = s.top().begin;
        const int end = s.top().end;
        const int level = s.top().level;
        long double speed = s.top().speed;
        //std::cout<<"fmcd _speed "<<speed<<std::endl;
        Node *node = s.top().node;
        s.pop();
        RT_ASSERT(end - begin >= 2);
        if (end - begin == 2) {
          Node *_ = build_tree_two((*_keys)[begin], (*_values)[begin], (*_keys)[begin + 1],
                                   (*_values)[begin + 1], speed, _time, _type);
          memcpy(node, _, sizeof(Node));
          delete_nodes(_, 1);
        } else {
          T *keys = &((*_keys)[begin]);
          P *values = &((*_values)[begin]);
          const int size = end - begin;
          const int BUILD_GAP_CNT = compute_gap_count(size);

          node->is_two = 0;
          node->build_size = size;
          node->size = size;
          node->fixed = 0;
          //node->num_inserts = node->num_insert_to_data = 0;

          //prob
          if (node->build_size < 64) {
            node->p_conflict = 1 / (0.1 * 2 * 64);
          } else {
            node->p_conflict = 1 / (0.1 * 2 * node->build_size);
          }
          node->conflict_distribution = std::bernoulli_distribution(node->p_conflict);
          node->build_time = _time;
          node->speed = speed;
          node->last_adjust_type = _type;
          node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
          node->model.type = 1;
          long double mid_target;
          if (size == 3) {
            node->model.mid = static_cast<long double>(keys[size / 2]);
            mid_target = static_cast<long double>(node->num_items - 1) *
                         (1 - (node->model.mid - static_cast<long double>(keys[0])) / (keys[size - 1] - keys[0]));
          } else {
            int left, right;
            T max = 0;
            for (int i = 1; i < size; i++) {
              T diff = keys[i] - keys[i - 1];
              if (diff > max) {
                left = i - 1;
                right = i;
                max = diff;
              } else if (diff == max) {
                right = i;
              }
            }
            node->model.mid = (static_cast<long double>(keys[left]) + static_cast<long double>(keys[right])) / 2;
            int t = -1;
            if (right - left == 1) {
              t = left;
            } else {
              int t_l = left;
              int t_r = right + 1;
              while (t_l < t_r) {
                int t_mid = t_l + (t_r - t_l) / 2;
                if (keys[t_mid] == node->model.mid) {
                  t = t_mid - 1;
                  break;
                } else if (keys[t_mid] < node->model.mid) {
                  t_l = t_mid + 1;
                } else {
                  t_r = t_mid;
                }
              }
              if (t == -1) {
                t = t_r - 1;
              }
            }
            mid_target = static_cast<long double>(node->num_items - 1) * ((long double) (t + 1) / (size));
          }

          node->model.a1 = (mid_target) / (node->model.mid - static_cast<long double>(keys[0]));
          node->model.b1 = (mid_target) - node->model.a1 * node->model.mid;
          node->model.a2 = (static_cast<long double>(node->num_items) - 1 - mid_target) /
                           (static_cast<long double>(keys[size - 1] - node->model.mid));
          node->model.b2 = mid_target - node->model.a2 * node->model.mid;


          const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
          node->num_items += lr_remains * 2;
          node->model.b1 += lr_remains;
          node->model.b2 += lr_remains;

          if (size > 1e6) {
            node->fixed = 1;
          }

          //std::cout<<"!!!!!!!!!!!!!!"<<node->num_items<<std::endl;
          node->items = new_items(node->num_items);

          for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size;) {
            int next = offset + 1, next_i = -1;
            while (next < size) {
              next_i = PREDICT_POS(node, keys[next]);
              //std::cout<<std::to_string(keys[next])<<" next_i: "<<next_i<<"\n";
              if (next_i == item_i) {
                next++;
              } else {
                break;
              }
            }
            if (next == offset + 1) {
              node->items[item_i].entry_type = 2;
              node->items[item_i].comp.data.key = keys[offset];
              node->items[item_i].comp.data.value = values[offset];
            } else {
              // RT_ASSERT(next - offset <= (size+2) / 3);
              node->items[item_i].entry_type = 1;
              node->items[item_i].comp.child = new_nodes(1);
              s.push((Segment) {begin + offset, begin + next, level + 1, node->items[item_i].comp.child,
                                speed / node->num_items});
              /*if(speed / node->num_items ==0){
                std::cout<<"speed == 0 parent_speed: "<<speed<<"num_items: "<<std::to_string(node->num_items)<<std::endl;
              }*/
            }
            if (next >= size) {
              break;
            } else {
              item_i = next_i;
              offset = next;
            }
          }
        }
      }

      return ret;
    }

    /// bulk build, _keys must be sorted in asc order.
    /// FMCD method.
    Node *
    build_tree_bulk_fmcd(std::vector <T> *_keys, std::vector <P> *_values, bulk_args &args) {

      int _size = args._size;
      long double _speed = args._speed;
      uint64_t _time = args._time;
      int _type = args._type;
      //std::cout<<"build_tree_bulk_fmcd "<<std::to_string(args.ratio)<<std::endl;
      double ratio = args.ratio < 1 ? 0 : (args.ratio - 1);
      //std::cout<<"build_tree_bulk_fmcd 0 "<<std::to_string(_size)<<std::endl;
      RT_ASSERT(_size > 1);

      typedef struct {
          int begin;
          int end;
          int level; // top level = 1
          Node *node;
          long double speed;
      } Segment;
      std::stack <Segment> s;

      Node *ret = new_nodes(1);
      s.push((Segment) {0, _size, 1, ret, _speed});

      while (!s.empty()) {
        const int begin = s.top().begin;
        const int end = s.top().end;
        const int level = s.top().level;
        long double speed = s.top().speed;
        //std::cout<<"fmcd _speed "<<speed<<std::endl;
        Node *node = s.top().node;
        s.pop();
        RT_ASSERT(end - begin >= 2);
        if (end - begin == 2) {
          Node *_ = build_tree_two((*_keys)[begin], (*_values)[begin], (*_keys)[begin + 1],
                                   (*_values)[begin + 1], speed, _time, _type);
          memcpy(node, _, sizeof(Node));
          delete_nodes(_, 1);
        } else {
          T *keys = &((*_keys)[begin]);
          P *values = &((*_values)[begin]);
          const int size = end - begin;
          const double BUILD_GAP_CNT = compute_gap_count(size);
          const double max_ratio = compute_max_ratio(size);
          double tmp_ratio = ratio > max_ratio ? max_ratio : ratio;
          node->is_two = 0;
          node->build_size = size;
          node->size = size;
          node->fixed = 0;
          //node->num_inserts = node->num_insert_to_data = 0;

          //prob
          if (node->build_size < 64) {
            node->p_conflict = 1 / (0.1 * 1 * 64);
          } else
            node->p_conflict = 1 / (0.1 * 1 * node->build_size);
          node->conflict_distribution = std::bernoulli_distribution(node->p_conflict);
          node->build_time = _time;
          node->speed = speed;
          node->last_adjust_type = _type;

          // FMCD method
          // Here the implementation is a little different with Algorithm 1 in our
          // paper. In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L
          // - 2). But according to the derivation described in our paper, M.A
          // should be less than 1 / U_T. So we added a small number (1e-6) to
          // U_T. In fact, it has only a negligible impact of the performance.
          {
            int L = static_cast<int>(size * (BUILD_GAP_CNT + 1 + tmp_ratio));
            int i = 0;
            int D = 1;
            RT_ASSERT(D <= size - 1 - D);
            long double Ut = (static_cast<long double>(keys[size - 1 - D]) -
                              static_cast<long double>(keys[D])) /
                             (static_cast<double>(L - 2)) +
                             1e-6;
            while (i < size - 1 - D) {
              while (i + D < size && keys[i + D] - keys[i] >= Ut) {
                i++;
              }
              if (i + D >= size) {
                break;
              }
              D = D + 1;
              if (D * 3 > size)
                break;
              RT_ASSERT(D <= size - 1 - D);
              Ut = (static_cast<long double>(keys[size - 1 - D]) -
                    static_cast<long double>(keys[D])) /
                   (static_cast<double>(L - 2)) +
                   1e-6;
            }
            if (D * 3 <= size) {
              stats.fmcd_success_times++;

              node->model.a1 = 1.0 / Ut;
              node->model.b1 =
                  (L -
                   node->model.a1 * (static_cast<long double>(keys[size - 1 - D]) +
                                     static_cast<long double>(keys[D]))) /
                  2;
              RT_ASSERT(isfinite(node->model.a1));
              RT_ASSERT(isfinite(node->model.b1));
              node->num_items = L;
            } else {
              stats.fmcd_broken_times++;

              int mid1_pos = (size - 1) / 3;
              int mid2_pos = (size - 1) * 2 / 3;

              RT_ASSERT(0 <= mid1_pos);
              RT_ASSERT(mid1_pos < mid2_pos);
              RT_ASSERT(mid2_pos < size - 1);

              const long double mid1_key =
                  (static_cast<long double>(keys[mid1_pos]) +
                   static_cast<long double>(keys[mid1_pos + 1])) /
                  2;
              const long double mid2_key =
                  (static_cast<long double>(keys[mid2_pos]) +
                   static_cast<long double>(keys[mid2_pos + 1])) /
                  2;

              node->num_items = L;
              //node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
              const double mid1_target =
                  mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1 + tmp_ratio) +
                  static_cast<int>(BUILD_GAP_CNT + 1 + tmp_ratio) / 2;
              const double mid2_target =
                  mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1 + tmp_ratio) +
                  static_cast<int>(BUILD_GAP_CNT + 1 + tmp_ratio) / 2;

              node->model.a1 = (mid2_target - mid1_target) / (mid2_key - mid1_key);
              node->model.b1 = mid1_target - node->model.a1 * mid1_key;
              RT_ASSERT(isfinite(node->model.a1));
              RT_ASSERT(isfinite(node->model.b1));
            }
          }

          /*std::cout<<"build_tree_bulk_fmcd 1 "<<std::to_string(node->num_items)<<std::endl;
          std::cout<<"build_tree_bulk_fmcd 2 "<<std::to_string(size * static_cast<int>(BUILD_GAP_CNT + 1))<<std::endl;
          std::cout<<"build_tree_bulk_fmcd 3 "<<std::to_string(static_cast<int>(ratio))<<std::endl;*/
          RT_ASSERT(node->model.a1 >= 0);
          const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
          node->model.b1 += lr_remains;
          node->num_items += lr_remains * 2;
          node->model.type = 0;

          if (size > 1e6) {
            node->fixed = 1;
          }

          node->items = new_items(node->num_items);

          for (int item_i = PREDICT_POS(node, keys[0]), offset = 0;
               offset < size;) {
            int next = offset + 1, next_i = -1;
            while (next < size) {
              next_i = PREDICT_POS(node, keys[next]);
              if (next_i == item_i) {
                next++;
              } else {
                break;
              }
            }
            if (next == offset + 1) {
              node->items[item_i].entry_type = 2;
              node->items[item_i].comp.data.key = keys[offset];
              node->items[item_i].comp.data.value = values[offset];
            } else {
              // RT_ASSERT(next - offset <= (size+2) / 3);
              node->items[item_i].entry_type = 1;
              node->items[item_i].comp.child = new_nodes(1);
              s.push((Segment) {begin + offset, begin + next, level + 1, node->items[item_i].comp.child,
                                speed / node->num_items});
              /*if(speed / node->num_items ==0){
                std::cout<<"speed == 0 parent_speed: "<<speed<<"num_items: "<<std::to_string(node->num_items)<<std::endl;
              }*/
            }
            if (next >= size) {
              break;
            } else {
              item_i = next_i;
              offset = next;
            }
          }
        }

      }

      return ret;
    }

    Node *
    build_tree_compress(std::vector <T> *_keys, std::vector <P> *_values, bulk_args &args) {
      int _size = args._size;
      long double _speed = args._speed;
      uint64_t _time = args._time;
      int _type = args._type;
      long double ratio = args.ratio < 1 ? 1 : args.ratio;
      ratio = ratio > 8 ? 8 : ratio;
      ratio = 1;
      RT_ASSERT(_size > 1);


      typedef struct {
          int begin;
          int end;
          int level; // top level = 1
          Node *node;
          long double speed;
      } Segment;
      std::stack <Segment> s;

      Node *ret = new_nodes(1);
      s.push((Segment) {0, _size, 1, ret, _speed});

      while (!s.empty()) {
        const int begin = s.top().begin;
        const int end = s.top().end;
        const int level = s.top().level;
        long double speed = s.top().speed;
        //std::cout<<"fmcd _speed "<<speed<<std::endl;
        Node *node = s.top().node;
        s.pop();
        RT_ASSERT(end - begin >= 2);
        if (end - begin == 2) {
          Node *_ = build_tree_two((*_keys)[begin], (*_values)[begin], (*_keys)[begin + 1],
                                   (*_values)[begin + 1], speed, _time, _type);
          memcpy(node, _, sizeof(Node));
          delete_nodes(_, 1);
        } else {
          T *keys = &((*_keys)[begin]);
          P *values = &((*_values)[begin]);
          const int size = end - begin;
          const int BUILD_GAP_CNT = compute_gap_count(size);

          node->is_two = 0;
          node->build_size = size;
          node->size = size;
          node->fixed = 0;
          //node->num_inserts = node->num_insert_to_data = 0;

          //prob
          if (node->build_size < 64) {
            node->p_conflict = 1 / (0.1 * 2 * 64);
          } else {
            node->p_conflict = 1 / (0.1 * 2 * node->build_size);
          }
          node->conflict_distribution = std::bernoulli_distribution(node->p_conflict);
          node->build_time = _time;
          node->speed = speed;
          node->last_adjust_type = _type;
          node->num_items = 0;

          int init_segment_count = 8;
          if (size < init_segment_count * 1024 * 1024) {
            //std::cout<<"??????????????"<<std::endl;
            //std::cout<<size<<std::endl;
            /*if(size==3){
              for(int i=0;i<size;i++){
                std::cout<<keys[i]<<std::endl;
              }
            }*/
            node->model.top_param.a = 0;
            node->model.top_param.b = 0;
            node->model.segment_count = 1;
            long double tmp_a;
            long double tmp_b;
            int segment_size = size * static_cast<int>(BUILD_GAP_CNT + 1);
            get_fmcd_output(segment_size, size, keys, &tmp_a, &tmp_b);
            /*node->model.params.emplace_back(tmp_a,tmp_b);
            node->model.segment_size.push_back(segment_size);
            node->model.segment_offset.push_back(0);*/
            node->model.params[0] = model_param(tmp_a, tmp_b);
            node->model.segment_size[0] = segment_size;
            node->model.segment_offset[0] = 0;
            node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
            /*if(size==3){
              node->model.print();
              for(int i=0;i<size;i++){
                std::cout<<node->model.predict_pos(keys[i])<<std::endl;
                std::cout<<PREDICT_POS(node,keys[i])<<std::endl;
              }
            }*/
            //node->model.print();
            //std::cout<<"!!!!!!!!!!!!!!"<<std::endl;
          } else {
            //std::cout<<"??????????????"<<init_segment_count<<std::endl;
            node->model.segment_count = init_segment_count;
            int segment_count = node->model.segment_count;
            int N = size;
            long double a = 0;
            long double b = 0;
            RT_ASSERT(N > 2);
            get_fmcd_output(segment_count, N, keys, &a, &b);
            node->model.top_param.a = a;
            node->model.top_param.b = b;
            std::vector<int> segments(segment_count, 0);
            for (int i = 0; i < size; i++) {
              double v = a * static_cast<long double>(keys[i]) + b;
              if (v > std::numeric_limits<int>::max() / 2) {
                segments[segment_count - 1]++;
              } else if (v < 0) {
                segments[0]++;
              } else {
                segments[std::min(segment_count - 1, static_cast<int>(v))]++;
              }
            }
            /*if(segments[segment_count-2]<=2||segments[segment_count-1]<=2){
              segment_count--;
              node->model.segment_count--;
            }*/
            /*for(int i=0;i<segment_count;i++){
              std::cout<<N<<" "<<segments[i]<<std::endl;
            }*/
            int offset = 0;
            int segment_offset = 0;
            for (int i = 0; i < segment_count; i++) {
              if (segments[i] < 2) {
                long double left_key =
                    (static_cast<long double>(i) - node->model.top_param.b) / node->model.top_param.a;
                long double right_key =
                    (static_cast<long double>(i + 1) - node->model.top_param.b) / node->model.top_param.a;

                long double tmp_a = (8) / (right_key - left_key);
                long double tmp_b = -(tmp_a) * left_key;
                /*node->model.params.emplace_back(tmp_a, tmp_b);
                node->model.segment_size.push_back(8);
                node->model.segment_offset.push_back(segment_offset);*/
                node->model.params[i] = model_param(tmp_a, tmp_b);
                node->model.segment_size[i] = 8;
                node->model.segment_offset[i] = segment_offset;
                offset += segments[i];
                segment_offset += 8;
                node->num_items += 8;
              } else if (segments[i] == 2) {
                long double mid1_key = keys[offset];
                long double mid2_key = keys[offset + 1];
                long double mid1_target = static_cast<long double>(8) / 3;
                long double mid2_target = static_cast<long double>(8) * 2 / 3;
                long double tmp_a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
                long double tmp_b = mid1_target - (tmp_a) * mid1_key;
                /*node->model.params.emplace_back(tmp_a, tmp_b);
                node->model.segment_size.push_back(8);
                node->model.segment_offset.push_back(segment_offset);*/
                node->model.params[i] = model_param(tmp_a, tmp_b);
                node->model.segment_size[i] = 8;
                node->model.segment_offset[i] = segment_offset;
                offset += segments[i];
                segment_offset += 8;
                node->num_items += 8;
              } else {
                long double tmp_a;
                long double tmp_b;
                int segment_size = segments[i] * static_cast<int>(BUILD_GAP_CNT + 1);
                RT_ASSERT(segments[i] > 2);
                get_fmcd_output(segment_size, segments[i], keys + offset, &tmp_a, &tmp_b);
                /*node->model.params.emplace_back(tmp_a, tmp_b);
                node->model.segment_size.push_back(segment_size);
                node->model.segment_offset.push_back(segment_offset);*/
                node->model.params[i] = model_param(tmp_a, tmp_b);
                node->model.segment_size[i] = segment_size;
                node->model.segment_offset[i] = segment_offset;
                offset += segments[i];
                segment_offset += segment_size;
                node->num_items += segment_size;
              }

            }
          }


          const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
          node->num_items += lr_remains * 2;
          for (int i = 0; i < node->model.segment_count; i++) {
            node->model.params[i].b += lr_remains;
          }

          if (size > 1e6) {
            node->fixed = 1;
          }

          //std::cout<<"!!!!!!!!!!!!!!"<<node->num_items<<std::endl;
          node->items = new_items(node->num_items);

          for (int item_i = PREDICT_POS(node, keys[0]), offset = 0;
               offset < size;) {
            int next = offset + 1, next_i = -1;
            while (next < size) {
              next_i = PREDICT_POS(node, keys[next]);
              //std::cout<<std::to_string(keys[next])<<" next_i: "<<next_i<<"\n";
              if (next_i == item_i) {
                next++;
              } else {
                break;
              }
            }
            if (next == offset + 1) {
              node->items[item_i].entry_type = 2;
              node->items[item_i].comp.data.key = keys[offset];
              node->items[item_i].comp.data.value = values[offset];
            } else {
              // RT_ASSERT(next - offset <= (size+2) / 3);
              node->items[item_i].entry_type = 1;
              node->items[item_i].comp.child = new_nodes(1);
              s.push((Segment) {begin + offset, begin + next, level + 1, node->items[item_i].comp.child,
                                speed / node->num_items});
              /*if(speed / node->num_items ==0){
                std::cout<<"speed == 0 parent_speed: "<<speed<<"num_items: "<<std::to_string(node->num_items)<<std::endl;
              }*/
            }
            if (next >= size) {
              break;
            } else {
              item_i = next_i;
              offset = next;
            }
          }
        }
      }

      return ret;
    }

    void destory_pending() {
      std::unordered_set < Node * > s;
      for (int i = 0; i < 1024; ++i) {

        while (!pending_two[i].empty()) {
          Node *node = pending_two[i].top();
          pending_two[i].pop();
          if (s.find(node) != s.end()) {
            printf("Error: destory_pending dup node %p", node);
          } else
            s.insert(node);
          delete_items(node->items, node->num_items);
          delete_nodes(node, 1);
        }
      }
      //std::cout<<" aaaaaaaaaaaa "<<s.size()<<std::endl;
      /*for(auto *node:s){
        delete_items(node->items, node->num_items);
        delete_nodes(node, 1);
      }*/

    }

    void destroy_tree(Node *root) {
      std::stack < Node * > s;
      s.push(root);
      while (!s.empty()) {
        Node *node = s.top();
        s.pop();

        for (int i = 0; i < node->num_items; i++) {
          if (node->items[i].entry_type == 1) {
            s.push(node->items[i].comp.child);
          }
        }

        if (node->is_two) {
          RT_ASSERT(node->build_size == 2);
          RT_ASSERT(node->num_items == 8);
          node->size = 2;
          //node->num_inserts = node->num_insert_to_data = 0;
          for (int i = 0; i < node->num_items; i++) node->items[i].typeVersionLockObsolete.store(0b100);;
          for (int i = 0; i < node->num_items; i++) node->items[i].entry_type = 0;
          pending_two[omp_get_thread_num()].push(node);
        } else {
          delete_items(node->items, node->num_items);
          delete_nodes(node, 1);
        }
      }
    }

    void dfs(Node *node, std::vector <T> *keys, std::vector <P> *values) {
      for (int i = 0; i < node->num_items; i++) { // the i-th entry of the node now
        if (node->items[i].entry_type == 2) { // means it is a data
          keys->push_back(node->items[i].comp.data.key);
          values->push_back(node->items[i].comp.data.value);
        } else if (node->items[i].entry_type == 1) {
          dfs(node->items[i].comp.child, keys, values);
        }
      }
      if (node->cooling == 1) {
        delete_from_cooling_pool(node);
      }
      if (node->is_two) {
        RT_ASSERT(node->build_size == 2);
        RT_ASSERT(node->num_items == 8);
        node->size = 2;
        //node->num_inserts = node->num_insert_to_data = 0;
        safe_delete_nodes(node, 1);
      } else {
        safe_delete_nodes(node, 1);
      }
    }

    void dfs2(Node *node, std::vector <T> *keys, std::vector <P> *values) {
      for (int i = 0; i < node->num_items; i++) { // the i-th entry of the node now
        if (node->items[i].entry_type == 2) { // means it is a data
          keys->push_back(node->items[i].comp.data.key);
          values->push_back(node->items[i].comp.data.value);
        } else if (node->items[i].entry_type == 1) {
          dfs2(node->items[i].comp.child, keys, values);
        }
      }
      //std::cout<<"ebr->addG(node)"<<node<< std::endl;
      if (node->is_two) {
        RT_ASSERT(node->build_size == 2);
        RT_ASSERT(node->num_items == 8);
        node->size = 2;
        //node->num_inserts = node->num_insert_to_data = 0;
        ebr->addG(node);
      } else {
        ebr->addG(node);
      }
    }

    int count_tree_size(Node *_subroot) {
      std::list < Node * > bfs;
      bfs.push_back(_subroot);
      int count = 0;
      bool needRestart = false;
      uint64_t versionItem;
      while (!bfs.empty()) {
        Node *node = bfs.front();
        bfs.pop_front();
        if (count + node->build_size >= 64) {
          return 64;
        }
        for (int i = 0; i < node->num_items; i++) {
          versionItem = node->items[i].readLockOrRestart(needRestart);
          if (needRestart) {
            return -1;
          }
          if (node->items[i].entry_type == 2) {
            count++;
            if (count >= 64) {
              return count;
            }
          } else if (node->items[i].entry_type == 1) { // child
            bfs.push_back(node->items[i].comp.child);
          }
          node->items[i].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart) {
            return -1;
          }
        }
      }
      return count;
    }

    int scan_and_destory_tree2(
        Node *_subroot, std::vector <T> **keys, std::vector <P> **values, // keys here is ptr to ptr
        bool destory = true) {

      std::list < Node * > bfs;
      std::list < Item * > lockedItems;

      bfs.push_back(_subroot);
      bool needRestart = false;
      //int count=0;

      while (!bfs.empty()) {
        Node *node = bfs.front();
        bfs.pop_front();

        for (int i = 0; i < node->num_items;
             i++) { // the i-th entry of the node now
          node->items[i].writeLockOrRestart(needRestart);

          if (needRestart) {
            // release locks on all locked items
            for (auto &n: lockedItems) {
              n->writeUnlock();
            }
            return -1;
          }
          lockedItems.push_back(&(node->items[i]));

          if (node->items[i].entry_type == 1) { // child
            bfs.push_back(node->items[i].comp.child);
          }
        }
      } // end while



      *keys = new std::vector<T>();
      *values = new std::vector<P>();
      (*keys)->reserve(_subroot->build_size);
      (*values)->reserve(_subroot->build_size);
      dfs2(_subroot, *keys, *values);

      const int ESIZE = (*keys)->size();
      //std::cout << "dfs ESIZE: " << std::to_string(ESIZE) << std::endl;

      return ESIZE;
    } // end scan_and_destory

    int scan_and_destory_tree(
        Node *_subroot, std::vector <T> **keys, std::vector <P> **values, // keys here is ptr to ptr
        bool destory = true) {

      std::list < Node * > bfs;
      std::list < Item * > lockedItems;

      bfs.push_back(_subroot);
      bool needRestart = false;
      //int count=0;

      while (!bfs.empty()) {
        Node *node = bfs.front();
        bfs.pop_front();

        for (int i = 0; i < node->num_items;
             i++) { // the i-th entry of the node now
          node->items[i].writeLockOrRestart(needRestart);

          if (needRestart) {
            // release locks on all locked items
            for (auto &n: lockedItems) {
              n->writeUnlock();
            }
            return -1;
          }
          lockedItems.push_back(&(node->items[i]));

          if (node->items[i].entry_type == 1) { // child
            bfs.push_back(node->items[i].comp.child);
          }
        }
      } // end while



      *keys = new std::vector<T>();
      *values = new std::vector<P>();
      (*keys)->reserve(_subroot->build_size);
      (*values)->reserve(_subroot->build_size);
      dfs(_subroot, *keys, *values);

      const int ESIZE = (*keys)->size();
      //std::cout << "dfs ESIZE: " << std::to_string(ESIZE) << std::endl;

      return ESIZE;
    } // end scan_and_destory

    void prob_rebuild(Node *node_prob_prev, Node *node_prob_cur, const T &key) {
      //Probability Rebuild Begin


    }

    // Node* insert_tree(Node *_node, const T &key, const P &value) {
    bool insert_tree(const T &key, const P &value) {
      int restartCount = 0;
      restart:
      if (restartCount++)
        yield(restartCount);
      bool needRestart = false;

      constexpr int MAX_DEPTH = 128;
      Node *path[MAX_DEPTH];
      int path_size = 0;
      int insert_to_data = 0;

      // for lock coupling
      uint64_t versionItem;
      Node *parent = nullptr;


      //prob
      //uint32_t temp=this->temp;
      bool conflict_flag = false;

      for (Node *node = root;;) {
        // R-lock this node
        /*if (node_prob_cur == nullptr
            && path_size >= 0
            && node->build_size < 7000000
            ) {
          if ((uintRand() & temp) == temp) {
            node_prob_prev = parent;
            node_prob_cur = node;
          }
        }
        temp>>=1;*/

        RT_ASSERT(path_size < MAX_DEPTH);
        path[path_size++] = node;

        int pos = PREDICT_POS(node, key);
        versionItem = node->items[pos].readLockOrRestart(needRestart);
        if (needRestart)
          goto restart;
        if (node->items[pos].entry_type == 0) // 0 means empty entry
        {
          RT_DEBUG("Empty %p pos %d, locking.", node, pos);
          node->items[pos].upgradeToWriteLockOrRestart(versionItem, needRestart);
          if (needRestart) {
            goto restart;
          }

          node->items[pos].entry_type = 2;
          node->items[pos].comp.data.key = key;
          node->items[pos].comp.data.value = value;


          node->items[pos].writeUnlock();

          break;
        } else if (node->items[pos].entry_type == 2) // 2 means existing entry has data already
        {
          if (node->last_adjust_type == 3) {
            node->items[pos].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart)
              goto restart;
            int compress_pos = get_real_pos_from_compressed_node(node, pos, key);
            if (compress_pos < 0) {
              goto restart;
            }
            versionItem = node->items[compress_pos].readLockOrRestart(needRestart);
            if (needRestart)
              goto restart;
            if (node->items[compress_pos].entry_type == 1) {
              parent = node;
              node = node->items[compress_pos].comp.child;
              parent->items[compress_pos].readUnlockOrRestart(versionItem, needRestart);
              if (needRestart)
                goto restart;
              continue;
            } else {
              pos = compress_pos;
            }
          }
          node->items[pos].upgradeToWriteLockOrRestart(versionItem, needRestart);
          if (needRestart) {
            goto restart;
          }

          node->items[pos].entry_type = 1;
          // printf("%d, %d\n", key, node->items[pos].comp.data.key);
          node->items[pos].comp.child =
              build_tree_two(key, value, node->items[pos].comp.data.key,
                             node->items[pos].comp.data.value,
                             node->speed / node->num_items, timeSinceEpochNanosec(), 1);
          /*if(node->speed / node->build_size==0){
            std::cout<<"speed == 0 origin_speed: "<<node->speed<<"build_size: "<<node->build_size<<std::endl;
          }*/
          insert_to_data = 1;


          node->items[pos].writeUnlock();
          conflict_flag = true;

          break;

        } else // 1 means has a child, need to go down and see
        {
          if (node->last_adjust_type == 3) {
            node->items[pos].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart)
              goto restart;
            int compress_pos = get_real_pos_from_compressed_node(node, pos, key);
            if (compress_pos < 0) {
              goto restart;
            }
            versionItem = node->items[compress_pos].readLockOrRestart(needRestart);
            if (needRestart)
              goto restart;
            if (node->items[compress_pos].entry_type == 2) {
              node->items[compress_pos].upgradeToWriteLockOrRestart(versionItem, needRestart);
              if (needRestart) {
                goto restart;
              }
              node->items[compress_pos].entry_type = 1;
              node->items[compress_pos].comp.child =
                  build_tree_two(key, value, node->items[compress_pos].comp.data.key,
                                 node->items[compress_pos].comp.data.value,
                                 node->speed / node->num_items, timeSinceEpochNanosec(), 1);

              node->items[compress_pos].writeUnlock();
              conflict_flag = true;
              break;
            } else {
              pos = compress_pos;
            }
          }
          parent = node;
          // RT_DEBUG("Child %p pos %d.", node, pos);
          node = node->items[pos].comp.child;           // now: node is the child

          parent->items[pos].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart)
            goto restart;
        }
      }

      /*for (int i = 0; i < path_size; i++) {
        path[i]->num_insert_to_data += insert_to_data;
        path[i]->num_inserts++;
        path[i]->size++;
      }*/

      //***** so, when reaching here, no node is locked.

      if (!conflict_flag) {
        return true;
      }
      Node *node_prob_prev = nullptr;
      Node *node_prob_cur = nullptr;
      int cur_idx = 0;

      uint64_t cur_time = timeSinceEpochNanosec();

      for (int i = 0; i < path_size - 1; i++) {
        Node *node = path[i];
        if (node->fixed == 0) {
          if (node->conflict_distribution(getGen())) {
            //epsilon=0.001
            long double p_acc = (node->speed * (cur_time - node->build_time)) /
                                (node->build_size + 1) + (path_size - i) / (long double) 128 + 0.1;
            /*std::cout << "p_conflict " << node->p_conflict << std::endl;
            std::cout << "node->speed " << node->speed << std::endl;
            std::cout << "cur_time " << cur_time << std::endl;
            std::cout << "node->build_time " << node->build_time << std::endl;
            std::cout << "node->build_size " << node->build_size << std::endl;
            std::cout << "p_acc " << p_acc << std::endl;*/
            if (p_acc >= 1) {
              node_prob_prev = i == 0 ? nullptr : path[i - 1];
              node_prob_cur = node;
              cur_idx = i;
              break;
            }
            std::bernoulli_distribution acc_distribution(p_acc);
            if (acc_distribution(getGen())) {
              node_prob_prev = i == 0 ? nullptr : path[i - 1];
              node_prob_cur = node;
              cur_idx = i;
              //std::cout << "p_conflict " << node->p_conflict << std::endl;
              //std::cout << "============================= " << p_acc << std::endl;
              break;
            }
          }
        }
      }

      if (node_prob_cur != nullptr) {

        int prev_size = node_prob_cur->build_size;
        uint64_t prev_build_time = node_prob_cur->build_time;
        long double prev_speed = node_prob_cur->speed;
        // const int ESIZE = node->size; //race here
        // T *keys = new T[ESIZE];
        // P *values = new P[ESIZE];
        std::vector <T> *keys;   // make it be a ptr here because we will let scan_and_destroy
        // to decide the size after getting the locks
        std::vector <P> *values; // scan_and_destroy will fill up the keys/values

#if COLLECT_TIME
        auto start_time_scan = std::chrono::high_resolution_clock::now();
#endif
        if (prev_size < 64) {
          int t_size = count_tree_size(node_prob_cur);
          if (t_size < 64) {
            return true;
          }
        }
        num_write_probability_trigger++;

        int numKeysCollected = scan_and_destory_tree(
            node_prob_cur, &keys, &values); // pass the (address) of the ptr
        if (numKeysCollected < 0) {
          return true;
        }
#if COLLECT_TIME
        auto end_time_scan = std::chrono::high_resolution_clock::now();
        auto duration_scan = end_time_scan - start_time_scan;
        stats.time_scan_and_destory_tree +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration_scan)
                .count() *
            1e-9;
#endif

#if COLLECT_TIME
        auto start_time_build = std::chrono::high_resolution_clock::now();
#endif

        if (cur_idx > 1) {
          for (int i = 1; i < cur_idx; i++) {
            if (path[i]->cooling == 1) {
              delete_from_cooling_pool(path[i]);
            }
          }
        }
        uint64_t cur_time = timeSinceEpochNanosec();
        long double speed = (long double) (numKeysCollected - prev_size) / (cur_time - prev_build_time + 1);
        //std::cout << "speed: " << speed << " size_inc: " << std::to_string(numKeysCollected - prev_size) << "time: "<< std::to_string(cur_time - prev_build_time) << std::endl;
        /*if(speed ==0){
          std::cout<<"speed == 0 size_inc: "<<std::to_string(numKeysCollected - prev_size)<<"time: "<<std::to_string(cur_time - prev_build_time)<<std::endl;
        }*/
        bulk_args args = {numKeysCollected, speed, cur_time, 1, speed / prev_speed};

        Node *new_node = build_tree_bulk(keys, values, args);


#if COLLECT_TIME
        auto end_time_build = std::chrono::high_resolution_clock::now();
        auto duration_build = end_time_build - start_time_build;
        stats.time_build_tree_bulk +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration_build)
                .count() *
            1e-9;
#endif

        delete keys;
        delete values;

        RT_DEBUG(
            "Final step of adjust, try to update parent/root, new node is %p",
            node);

        if (node_prob_prev != nullptr) {

          int retryLockCount = 0;
          retryLock:
          if (retryLockCount++)
            yield(retryLockCount);

          int pos = PREDICT_POS(node_prob_prev, key);

          bool needRetry = false;

          node_prob_prev->items[pos].writeLockOrRestart(needRetry);
          if (needRetry) {
            RT_DEBUG("Final step of adjust, obtain parent %p lock FAIL, retry",
                     path[i - 1]);
            goto retryLock;
          }
          RT_DEBUG("Final step of adjust, obtain parent %p lock OK, now give "
                   "the adjusted tree to parent",
                   path[i - 1]);
          node_prob_prev->items[pos].comp.child = new_node;
          node_prob_prev->items[pos].writeUnlock();
#if COMPRESS
          if (cool_select_distribution(getGen())) {
            if ((!new_node->fixed) && new_node->num_items >= 64 ) {
            //std::cout << "push_to_cooling_pool, new_node: " << new_node << std::endl;
            push_to_cooling_pool(node_prob_prev, new_node);
            }
            long allocated = 0;
            for (int i = 0; i < 1024; i++) {
              allocated += al[i].field;
            }
            if (allocated > memory_budget_) {
              std::cout << "allocated > memory_budget_, allocated: " << allocated << std::endl;
              compress_cv_.notify_all();
            }
          }
#endif
        } else { // new node is the root, need to update it
          root = new_node;
        }

      }        // end REBUILD

      return true;
    } // end of insert_tree

    // SATISFY_LOWER = true means all the keys in the subtree of `node` are no less than to `lower`.
    template<bool SATISFY_LOWER>
    int range_core_len(std::pair <T, P> *results, int pos, Node *node, const T &lower, int len) {
      if constexpr(SATISFY_LOWER)
      {
        int lower_pos = 0;
        while (lower_pos < node->num_items) {
          if (node->items[lower_pos].entry_type == 2) {
            results[pos] = {node->items[lower_pos].comp.data.key, node->items[lower_pos].comp.data.value};
            pos++;
          } else if (node->items[lower_pos].entry_type == 1) {
            pos = range_core_len < true > (results, pos, node->items[lower_pos].comp.child, lower, len);
          }
          if (pos >= len) {
            return pos;
          }
          lower_pos++;
        }
        return pos;
      } else {

        /*EpochGuard guard(this);
        int restartCount = 0;
        restart:
        if (restartCount++)
          yield(restartCount);
        bool needRestart = false;*/

        // for lock coupling
        //uint64_t versionItem;

        int lower_pos = PREDICT_POS(node, lower);
        if (node->last_adjust_type == 3) {
          lower_pos = get_real_pos_from_compressed_node(node, lower_pos, lower);
        }
        /*versionItem = node->items[lower_pos].readLockOrRestart(needRestart);
        if (needRestart)
          goto restart;*/

        if (node->items[lower_pos].entry_type != 0) {
          if (node->items[lower_pos].entry_type == 2) {
            if (node->items[lower_pos].comp.data.key >= lower) {
              results[pos] = {node->items[lower_pos].comp.data.key, node->items[lower_pos].comp.data.value};
              pos++;
            }

          } else {
            pos = range_core_len < false > (results, pos, node->items[lower_pos].comp.child, lower, len);
          }
          /*node->items[lower_pos].readUnlockOrRestart(versionItem, needRestart);
          if (needRestart)
            goto restart;*/
          if (pos >= len) {
            return pos;
          }
        }
        if (lower_pos + 1 >= node->num_items) {
          return pos;
        }
        lower_pos++;
        while (lower_pos < node->num_items) {
          if (node->items[lower_pos].entry_type != 0) {
            if (node->items[lower_pos].entry_type == 2) {
              results[pos] = {node->items[lower_pos].comp.data.key, node->items[lower_pos].comp.data.value};
              pos++;
            } else {
              pos = range_core_len < true > (results, pos, node->items[lower_pos].comp.child, lower, len);
            }
            if (pos >= len) {
              return pos;
            }
          }
          lower_pos++;
        }
        return pos;
      }
    }

};

}

#endif
