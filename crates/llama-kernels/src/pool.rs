//! Minimal persistent worker pool for matvec parallelism.
//!
//! Why: `std::thread::scope` costs ~32 µs per call on this machine; the forward
//! pass issues ~150 matvecs per token, so spawn overhead alone would eat ~5 ms
//! per token (~5% of a 10 tok/s budget). llama.cpp solves this with its own
//! persistent thread pool (`ggml_threadpool`); this is the Rust equivalent.
//!
//! Design:
//! - workers are spawned once (lazily) and park on a condvar between jobs;
//! - `par_chunks` enqueues `n_chunks - 1` jobs and runs chunk 0 on the caller
//!   thread (the caller participates, exactly like ggml's `ith == 0` thread);
//! - jobs borrow caller stack data; this is sound because `par_chunks` does
//!   not return until every enqueued job has finished (the same lifetime
//!   guarantee `std::thread::scope` provides);
//! - reentrant calls from worker threads fall back to sequential execution.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct PoolState {
    queue: VecDeque<Job>,
    active: usize,
    shutdown: bool,
}

/// Signals new work without taking the state mutex (spin-then-park).
struct PoolShared {
    state: Mutex<PoolState>,
    work_cv: Condvar, // workers wait for new jobs / shutdown
    done_cv: Condvar, // caller waits until the queue is drained and active == 0
    pending: AtomicUsize, // jobs queued but not yet popped (spin path)
}

static POOL: OnceLock<&'static PoolShared> = OnceLock::new();

fn available_parallelism() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn shared() -> &'static PoolShared {
    *POOL.get_or_init(|| {
        let n_workers = available_parallelism().saturating_sub(1);
        let sh = PoolShared {
            state: Mutex::new(PoolState { queue: VecDeque::new(), active: 0, shutdown: false }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
            pending: AtomicUsize::new(0),
        };
        let arc = Arc::new(sh);
        for _ in 0..n_workers {
            let arc = arc.clone();
            std::thread::Builder::new()
                .name("llama-rs-worker".into())
                .spawn(move || worker_loop(arc))
                .expect("spawn worker");
        }
        // Leak the Arc on purpose: the pool lives for the whole process;
        // the workers hold their own Arc refs, process exit reclaims the rest.
        let r: &'static PoolShared = unsafe { &*Arc::into_raw(arc) };
        r
    })
}

fn worker_loop(arc: Arc<PoolShared>) {
    IN_WORKER.with(|f| f.store(true, Ordering::Relaxed));
    loop {
        // Spin briefly for new work before parking on the condvar — the
        // forward pass issues ~150 small matvecs per token and a condvar
        // wake alone costs 5-20 µs; spinning cuts ramp-up latency.
        let mut spins = 4096u32;
        while spins > 0 {
            if arc.pending.load(Ordering::Relaxed) != 0 {
                break;
            }
            std::hint::spin_loop();
            spins -= 1;
        }
        let job: Option<Job> = {
            let mut st = arc.state.lock().unwrap();
            loop {
                if let Some(job) = st.queue.pop_front() {
                    st.active += 1;
                    arc.pending.fetch_sub(1, Ordering::Relaxed);
                    break Some(job);
                }
                if st.shutdown {
                    break None;
                }
                st = arc.work_cv.wait(st).unwrap();
            }
        };
        match job {
            Some(job) => {
                job();
                let mut st = arc.state.lock().unwrap();
                st.active -= 1;
                if st.queue.is_empty() && st.active == 0 {
                    arc.done_cv.notify_all();
                }
            }
            None => return,
        }
    }
}

thread_local! {
    static IN_WORKER: AtomicBool = AtomicBool::new(false);
}

/// Scoped raw pointer wrapper: lets jobs borrow caller stack data.
/// Sound because `par_chunks` joins all jobs before returning.
#[derive(Clone, Copy)]
struct ScopedFn<'a> {
    f: *const (dyn Fn(usize) + Sync + 'a),
}
unsafe impl Send for ScopedFn<'_> {}
impl<'a> ScopedFn<'a> {
    /// SAFETY: the borrow must stay alive while the job runs — guaranteed by
    /// par_chunks joining all jobs before returning.
    #[inline]
    fn get(self) -> &'a (dyn Fn(usize) + Sync) {
        unsafe { &*self.f }
    }
}

/// Runs `f(i)` for every `i` in `0..n_chunks`, using the persistent worker
/// pool plus the calling thread. `n_threads` is the total participation target
/// (caller + `n_threads - 1` workers).
pub fn par_chunks(n_chunks: usize, n_threads: usize, f: &(dyn Fn(usize) + Sync)) {
    if n_chunks <= 1 || n_threads <= 1 || available_parallelism() < 2 {
        for i in 0..n_chunks {
            f(i);
        }
        return;
    }
    if IN_WORKER.with(|w| w.load(Ordering::Relaxed)) {
        // reentrant call from a worker: run sequentially to avoid deadlock
        for i in 0..n_chunks {
            f(i);
        }
        return;
    }
    let sh = shared();
    let scoped = ScopedFn { f };
    let mut n_enqueued = 0usize;
    {
        let mut st = sh.state.lock().unwrap();
        for i in 1..n_chunks {
            // Lifetime erasure: ScopedFn<'a> → ScopedFn<'static>.
            // SAFETY: sound because par_chunks waits for all jobs before
            // returning, so the erased borrow is always alive when a job
            // runs (the same guarantee std::thread::scope provides).
            let s_static: ScopedFn<'static> = unsafe { std::mem::transmute(scoped) };
            let job: Job = Box::new(move || {
                let fr = s_static.get();
                fr(i);
            });
            st.queue.push_back(job);
            n_enqueued += 1;
        }
        sh.pending.store(n_enqueued, Ordering::Release);
        sh.work_cv.notify_all();
    }
    // caller runs chunk 0 while workers take the rest
    f(0);
    // wait for the enqueued jobs to finish
    if n_enqueued > 0 {
        let mut st = sh.state.lock().unwrap();
        while !st.queue.is_empty() || st.active > 0 {
            st = sh.done_cv.wait(st).unwrap();
        }
    }
}
