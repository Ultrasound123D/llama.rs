//! Persistent worker pool — Phase 6.2 spin-barrier edition.
//!
//! Why: profiling (Phase 6.1) showed the condvar handoff costs two futex
//! round-trips per matvec (worker wake + caller wake), ~65-130 µs each on
//! this KVM guest — 11-24 ms per decode token at ~170-183 matvecs/token.
//! Both vCPUs have real work at all times (caller band + worker band), so
//! context switches during handoff buy nothing.
//!
//! Design (delta vs the v0.2 pool):
//! - `outstanding` AtomicUsize counts jobs published-but-not-finished;
//!   workers decrement with Release after each job; the caller spins on it
//!   (Acquire + `pause`) instead of parking on `done_cv`.
//! - workers spin on `pending` (as before) but with a much larger,
//!   time-bounded budget before parking; parking remains as fallback for
//!   long idle / descheduling, so shutdown and long jobs stay correct.
//! - the Mutex+queue protocol is unchanged: it still serializes job pickup
//!   (uncontended lock/unlock ≈ 40 ns total).
//! - after a caller-side spin timeout (safety net against a descheduled
//!   worker under an oversubscribed host) the caller falls back to the
//!   condvar wait exactly as before.
//!
//! Soundness of the scoped borrow is unchanged: `par_chunks` returns only
//! after every enqueued job has completed (now observed via `outstanding`
//! plus, on the fallback path, the condvar protocol).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::kstats;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct PoolState {
    queue: VecDeque<Job>,
    shutdown: bool,
}

struct PoolShared {
    state: Mutex<PoolState>,
    work_cv: Condvar, // workers wait for new jobs / shutdown
    done_cv: Condvar, // caller (fallback path) waits until drained
    /// jobs published but not completed (spin barrier counter)
    outstanding: AtomicUsize,
    /// jobs queued but not yet popped (worker spin path)
    pending: AtomicUsize,
}

static POOL: OnceLock<&'static PoolShared> = OnceLock::new();

/// Dispatch serializer (Phase 6.5): the slot protocol and the `outstanding`
/// counter are single-flight by construction — two OS threads dispatching
/// concurrently would overwrite each other's published job. The engine
/// drives the pool from one thread, but tests (cargo-test runs them in
/// parallel) and any future multi-driver use must not corrupt results: a
/// dispatch holds this mutex from publish through wait. Uncontended cost
/// ~40 ns — noise against the ~µs dispatch. NOT reentrant: closures must
/// not dispatch back into the pool from the caller thread (workers' calls
/// are already routed to the sequential fallback below).
static DISPATCH_MUX: Mutex<()> = Mutex::new(());

/// Spin budgets (tuned on the 2-vCPU sapphirerapids guest @3.2 GHz):
/// - worker spin: ~150 µs of pause-loop before parking;
/// - caller spin: ~100 µs of pause-loop before falling back to the condvar.
const WORKER_SPINS: u32 = 1 << 18;
const CALLER_SPINS: u32 = 1 << 17;

fn available_parallelism() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn shared() -> &'static PoolShared {
    *POOL.get_or_init(|| {
        let n_workers = available_parallelism().saturating_sub(1);
        let sh = PoolShared {
            state: Mutex::new(PoolState { queue: VecDeque::new(), shutdown: false }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
            outstanding: AtomicUsize::new(0),
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
    // how this worker's next job pickup was observed: condvar wake vs spin
    let mut came_from_park = false;
    loop {
        // Spin for new work before parking on the condvar. Phase 6.2: the
        // budget is large enough to cover the inter-matvec glue (~10-20 µs)
        // so that in steady decode the worker never parks at all. The
        // par_chunks2 slot is checked first (claim is a single CAS).
        let mut spins = WORKER_SPINS;
        let mut spin_catch = false;
        while spins > 0 {
            if let Some(fat) = slot_try_claim() {
                run_slot_job(fat);
                // restart the spin budget after a completed job
                spins = WORKER_SPINS;
                continue;
            }
            if arc.pending.load(Ordering::Relaxed) != 0 {
                spin_catch = true;
                break;
            }
            std::hint::spin_loop();
            spins -= 1;
        }
        let job: Option<Job> = {
            let mut st = arc.state.lock().unwrap();
            loop {
                if let Some(job) = st.queue.pop_front() {
                    arc.pending.fetch_sub(1, Ordering::Relaxed);
                    break Some(job);
                }
                if st.shutdown {
                    break None;
                }
                // before parking, drain any slot job that was published
                // between the spin phase and this lock acquisition
                if let Some(fat) = slot_try_claim() {
                    drop(st);
                    run_slot_job(fat);
                    came_from_park = false;
                    st = arc.state.lock().unwrap();
                    continue;
                }
                st = arc.work_cv.wait(st).unwrap();
                came_from_park = true; // next pickup is a condvar wake
            }
        };
        match job {
            Some(job) => {
                if kstats::enabled() {
                    let now = kstats::tsc();
                    let enq = kstats::ENQUEUE_TSC.load(Ordering::Relaxed);
                    if now >= enq {
                        kstats::WAKE_TICKS.fetch_add(now - enq, Ordering::Relaxed);
                        kstats::WAKE_JOBS.fetch_add(1, Ordering::Relaxed);
                        if came_from_park || !spin_catch {
                            kstats::PARKS.fetch_add(1, Ordering::Relaxed);
                        } else {
                            kstats::SPIN_HITS.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    came_from_park = false;
                }
                let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
                job();
                if kstats::enabled() {
                    kstats::JOB_TICKS.fetch_add(kstats::tsc().wrapping_sub(t0), Ordering::Relaxed);
                }
                // publish completion: the caller's spin barrier observes this
                arc.outstanding.fetch_sub(1, Ordering::Release);
                // legacy path: wake a parked caller, if any
                let st = arc.state.lock().unwrap();
                if st.queue.is_empty() && arc.outstanding.load(Ordering::Relaxed) == 0 {
                    arc.done_cv.notify_all();
                }
            }
            None => return,
        }
    }
}

/// Run a claimed par_chunks2 job. The closure is invoked once through its
/// FnMut vtable; the caller's stack slot stays alive until `done == g`
/// (the caller waits in par_chunks2), so the borrow is sound.
#[cold]
fn run_slot_job((data, vt, g): (*mut (), *mut (), u64)) {
    let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    unsafe {
        let fat: *mut dyn FnMut() =
            std::mem::transmute::<(*mut (), *mut ()), *mut dyn FnMut()>((data, vt));
        (*fat)();
    }
    SLOT.done.store(g, Ordering::Release);
    if kstats::enabled() {
        kstats::SLOT_JOBS.fetch_add(1, Ordering::Relaxed);
        kstats::JOB_TICKS.fetch_add(kstats::tsc().wrapping_sub(t0), Ordering::Relaxed);
    }
}

thread_local! {
    static IN_WORKER: AtomicBool = AtomicBool::new(false);
}

// ---------- Phase 6.2 fast path: lock-free two-closure dispatch ----------
//
// par_chunks2(caller_job, worker_job): the caller runs `caller_job` inline
// while ONE worker runs `worker_job` through a single lock-free job slot.
// Protocol (generations make ABA impossible because the caller waits for
// `done == g` before publishing again):
//   publish:  job.store(ptr, Relaxed); g = gen.fetch_add(1, Release) + 1
//   claim:    CAS claimed: seen -> g (Acq/Rel) — exactly one worker wins
//   complete: done.store(g, Release)
//   caller:   after caller_job, spin until done.load(Acquire) >= g

struct JobSlot {
    /// fat-pointer halves of `*mut dyn FnMut()` published by the caller
    job_data: AtomicPtr<()>,
    job_vt: AtomicPtr<()>,
    gen: AtomicU64,
    claimed: AtomicU64,
    done: AtomicU64,
}

static SLOT: JobSlot = JobSlot {
    job_data: AtomicPtr::new(std::ptr::null_mut()),
    job_vt: AtomicPtr::new(std::ptr::null_mut()),
    gen: AtomicU64::new(0),
    claimed: AtomicU64::new(0),
    done: AtomicU64::new(0),
};

/// Try to claim the published slot job; returns the fat pointer if won.
#[inline]
fn slot_try_claim() -> Option<(*mut (), *mut (), u64)> {
    let g = SLOT.gen.load(Ordering::Acquire);
    let claimed = SLOT.claimed.load(Ordering::Acquire);
    if g <= claimed {
        return None;
    }
    if SLOT
        .claimed
        .compare_exchange(claimed, g, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok()
    {
        // ordered by the Acquire load of `gen` pairing the Release fetch_add
        Some((SLOT.job_data.load(Ordering::Relaxed), SLOT.job_vt.load(Ordering::Relaxed), g))
    } else {
        None
    }
}

/// Runs `a` on the caller thread and `b` on one worker; returns after both
/// completed. Sequential fallback when no pool / called from a worker.
/// `LLAMA_RS_SLOT=0` disables the fast slot path (A/B testing knob).
pub fn par_chunks2<A: FnOnce(), B: FnMut()>(a: A, mut b: B) {
    if available_parallelism() < 2 || num_workers() == 0 {
        a();
        b();
        return;
    }
    if IN_WORKER.with(|w| w.load(Ordering::Relaxed)) {
        a();
        b();
        return;
    }
    // single-flight across ALL driver threads (see DISPATCH_MUX above)
    let _dispatch_guard = DISPATCH_MUX.lock().unwrap();
    if !slot_enabled() {
        // legacy queue path (same semantics, measurable A/B).
        // Lifetime erasure on the trait-object fat pointer — sound because
        // this function does not return before the job has completed.
        let sh = shared();
        let fat: *mut dyn FnMut() = &mut b;
        let fat_static: *mut (dyn FnMut() + 'static) =
            unsafe { std::mem::transmute(fat) };
        // SAFETY (Send): the raw pointer only crosses threads inside the
        // job; the pointed-to closure is invoked on one worker at a time and
        // stays alive until this call joins (below).
        struct SendPtr<T: ?Sized>(*mut T);
        unsafe impl<T: ?Sized> Send for SendPtr<T> {}
        #[inline]
        fn take_ptr<T: ?Sized>(s: SendPtr<T>) -> *mut T {
            s.0
        }
        let send_fat = SendPtr(fat_static);
        let job: Job = Box::new(move || unsafe {
            let fat: *mut (dyn FnMut() + 'static) = take_ptr(send_fat);
            (*fat)();
        });
        {
            let mut st = sh.state.lock().unwrap();
            st.queue.push_back(job);
            sh.outstanding.fetch_add(1, Ordering::Release);
            sh.pending.store(1, Ordering::Release);
            sh.work_cv.notify_all();
        }
        if kstats::enabled() {
            kstats::ENQUEUE_TSC.store(kstats::tsc(), Ordering::Relaxed);
        }
        a();
        // spin barrier + condvar fallback (identical to par_chunks)
        let mut spins = CALLER_SPINS;
        while sh.outstanding.load(Ordering::Acquire) != 0 {
            if spins == 0 {
                let mut st = sh.state.lock().unwrap();
                while sh.outstanding.load(Ordering::Acquire) != 0 {
                    st = sh.done_cv.wait(st).unwrap();
                }
                break;
            }
            std::hint::spin_loop();
            spins -= 1;
        }
        return;
    }
    let sh = shared();
    debug_assert_eq!(sh.outstanding.load(Ordering::Relaxed), 0, "par_chunks2 overlaps another dispatch");

    // publish b (stack slot lives until the caller's wait below completes)
    let fat: *mut dyn FnMut() = &mut b;
    let (job_data, job_vt) =
        unsafe { std::mem::transmute::<*mut dyn FnMut(), (*mut (), *mut ())>(fat) };
    SLOT.job_data.store(job_data, Ordering::Relaxed);
    SLOT.job_vt.store(job_vt, Ordering::Relaxed);
    let g = SLOT.gen.fetch_add(1, Ordering::Release) + 1;
    // wake any parked worker so it can claim the slot
    {
        let _st = sh.state.lock().unwrap();
        sh.work_cv.notify_all();
    }
    // caller runs a
    a();
    // wait for done >= g: spin, then yield-loop (robust if the worker got
    // descheduled by the hypervisor — no futex, no lost wakeup)
    let tw0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    let mut spins = CALLER_SPINS;
    loop {
        if SLOT.done.load(Ordering::Acquire) >= g {
            if kstats::enabled() {
                kstats::SLOT_WAIT_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), Ordering::Relaxed);
            }
            return;
        }
        if spins > 0 {
            std::hint::spin_loop();
            spins -= 1;
        } else {
            std::thread::yield_now();
        }
    }
}

fn slot_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| match std::env::var("LLAMA_RS_SLOT") {
        Ok(v) if v == "0" => false,
        _ => true,
    })
}

fn num_workers() -> usize {
    // set once at pool init; workers are never removed before shutdown
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| available_parallelism().saturating_sub(1))
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
    // single-flight across all driver threads (same serializer as par_chunks2)
    let _dispatch_guard = DISPATCH_MUX.lock().unwrap();
    let sh = shared();
    let scoped = ScopedFn { f };
    let prof = kstats::enabled();
    let te0 = if prof { kstats::tsc() } else { 0 };
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
        sh.outstanding.fetch_add(n_enqueued, Ordering::Release);
        sh.pending.store(n_enqueued, Ordering::Release);
        sh.work_cv.notify_all();
    }
    if prof {
        // publish the enqueue-release timestamp for worker wakeup latency
        kstats::POOL_ENQ_TICKS.fetch_add(kstats::tsc().wrapping_sub(te0), Ordering::Relaxed);
        kstats::ENQUEUE_TSC.store(kstats::tsc(), Ordering::Relaxed);
    }
    // caller runs chunk 0 while workers take the rest
    f(0);
    // wait for the enqueued jobs to finish: spin barrier first (Phase 6.2 —
    // no futex round-trip, no context switch), condvar fallback after a
    // bounded spin (robust if a worker got descheduled by the hypervisor).
    if n_enqueued > 0 {
        let tw0 = if prof { kstats::tsc() } else { 0 };
        let mut spins = CALLER_SPINS;
        let mut parked = false;
        while sh.outstanding.load(Ordering::Acquire) != 0 {
            if spins == 0 {
                // fallback: park on the condvar (state lock + done_cv)
                let mut st = sh.state.lock().unwrap();
                while sh.outstanding.load(Ordering::Acquire) != 0 {
                    st = sh.done_cv.wait(st).unwrap();
                }
                parked = true;
                break;
            }
            std::hint::spin_loop();
            spins -= 1;
        }
        if prof {
            kstats::POOL_WAIT_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), Ordering::Relaxed);
            if parked {
                kstats::CALLER_PARKS.fetch_add(1, Ordering::Relaxed);
            } else {
                kstats::CALLER_SPIN_WINS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
