//! Phase 6.6: hardware performance-counter availability probe.
//!
//! The audit's gate list requires "cache misses / CPU counters if the
//! environment allows" — this example answers that question *programmatically*
//! instead of assuming the Phase 6.1 finding (perf_event_open = EPERM in this
//! KVM guest) still holds. It calls perf_event_open(2) directly through a
//! raw syscall (same no-libc pattern as the MADV_DONTNEED reclaim) for
//! hardware cycles/instructions/cache events and one software event, trying
//! several attr sizes for forward/backward layout compatibility.
//!
//! Run: cargo run --release --example perfcheck --manifest-path crates/llama-kernels/Cargo.toml

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn sys_perf_open(ty: u64, cfg: u64, sz: usize) -> Result<i64, i64> {
    use std::arch::asm;
    // zeroed 120-byte perf_event_attr; fields we set: type (u32), size (u32),
    // config (u64). The kernel zero-pads when size < its own struct size.
    let mut attr = [0u64; 15];
    attr[0] = ty | ((sz as u64) << 32);
    attr[1] = cfg;
    let ret: i64;
    unsafe {
        asm!(
            "syscall",
            inlateout("rax") 298i64 => ret, // __NR_perf_event_open
            in("rdi") attr.as_ptr(),
            in("rsi") 0i64,   // pid = current task
            in("rdx") -1i64,  // cpu = task context
            in("r10") -1i64,  // group_fd
            in("r8") 0i64,    // flags
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
    }
    if ret < 0 { Err(-ret) } else { Ok(ret) }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn sys_close(fd: i64) {
    use std::arch::asm;
    unsafe {
        asm!("syscall",
            in("rax") 3i64, // __NR_close
            in("rdi") fd,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack));
    }
}

fn errno_name(e: i64) -> String {
    match e {
        1 => "EPERM".into(),
        13 => "EACCES".into(),
        22 => "EINVAL".into(),
        38 => "ENOSYS".into(),
        _ => format!("errno {e}"),
    }
}

fn main() {
    println!("perf_event_open availability probe (raw syscall):");
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        // (name, type, config): PERF_TYPE_HARDWARE=0, PERF_TYPE_SOFTWARE=1
        let events: &[(&str, u64, u64)] = &[
            ("hw cpu-cycles", 0, 0),
            ("hw instructions", 0, 1),
            ("hw cache-references", 0, 2),
            ("hw cache-misses", 0, 3),
            ("hw branch-misses", 0, 5),
            ("sw cpu-clock", 1, 0),
        ];
        let mut any_open = false;
        for (name, ty, cfg) in events {
            let mut verdict = "no size accepted".to_string();
            for &sz in &[56usize, 64, 72, 80, 96, 120] {
                match sys_perf_open(*ty, *cfg, sz) {
                    Ok(fd) => {
                        verdict = format!("OPEN fd={fd} (attr size {sz})");
                        sys_close(fd);
                        any_open = true;
                        break;
                    }
                    Err(e) => verdict = errno_name(e),
                }
            }
            println!("  {name:<20} -> {verdict}");
        }
        if any_open {
            println!("VERDICT: hardware counters available on this guest.");
            println!("#audit perf_hw_counters=available");
        } else {
            println!("VERDICT: perf_event_open blocked (EPERM/EACCES) for every hardware and");
            println!("software event — the audit stays on TSC timing + bandwidth arithmetic,");
            println!("as documented in PERFORMANCE_REPORT.md section 1.2.");
            println!("#audit perf_hw_counters=blocked");
        }
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        println!("  probe not implemented for this platform");
        println!("#audit perf_hw_counters=not-probed");
    }
}
