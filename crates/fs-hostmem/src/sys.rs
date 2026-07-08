//! Thin `libc` wrappers around `memfd_create`/`ftruncate`/`pwrite`/`mmap`/`munmap`/`madvise`.
//!
//! This module is the *only* place in the crate (and in the whole workspace — every other crate
//! keeps `forbid(unsafe_code)`) where raw pointers get poked. Linux-only by design (host is
//! Linux); every function returns `io::Error::last_os_error()` on failure so callers get a normal
//! `Result`.

use std::ffi::CString;
use std::io;
use std::os::unix::io::RawFd;

/// Create an anonymous memfd and `ftruncate` it to `size` bytes. `name` is cosmetic (shows up in
/// `/proc/<pid>/maps`); it is not a path and creates no directory entry.
pub fn memfd_create(name: &str, size: usize) -> io::Result<RawFd> {
    let cname = CString::new(name).expect("memfd name must not contain NUL");
    // SAFETY: `cname` is a valid NUL-terminated C string that outlives the call; `memfd_create`
    // with flags=0 either returns a valid owned fd or -1 (checked below).
    let fd = unsafe { libc::memfd_create(cname.as_ptr(), 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fd` was just created above and is open; ftruncate is a plain size-set syscall.
    let rc = unsafe { libc::ftruncate(fd, size as libc::off_t) };
    if rc < 0 {
        let e = io::Error::last_os_error();
        // SAFETY: `fd` is open and owned by us; closing it on the error path leaks nothing.
        unsafe { libc::close(fd) };
        return Err(e);
    }
    Ok(fd)
}

/// Write all of `data` at offset 0 of `fd` (used once, to seed the golden image into the memfd's
/// page cache before anyone maps it).
pub fn pwrite_all(fd: RawFd, data: &[u8]) -> io::Result<()> {
    let mut off = 0usize;
    while off < data.len() {
        // SAFETY: `fd` is a valid open fd (caller-owned); the source slice
        // `data[off..]` is valid for `data.len() - off` bytes for the duration of the call.
        let n = unsafe {
            libc::pwrite(
                fd,
                data[off..].as_ptr() as *const libc::c_void,
                data.len() - off,
                off as libc::off_t,
            )
        };
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "pwrite wrote 0 bytes"));
        }
        off += n as usize;
    }
    Ok(())
}

/// A private (copy-on-write) `PROT_READ|PROT_WRITE` mapping of `fd`'s first `size` bytes. Reads
/// of untouched pages transparently share `fd`'s physical pages; the first write to a page
/// triggers a kernel-managed copy.
pub fn mmap_private(fd: RawFd, size: usize) -> io::Result<*mut u8> {
    // SAFETY: requests an anonymous-address (`addr=NULL`) mapping of a valid, caller-owned `fd`
    // for `size` bytes; the kernel picks the address, or we get back `MAP_FAILED` (checked below).
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE,
            fd,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    Ok(ptr as *mut u8)
}

/// Unmap a region previously returned by [`mmap_private`].
///
/// # Safety
/// `ptr` must be a mapping base previously returned by `mmap_private` with the same `size`, not
/// already unmapped, and the caller must guarantee no other code still holds live references
/// (slices/pointers) into `[ptr, ptr+size)`.
pub unsafe fn munmap(ptr: *mut u8, size: usize) {
    // SAFETY: upheld by this function's own safety contract.
    unsafe {
        libc::munmap(ptr as *mut libc::c_void, size);
    }
}

/// Drop every private (COW-dirtied) page in `[ptr, ptr+size)` back to the mapping's backing file.
/// The *next* access to a dropped page refaults transparently against the golden memfd contents —
/// this is the O(1) "reset to snapshot" primitive being benchmarked against `Mmu::reset_dirty`.
///
/// # Safety
/// `ptr` must be a live mapping base of at least `size` bytes (as returned by `mmap_private`); no
/// other thread may be concurrently reading/writing pages in this range in a way that assumes
/// their prior (dirtied) contents survive the call — `MADV_DONTNEED` on a `MAP_PRIVATE` mapping is
/// destructive by design.
pub unsafe fn madvise_dontneed(ptr: *mut u8, size: usize) -> io::Result<()> {
    // SAFETY: upheld by this function's own safety contract.
    let rc = unsafe { libc::madvise(ptr as *mut libc::c_void, size, libc::MADV_DONTNEED) };
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Close a raw fd (used to release the golden memfds on `Golden` drop).
///
/// # Safety
/// `fd` must be open and not used again afterwards (by this or any other owner).
pub unsafe fn close(fd: RawFd) {
    // SAFETY: upheld by this function's own safety contract.
    unsafe {
        libc::close(fd);
    }
}
