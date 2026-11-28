#ifndef ORIGAMI_EXACT_IO_H
#define ORIGAMI_EXACT_IO_H
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

// Exact positional file I/O and process/VM probes for the Swift engine. Every call reports failure as a
// return value (0 or an errno / kern_return_t), never through a stale errno.

// Result of an exact read: the bytes copied before any error, and 0 or the errno that stopped it.
typedef struct { size_t bytes_read; int error_code; } origami_read_result;
// fstat snapshot that pins one regular file: device, inode, size and modification/change stamps.
typedef struct {
    uint64_t device, inode, size;
    int64_t modification_seconds, modification_nanoseconds;
    int64_t change_seconds, change_nanoseconds;
} origami_file_identity;
// pread-shaped reader (fd, buffer, length, offset, context) that fault tests inject.
typedef ssize_t (*origami_read_fn)(int, void *, size_t, off_t, void *);

// Reads exactly `length` bytes at `offset` in chunks of at most INT_MAX, retrying EINTR; EOF or a short
// read is EIO, so an incomplete record is never published. Errors are returned explicitly, never inferred
// from errno after successful I/O. A chunk whose buffer or offset is not 4 KiB-aligned is at most 1 MiB, so an
// uncached (F_NOCACHE) read leaves no pages in the file cache.
#define ORIGAMI_UNALIGNED_PIECE_ALIGN 4096
#define ORIGAMI_UNALIGNED_READ_PIECE (1u << 20)
origami_read_result origami_pread_exact(int fd, void *buffer, size_t length, int64_t offset);
// Injectable reader for deterministic short-read, EOF and EINTR fault tests.
origami_read_result origami_pread_exact_with(int fd, void *buffer, size_t length,
                                   int64_t offset, origami_read_fn reader, void *context);
// open(O_RDONLY | O_CLOEXEC), following symlinks; `bypass_cache` sets F_NOCACHE. Returns the fd, or -1
// with *error_code set.
int origami_open_readonly(const char *path, int bypass_cache, int *error_code);
// fstat into *identity; 0, errno, or EINVAL for anything but a regular file.
int origami_get_identity(int fd, origami_file_identity *identity);
// APFS clone only: return errno explicitly, never copy as a fallback.
int origami_clone_to(int source_fd, int directory_fd, const char *name);
// Stable filesystem UUID for persisted identities across mount/device renumbering.
// Writes the canonical uppercase UUID plus NUL (37 bytes); errno-style failure.
int origami_get_volume_uuid(int fd, char *output, size_t capacity);
// This process's phys_footprint (TASK_VM_INFO), the figure memory pressure is judged by; 0 if unavailable.
uint64_t origami_physical_footprint(void);
// Estimated reclaimable free, purgeable and file-backed bytes, counting overlapping speculative pages
// only once. Inactive anonymous pages are excluded. File pages may require writeback; this is not a
// promise of immediately free RAM. 0 if unavailable or the sample cannot be represented safely.
uint64_t origami_available_memory(void);
// Pure conversion of HOST_VM_INFO64 counters for deterministic accounting checks. free_pages and
// external_pages each already contain speculative_pages. Invalid arithmetic/page size returns 0.
uint64_t origami_available_memory_from_pages(uint64_t free_pages, uint64_t speculative_pages,
                                            uint64_t purgeable_pages, uint64_t external_pages,
                                            uint64_t page_size);
/// Cumulative bytes the system swapped out since boot.
uint64_t origami_swapout_bytes(void);
// Process task_events_info counters (each a 32-bit kernel counter widened). 0 or a kern_return_t.
int origami_task_events(uint64_t *faults, uint64_t *pageins, uint64_t *cow_faults);
// One volatile write of a byte's own value per host page the range touches: faults each page in, dirty,
// without changing a byte (the expert cache's path when slabs are not wired or wiring fails). Does not wire
// memory or promise simultaneous residency. Caller joins all users.
int origami_touch_preserving_pages(void *buffer, size_t length, size_t page_size,
                               uint64_t *pages_touched);
// User-wire (mlock) a page-aligned whole-page range so the VM cannot compress or
// reclaim it; faults any non-resident page in without changing its bytes.
// Returns 0 or errno (EINVAL for an unaligned range; EAGAIN/ENOMEM at the wire limit).
int origami_wire_pages(void *buffer, size_t length, size_t page_size, uint64_t *pages_wired);
// Undo origami_wire_pages for the same range. Returns 0 or errno.
int origami_unwire_pages(void *buffer, size_t length);
// CPU publication to a shared Metal atomic flag. The caller owns aligned storage.
void origami_atomic_store_u32(void *address, uint32_t value);
// Single-precision BLAS through the current CBLAS interface (Package.swift defines ACCELERATE_NEW_LAPACK for this
// target; 32-bit indices). The same Accelerate kernels as the deprecated classic entry points, bit for bit.
// y[rows][outputs] = x[rows][width] . w[outputs][width]^T, all row-major.
void origami_sgemm_nt(int rows, int outputs, int width, const float *x, const float *w, float *y);
// y[rows] = a[rows][cols] . x[cols], row-major.
void origami_sgemv(int rows, int cols, const float *a, const float *x, float *y);
#endif
