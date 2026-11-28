#include "CExactIO.h"
#include <Accelerate/Accelerate.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/attr.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/clonefile.h>
#include <uuid/uuid.h>
#include <unistd.h>
#include <mach/mach.h>

int origami_clone_to(int source_fd, int directory_fd, const char *name) {
    if (!name || !*name || strchr(name, '/')) return EINVAL;
    return fclonefileat(source_fd, directory_fd, name, CLONE_NOFOLLOW) == 0 ? 0 : errno;
}

static ssize_t system_read(int fd, void *buffer, size_t size, off_t offset, void *context) {
    (void)context;
    return pread(fd, buffer, size, offset);
}

origami_read_result origami_pread_exact_with(int fd, void *buffer, size_t length,
                                   int64_t offset, origami_read_fn reader, void *context) {
    origami_read_result result = {0, 0};
    if (offset < 0 || !reader || (!buffer && length) || length > (uint64_t)(INT64_MAX - offset)) {
        result.error_code = EINVAL;
        return result;
    }
    while (result.bytes_read < length) {
        size_t remaining = length - result.bytes_read;
        // Darwin rejects read requests greater than INT_MAX.
        size_t chunk = remaining > INT_MAX ? INT_MAX : remaining;
        // An F_NOCACHE read whose buffer or offset is not 4 KiB-aligned goes through the file cache, and Darwin drops
        // only its first 1 MiB afterwards: the rest stays cached (a Qwen GGUF expert record read into a cache slot at
        // `index * recordBytes`). Reads of at most 1 MiB leave nothing behind; aligned reads stay whole (direct).
        if ((((uintptr_t)buffer + result.bytes_read) | (uint64_t)(offset + result.bytes_read)) & (ORIGAMI_UNALIGNED_PIECE_ALIGN - 1))
            chunk = chunk > ORIGAMI_UNALIGNED_READ_PIECE ? ORIGAMI_UNALIGNED_READ_PIECE : chunk;
        ssize_t count = reader(fd, (char *)buffer + result.bytes_read, chunk,
                               (off_t)(offset + result.bytes_read), context);
        if (count < 0) {
            int code = errno;
            if (code == EINTR) continue;
            result.error_code = code ? code : EIO;
            return result;
        }
        if (count == 0 || (size_t)count > chunk) {
            result.error_code = EIO; // An incomplete record must never be published.
            return result;
        }
        result.bytes_read += (size_t)count;
    }
    return result;
}

origami_read_result origami_pread_exact(int fd, void *buffer, size_t length, int64_t offset) {
    return origami_pread_exact_with(fd, buffer, length, offset, system_read, NULL);
}

int origami_open_readonly(const char *path, int bypass_cache, int *error_code) {
    if (!path || !error_code) { errno = EINVAL; return -1; }
    *error_code = 0;
    int fd;
    do { fd = open(path, O_RDONLY | O_CLOEXEC); } while (fd < 0 && errno == EINTR);
    if (fd < 0) { *error_code = errno; return -1; }
    if (bypass_cache && fcntl(fd, F_NOCACHE, 1) < 0) {
        *error_code = errno;
        close(fd);
        return -1;
    }
    return fd;
}

int origami_get_identity(int fd, origami_file_identity *identity) {
    if (!identity) return EINVAL;
    struct stat st;
    if (fstat(fd, &st) < 0) return errno;
    if (!S_ISREG(st.st_mode) || st.st_size < 0) return EINVAL;
    *identity = (origami_file_identity){(uint64_t)st.st_dev, st.st_ino, (uint64_t)st.st_size,
        st.st_mtimespec.tv_sec, st.st_mtimespec.tv_nsec,
        st.st_ctimespec.tv_sec, st.st_ctimespec.tv_nsec};
    return 0;
}

int origami_get_volume_uuid(int fd, char *output, size_t capacity) {
    if (!output || capacity < 37) return EINVAL;
    output[0] = '\0';
    struct stat file_stat, root_stat;
    struct statfs file_fs, root_fs;
    if (fstat(fd, &file_stat) < 0 || fstatfs(fd, &file_fs) < 0) return errno;
    int root;
    do { root = open(file_fs.f_mntonname, O_RDONLY | O_DIRECTORY | O_CLOEXEC); }
    while (root < 0 && errno == EINTR);
    if (root < 0) return errno;
    int code = 0;
    if (fstat(root, &root_stat) < 0 || fstatfs(root, &root_fs) < 0) code = errno;
    else if (root_stat.st_dev != file_stat.st_dev ||
             memcmp(&file_fs.f_fsid, &root_fs.f_fsid, sizeof(fsid_t)) != 0) code = ESTALE;
    struct attrlist attributes = {0};
    attributes.bitmapcount = ATTR_BIT_MAP_COUNT;
    attributes.volattr = ATTR_VOL_INFO | ATTR_VOL_UUID;
    struct { uint32_t length; uuid_t uuid; } value = {0};
    if (!code && fgetattrlist(root, &attributes, &value, sizeof(value), 0) < 0) code = errno;
    if (!code && (value.length != sizeof(value) || uuid_is_null(value.uuid))) code = EIO;
    if (!code) uuid_unparse_upper(value.uuid, output);
    close(root);
    return code;
}

uint64_t origami_physical_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
    return info.phys_footprint;
}
int origami_task_events(uint64_t *faults, uint64_t *pageins, uint64_t *cow_faults) {
    if (!faults || !pageins || !cow_faults) return EINVAL;
    task_events_info_data_t info;
    mach_msg_type_number_t count = TASK_EVENTS_INFO_COUNT;
    kern_return_t result = task_info(mach_task_self(), TASK_EVENTS_INFO, (task_info_t)&info, &count);
    if (result != KERN_SUCCESS) return (int)result;
    *faults = (uint32_t)info.faults; *pageins = (uint32_t)info.pageins; *cow_faults = (uint32_t)info.cow_faults;
    return 0;
}
uint64_t origami_available_memory_from_pages(uint64_t free_pages, uint64_t speculative_pages,
                                            uint64_t purgeable_pages, uint64_t external_pages,
                                            uint64_t page_size) {
    // XNU host.c includes speculative pages in free_count; vm_page_speculate() in
    // vm_resident.c includes them in pageable_external_count too. Count that overlap once.
    // https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/host.c
    // https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_resident.c
    // An invalid sample must never become a huge admission budget through unsigned wrap.
    if (!page_size || (page_size & (page_size - 1)) || speculative_pages > free_pages) return 0;
    uint64_t pages = free_pages - speculative_pages;
    if (external_pages > UINT64_MAX - pages) return 0;
    pages += external_pages;
    if (purgeable_pages > UINT64_MAX - pages) return 0;
    pages += purgeable_pages;
    if (pages > UINT64_MAX / page_size) return 0;
    return pages * page_size;
}

uint64_t origami_available_memory(void) {
    // Estimate reclaimable memory from free, purgeable and pageable file-backed pages.
    // Speculative file pages overlap free_count and external_page_count. Anonymous
    // inactive pages are excluded; file-backed pages may still require writeback/reclaim.
    vm_statistics64_data_t info = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    kern_return_t page_result = host_page_size(host, &page_size);
    kern_return_t result = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&info, &count);
    mach_port_deallocate(mach_task_self(), host);
    if (page_result != KERN_SUCCESS || result != KERN_SUCCESS || count < HOST_VM_INFO64_REV1_COUNT) return 0;
    return origami_available_memory_from_pages(info.free_count, info.speculative_count,
                                              info.purgeable_count, info.external_page_count, page_size);
}

uint64_t origami_swapout_bytes(void) {
    vm_statistics64_data_t info;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    kern_return_t page_result = host_page_size(host, &page_size);
    kern_return_t result = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&info, &count);
    mach_port_deallocate(mach_task_self(), host);
    if (page_result != KERN_SUCCESS || result != KERN_SUCCESS) return 0;
    return (uint64_t)info.swapouts * page_size;
}

void origami_atomic_store_u32(void *address, uint32_t value) {
    atomic_store_explicit((_Atomic uint32_t *)address, value, memory_order_release);
}

int origami_touch_preserving_pages(void *buffer, size_t length, size_t page_size,
                               uint64_t *pages_touched) {
    if (pages_touched) *pages_touched = 0;
    if (!buffer || !length || !pages_touched || !page_size ||
        (page_size & (page_size - 1)) || length > UINTPTR_MAX - (uintptr_t)buffer)
        return EINVAL;
    volatile unsigned char *bytes = buffer;
    size_t offset = 0;
    while (offset < length) {
        unsigned char original = bytes[offset];
        bytes[offset] = original;
        if (bytes[offset] != original) return EIO;
        ++*pages_touched;
        size_t advance = page_size - (((uintptr_t)buffer + offset) & (page_size - 1));
        if (advance >= length - offset) break;
        offset += advance;
    }
    return 0;
}

int origami_wire_pages(void *buffer, size_t length, size_t page_size, uint64_t *pages_wired) {
    if (pages_wired) *pages_wired = 0;
    if (!buffer || !length || !pages_wired || !page_size || (page_size & (page_size - 1)) ||
        ((uintptr_t)buffer & (page_size - 1)) || (length & (page_size - 1)) ||
        length > UINTPTR_MAX - (uintptr_t)buffer)
        return EINVAL;
    if (mlock(buffer, length) != 0) return errno ? errno : EIO;
    *pages_wired = length / page_size;
    return 0;
}

int origami_unwire_pages(void *buffer, size_t length) {
    if (!buffer || !length) return EINVAL;
    return munlock(buffer, length) != 0 ? (errno ? errno : EIO) : 0;
}

void origami_sgemm_nt(int rows, int outputs, int width, const float *x, const float *w, float *y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, outputs, width, 1, x, width, w, width, 0, y, outputs);
}

void origami_sgemv(int rows, int cols, const float *a, const float *x, float *y) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1, a, cols, x, 1, 0, y, 1);
}
