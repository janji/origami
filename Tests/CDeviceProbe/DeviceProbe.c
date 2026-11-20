#include "CDeviceProbe.h"
#include <dlfcn.h>
#include <stdatomic.h>

// Defined in the executable, this symbol takes the place of Metal's for every caller statically linked into
// engine-checks, the engine included. It counts the request and forwards to Metal's own function, whose (retained)
// result it returns unchanged. Metal's header is not included so the two declarations cannot clash.
void *MTLCreateSystemDefaultDevice(void);

static _Atomic long requests;
static _Atomic int hidden;
static void *(*_Atomic metal)(void);

void *MTLCreateSystemDefaultDevice(void) {
    atomic_fetch_add(&requests, 1);
    if (atomic_load(&hidden)) return 0;
    void *(*create)(void) = atomic_load(&metal);
    if (!create) {
        create = (void *(*)(void))dlsym(RTLD_NEXT, "MTLCreateSystemDefaultDevice");
        atomic_store(&metal, create);
    }
    return create ? create() : 0;
}

long origami_metal_device_requests(void) { return atomic_load(&requests); }

void origami_hide_metal_device(void) { atomic_store(&hidden, 1); }
