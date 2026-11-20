#ifndef ORIGAMI_DEVICE_PROBE_H
#define ORIGAMI_DEVICE_PROBE_H

// How many times anything linked into engine-checks (the checks or the engine) has asked for the system Metal
// device. The runner compares it around each check to tell a GPU check from a logic check.
long origami_metal_device_requests(void);

// From now on the system Metal device is reported absent (`engine-checks --no-gpu`), as on a host without one: checks
// that need it must skip with a reason, and none may crash.
void origami_hide_metal_device(void);

#endif
