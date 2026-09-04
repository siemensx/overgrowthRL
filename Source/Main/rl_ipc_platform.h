//-----------------------------------------------------------------------------
//           Name: rl_ipc_platform.h
//      Developer: Wolfire Games LLC / Bad Bunny RL
//         Author: OGRL
//    Description: Platform abstraction for the named shared memory and named
//                 semaphores used by RLShmTransport to exchange observations
//                 and actions with the Python trainer.
//
//                 macOS/Linux use POSIX shm_open/mmap and sem_open. Windows
//                 has neither, so it uses CreateFileMapping/MapViewOfFile and
//                 CreateSemaphore. The wire format in the mapped region is
//                 byte-identical on both platforms; only the naming and the
//                 handle types differ.
//
//                 Naming: callers pass POSIX-style names beginning with '/'
//                 (e.g. "/ogrl_vec0"). On Windows the leading '/' is dropped
//                 and the name is placed in the "Local\" session namespace,
//                 which is what a non-elevated trainer process can open.
//-----------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <string>

namespace RLIpc {

#if defined(_WIN32)
typedef void* SemHandle;   // HANDLE
const SemHandle kInvalidSem = nullptr;
#else
typedef void* SemHandle;   // sem_t* (opaque here so callers stay platform-free)
extern const SemHandle kInvalidSem;  // SEM_FAILED
#endif

// A mapped shared-memory region plus whatever the platform needs to release it.
struct ShmRegion {
    void* addr = nullptr;
    size_t size = 0;
    int fd = -1;             // POSIX only
    void* mapping = nullptr; // Windows only (HANDLE from CreateFileMapping)
};

// Remove any stale segment left by a crashed run, then create a fresh
// zero-length-initialised region of `size` bytes and map it read/write.
// Returns false and leaves *out untouched on failure; the reason is printed
// to stderr by the implementation.
bool CreateShm(const std::string& name, size_t size, ShmRegion* out);

// Unmap and close. Safe to call on a zeroed/already-released region.
void CloseShm(ShmRegion* region);

// Remove the name from the system namespace. No-op on Windows, where the
// kernel refcounts the object and reclaims it when the last handle closes.
void UnlinkShm(const std::string& name);

// Create a named counting semaphore with initial value 0, replacing any stale
// one from a prior crashed run so a leftover posted count cannot desync a new
// run. Returns kInvalidSem on failure.
SemHandle CreateSem(const std::string& name);

void WaitSem(SemHandle sem);
void PostSem(SemHandle sem);
void CloseSem(SemHandle sem);
void UnlinkSem(const std::string& name);

}  // namespace RLIpc
