//-----------------------------------------------------------------------------
//           Name: rl_ipc_platform.cpp
//      Developer: Wolfire Games LLC / Bad Bunny RL
//         Author: OGRL
//    Description: See rl_ipc_platform.h.
//-----------------------------------------------------------------------------
#include "rl_ipc_platform.h"

#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace RLIpc {

#if defined(_WIN32)

namespace {
// "/ogrl_vec0" -> L"Local\ogrl_vec0". Windows object names may not contain a
// backslash outside the namespace prefix, so only the leading POSIX '/' is
// stripped; the rest of the name is already restricted to the same characters
// the POSIX side uses.
std::wstring ToObjectName(const std::string& name) {
    const char* p = name.c_str();
    if (*p == '/') ++p;
    std::wstring w = L"Local\\";
    while (*p) w.push_back(static_cast<wchar_t>(static_cast<unsigned char>(*p++)));
    return w;
}

void ReportLastError(const char* what, const std::string& name) {
    std::fprintf(stderr, "RLIpc: %s(%s) failed: win32 error %lu\n",
                 what, name.c_str(), static_cast<unsigned long>(GetLastError()));
}
}  // namespace

bool CreateShm(const std::string& name, size_t size, ShmRegion* out) {
    const std::wstring obj = ToObjectName(name);
    HANDLE mapping = CreateFileMappingW(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        static_cast<DWORD>((static_cast<unsigned long long>(size) >> 32) & 0xFFFFFFFFull),
        static_cast<DWORD>(size & 0xFFFFFFFFull), obj.c_str());
    if (mapping == nullptr) {
        ReportLastError("CreateFileMapping", name);
        return false;
    }
    void* addr = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (addr == nullptr) {
        ReportLastError("MapViewOfFile", name);
        CloseHandle(mapping);
        return false;
    }
    // A fresh POSIX segment is zero-filled; an existing Windows mapping is not
    // guaranteed to be, so match the POSIX contract explicitly.
    std::memset(addr, 0, size);
    out->addr = addr;
    out->size = size;
    out->fd = -1;
    out->mapping = mapping;
    return true;
}

void CloseShm(ShmRegion* region) {
    if (region->addr) {
        UnmapViewOfFile(region->addr);
        region->addr = nullptr;
    }
    if (region->mapping) {
        CloseHandle(static_cast<HANDLE>(region->mapping));
        region->mapping = nullptr;
    }
    region->size = 0;
}

void UnlinkShm(const std::string&) {
    // Windows reclaims the section object when the last handle closes.
}

SemHandle CreateSem(const std::string& name) {
    const std::wstring obj = ToObjectName(name);
    // Unlike POSIX there is no unlink; opening an existing semaphore would
    // inherit its count, so create exclusively and fail loudly if a previous
    // run still holds one.
    HANDLE h = CreateSemaphoreW(nullptr, 0, LONG_MAX, obj.c_str());
    if (h != nullptr && GetLastError() == ERROR_ALREADY_EXISTS) {
        // Drain any stale posted count so a crashed run cannot desync this one.
        while (WaitForSingleObject(h, 0) == WAIT_OBJECT_0) {
        }
    }
    if (h == nullptr) {
        ReportLastError("CreateSemaphore", name);
        return kInvalidSem;
    }
    return h;
}

void WaitSem(SemHandle sem) {
    if (sem) WaitForSingleObject(static_cast<HANDLE>(sem), INFINITE);
}

void PostSem(SemHandle sem) {
    if (sem) ReleaseSemaphore(static_cast<HANDLE>(sem), 1, nullptr);
}

void CloseSem(SemHandle sem) {
    if (sem) CloseHandle(static_cast<HANDLE>(sem));
}

void UnlinkSem(const std::string&) {
    // See UnlinkShm.
}

#else  // POSIX

const SemHandle kInvalidSem = reinterpret_cast<SemHandle>(SEM_FAILED);

bool CreateShm(const std::string& name, size_t size, ShmRegion* out) {
    shm_unlink(name.c_str());  // best-effort: clear a stale segment from a prior crashed run
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        std::fprintf(stderr, "RLIpc: shm_open(%s) failed: %s\n", name.c_str(), strerror(errno));
        return false;
    }
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        std::fprintf(stderr, "RLIpc: ftruncate failed: %s\n", strerror(errno));
        close(fd);
        return false;
    }
    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        std::fprintf(stderr, "RLIpc: mmap failed: %s\n", strerror(errno));
        close(fd);
        return false;
    }
    out->addr = addr;
    out->size = size;
    out->fd = fd;
    out->mapping = nullptr;
    return true;
}

void CloseShm(ShmRegion* region) {
    if (region->addr) {
        munmap(region->addr, region->size);
        region->addr = nullptr;
    }
    if (region->fd >= 0) {
        close(region->fd);
        region->fd = -1;
    }
    region->size = 0;
}

void UnlinkShm(const std::string& name) {
    shm_unlink(name.c_str());
}

SemHandle CreateSem(const std::string& name) {
    // Darwin has no unnamed process-shared semaphores (sem_init(pshared=1) is
    // ENOSYS) -- named semaphores via sem_open are the only option. Clear any
    // stale semaphore from a prior crashed run before creating a fresh one at
    // value 0, so a leftover posted count can't desync a new run.
    sem_unlink(name.c_str());
    sem_t* s = sem_open(name.c_str(), O_CREAT | O_EXCL, 0600, 0);
    if (s == SEM_FAILED) {
        std::fprintf(stderr, "RLIpc: sem_open(%s) failed: %s\n", name.c_str(), strerror(errno));
        return kInvalidSem;
    }
    return reinterpret_cast<SemHandle>(s);
}

void WaitSem(SemHandle sem) {
    if (sem != kInvalidSem && sem != nullptr) sem_wait(reinterpret_cast<sem_t*>(sem));
}

void PostSem(SemHandle sem) {
    if (sem != kInvalidSem && sem != nullptr) sem_post(reinterpret_cast<sem_t*>(sem));
}

void CloseSem(SemHandle sem) {
    if (sem != kInvalidSem && sem != nullptr) sem_close(reinterpret_cast<sem_t*>(sem));
}

void UnlinkSem(const std::string& name) {
    sem_unlink(name.c_str());
}

#endif

}  // namespace RLIpc
