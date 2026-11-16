/*
 * Runtime shield for the packaged engine.
 *
 * Every signal below feeds a digest that derives the bytecode decryption key.
 * There is deliberately no "if detected then refuse" branch: patching a check
 * changes a digest input, so decryption fails later with bad marshal data.
 * The derived key is baked in at build time by protect_engine.py.
 */
#include "shield_config.h"

#include <CommonCrypto/CommonDigest.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach-o/dyld.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define SIGNAL_COUNT 5
#define DIGEST_SIZE (SIGNAL_COUNT + CC_SHA256_DIGEST_LENGTH)
#define TIMING_ITERATIONS 200000

static unsigned char g_key[CC_SHA256_DIGEST_LENGTH];
static int g_ready = 0;

static int vx_traced(void) {
    struct kinfo_proc info;
    size_t size = sizeof(info);
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, (int)getpid()};
    memset(&info, 0, sizeof(info));
    if (sysctl(mib, 4, &info, &size, NULL, 0) != 0) return 0;
    return (info.kp_proc.p_flag & P_TRACED) != 0;
}

static unsigned char vx_deny_and_verify(void) {
    typedef int (*ptrace_fn)(int, pid_t, void *, int);
    ptrace_fn fn = (ptrace_fn)dlsym(RTLD_DEFAULT, "ptrace");
    if (fn) (void)fn(PT_DENY_ATTACH, 0, NULL, 0);
    return (unsigned char)vx_traced();
}

static int vx_contains(const char *haystack, const char *needle) {
    if (!haystack || !needle) return 0;
    size_t n = strlen(needle);
    for (const char *p = haystack; *p; ++p) {
        size_t i = 0;
        while (i < n && (p[i] | 0x20) == (needle[i] | 0x20)) ++i;
        if (i == n) return 1;
    }
    return 0;
}

static unsigned char vx_injected_images(void) {
    static const char *markers[] = {"frida", "gum", "gadget", "substrate",
                                    "cynject", "fishhook", "objection", "cycript", NULL};
    int found = 0;
    uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; ++i) {
        const char *name = _dyld_get_image_name(i);
        for (int j = 0; markers[j]; ++j) {
            if (vx_contains(name, markers[j])) { ++found; break; }
        }
    }
    return (unsigned char)(found > 255 ? 255 : found);
}

static unsigned char vx_insert_library(void) {
    const char *value = getenv("DYLD_INSERT_LIBRARIES");
    return (value && *value) ? 1 : 0;
}

static unsigned char vx_injected_threads(void) {
    static const char *names[] = {"gmain", "gum-js-loop", "pool-frida", "frida", NULL};
    thread_array_t threads = NULL;
    mach_msg_type_number_t count = 0;
    unsigned char found = 0;
    if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS) return 0;
    for (mach_msg_type_number_t i = 0; i < count; ++i) {
        pthread_t thread = pthread_from_mach_thread_np(threads[i]);
        char name[64] = {0};
        if (!thread || pthread_getname_np(thread, name, sizeof(name)) != 0) continue;
        for (int j = 0; names[j]; ++j) {
            if (vx_contains(name, names[j])) { found = 1; break; }
        }
        if (found) break;
    }
    vm_deallocate(mach_task_self(), (vm_address_t)threads, count * sizeof(thread_t));
    return found;
}

static unsigned char vx_timing_anomaly(void) {
    volatile unsigned long accumulator = 0;
    mach_timebase_info_data_t base;
    mach_timebase_info(&base);
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < TIMING_ITERATIONS; ++i) accumulator += (unsigned long)i;
    uint64_t elapsed = mach_absolute_time() - start;
    uint64_t nanoseconds = elapsed * base.numer / base.denom;
    (void)accumulator;
    return nanoseconds > (100ULL * 1000 * 1000) ? 1 : 0;
}

static void vx_self_hash(unsigned char *out) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    const char *own = NULL;
    uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; ++i) {
        const char *name = _dyld_get_image_name(i);
        if (name && strstr(name, "_protect_key.dylib")) { own = name; break; }
    }
    char directory[1024] = {0};
    if (own) {
        strncpy(directory, own, sizeof(directory) - 1);
        char *slash = strrchr(directory, '/');
        if (slash) *slash = '\0';
    }
    const char *names[] = {"_protect.py", "sitecustomize.py"};
    for (int i = 0; i < 2; ++i) {
        char path[1100] = {0};
        snprintf(path, sizeof(path), "%s/%s", directory, names[i]);
        FILE *file = fopen(path, "rb");
        if (!file) continue;
        unsigned char buffer[4096];
        size_t read;
        while ((read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
            CC_SHA256_Update(&context, buffer, (CC_LONG)read);
        }
        fclose(file);
    }
    CC_SHA256_Final(out, &context);
}

void vx_shield_signals(unsigned char out[DIGEST_SIZE]) {
    out[0] = vx_deny_and_verify();
    out[1] = vx_injected_images();
    out[2] = vx_insert_library();
    out[3] = vx_injected_threads();
    out[4] = vx_timing_anomaly();
    vx_self_hash(out + SIGNAL_COUNT);
}

const unsigned char *vx_shield_key(void) {
    if (!g_ready) {
        unsigned char buffer[DIGEST_SIZE];
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        vx_shield_signals(buffer);
        CC_SHA256(buffer, DIGEST_SIZE, digest);
        for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
            g_key[i] = (unsigned char)(vx_derived_key[i] ^ digest[i]);
        }
        g_ready = 1;
    }
    return g_key;
}

int vx_shield_key_length(void) { return CC_SHA256_DIGEST_LENGTH; }
