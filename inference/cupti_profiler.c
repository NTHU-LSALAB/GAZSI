#include "cupti_profiler.h"
#include <cupti.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE 8
#define MAX_KERNELS 64

struct kernel_stats {
    char name[128];
    uint64_t total_ns;
    uint64_t count;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
};

static struct kernel_stats g_stats[MAX_KERNELS];
static int g_num_kernels = 0;
static int g_initialized = 0;

static struct kernel_stats *find_or_create(const char *name)
{
    for (int i = 0; i < g_num_kernels; i++) {
        if (strcmp(g_stats[i].name, name) == 0)
            return &g_stats[i];
    }
    if (g_num_kernels >= MAX_KERNELS)
        return NULL;
    struct kernel_stats *s = &g_stats[g_num_kernels++];
    memset(s, 0, sizeof(*s));
    strncpy(s->name, name, sizeof(s->name) - 1);
    return s;
}

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                                       size_t *maxNumRecords)
{
    *buffer = (uint8_t *)aligned_alloc(ALIGN_SIZE, BUF_SIZE);
    *size = BUF_SIZE;
    *maxNumRecords = 0;
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t streamId,
                                       uint8_t *buffer, size_t size,
                                       size_t validSize)
{
    CUpti_Activity *record = NULL;
    while (1) {
        CUptiResult r = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (r == CUPTI_ERROR_MAX_LIMIT_REACHED || r != CUPTI_SUCCESS)
            break;

        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
            record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            CUpti_ActivityKernel5 *k = (CUpti_ActivityKernel5 *)record;
            struct kernel_stats *s = find_or_create(k->name);
            if (s) {
                s->total_ns += (k->end - k->start);
                s->count++;
                s->grid_x = k->gridX;
                s->grid_y = k->gridY;
                s->grid_z = k->gridZ;
                s->block_x = k->blockX;
                s->block_y = k->blockY;
                s->block_z = k->blockZ;
            }
        }
    }
    free(buffer);
}

int cupti_profiler_init(void)
{
    CUptiResult r;
    r = cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    if (r != CUPTI_SUCCESS) {
        fprintf(stderr, "[CUPTI] RegisterCallbacks failed: %d\n", r);
        return -1;
    }
    r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (r != CUPTI_SUCCESS) {
        r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
        if (r != CUPTI_SUCCESS) {
            fprintf(stderr, "[CUPTI] ActivityEnable failed: %d\n", r);
            return -1;
        }
    }
    g_initialized = 1;
    fprintf(stderr, "[CUPTI] Profiler initialized\n");
    return 0;
}

void cupti_profiler_dump(void)
{
    if (!g_initialized) return;
    cuptiActivityFlushAll(0);

    uint64_t total_all = 0;
    for (int i = 0; i < g_num_kernels; i++)
        total_all += g_stats[i].total_ns;

    fprintf(stderr, "\n[CUPTI] === Kernel SM Breakdown ===\n");
    fprintf(stderr, "%-60s %10s %10s %8s  grid         block\n",
            "Kernel", "Time(ms)", "Count", "Share%");
    for (int i = 0; i < g_num_kernels; i++) {
        struct kernel_stats *s = &g_stats[i];
        double ms = s->total_ns / 1e6;
        double pct = total_all > 0 ? (double)s->total_ns / total_all * 100.0 : 0;
        fprintf(stderr, "%-60s %10.1f %10lu %7.2f%%  (%u,%u,%u)  (%u,%u,%u)\n",
                s->name, ms, (unsigned long)s->count, pct,
                s->grid_x, s->grid_y, s->grid_z,
                s->block_x, s->block_y, s->block_z);
    }
    fprintf(stderr, "%-60s %10.1f\n", "TOTAL", total_all / 1e6);
    fprintf(stderr, "[CUPTI] === End ===\n\n");
}

void cupti_profiler_fini(void)
{
    if (!g_initialized) return;
    cuptiActivityFlushAll(0);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    g_initialized = 0;
}
