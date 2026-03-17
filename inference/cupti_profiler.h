#ifndef CUPTI_PROFILER_H
#define CUPTI_PROFILER_H

#ifdef __cplusplus
extern "C" {
#endif

int cupti_profiler_init(void);
void cupti_profiler_dump(void);
void cupti_profiler_fini(void);

#ifdef __cplusplus
}
#endif

#endif
