/*
 * GPU Packet Processing - Utility Functions Header
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <stdint.h>
#include <doca_error.h>

#ifndef MIN
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#endif

/* Prints DOCA SDK and runtime versions, then exits */
doca_error_t sdk_version_callback(void *param, void *doca_config);

#ifdef DOCA_USE_LIBBSD
#include <bsd/string.h>
#else
size_t strlcpy(char *dst, const char *src, size_t size);
size_t strlcat(char *dst, const char *src, size_t size);
#endif

#endif /* UTILS_H_ */
