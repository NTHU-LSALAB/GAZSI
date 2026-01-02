/*
 * GPU Packet Processing - Utility Functions
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>

#include <doca_version.h>
#include <doca_argp.h>

#include "utils.h"

noreturn doca_error_t sdk_version_callback(void *param, void *doca_config)
{
	(void)(param);
	(void)(doca_config);

	printf("DOCA SDK     Version (Compilation): %s\n", doca_version());
	printf("DOCA Runtime Version (Runtime):     %s\n", doca_version_runtime());

	doca_argp_destroy();
	exit(EXIT_SUCCESS);
}

#ifndef DOCA_USE_LIBBSD

#include <string.h>

size_t strlcpy(char *dst, const char *src, size_t size)
{
	size_t src_len = strlen(src);
	if (size > 0) {
		size_t copy_len = (src_len < size - 1) ? src_len : size - 1;
		memcpy(dst, src, copy_len);
		dst[copy_len] = '\0';
	}
	return src_len;
}

size_t strlcat(char *dst, const char *src, size_t size)
{
	size_t dst_len = strnlen(dst, size);
	if (dst_len >= size)
		return size;
	return dst_len + strlcpy(dst + dst_len, src, size - dst_len);
}

#endif /* DOCA_USE_LIBBSD */
