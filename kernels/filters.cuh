/*
 * GPU Packet Processing - Packet Filters
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef GPU_PKT_FILTERS_H
#define GPU_PKT_FILTERS_H

#include "common.h"
#include "packets.h"

#define ACK_MASK (0x00 | TCP_FLAG_ACK)

__device__ __inline__ int
raw_to_tcp(const uintptr_t buf_addr, struct eth_ip_tcp_hdr **hdr, uint8_t **payload)
{
	(*hdr) = (struct eth_ip_tcp_hdr *) buf_addr;
	(*payload) = (uint8_t *) (buf_addr + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + (((*hdr)->l4_hdr.dt_off >> 4) * sizeof(int)));

	return 0;
}

__device__ __inline__ int
wipe_packet_32b(uint8_t *payload)
{
#pragma unroll
	for (int idx = 0; idx < 32; idx++)
		payload[idx] = 0;

	return 0;
}

/* TCP */
__device__ __inline__ int
filter_is_http(const uint8_t *pld)
{
	/* HTTP/1.1 */
	if (pld[0] != 'H')
		return 0;
	if (pld[1] == 'T' && pld[2] == 'T' && pld[3] == 'P' && pld[4] == '/' && pld[5] == '1' && pld[6] == '.' && pld[7] == '1')
		return 1;
	return 0;
}

__device__ __inline__ int
filter_is_http_get(const uint8_t *pld)
{
	/* GET / */
	if (pld[0] != 'G')
		return 0;
	if (pld[1] == 'E' && pld[2] == 'T' && pld[3] == WHITESPACE_ASCII && pld[4] == '/')
		return 1;
	return 0;
}

__device__ __inline__ int
filter_is_http_post(const uint8_t *pld)
{
	/* POST / */
	if (pld[0] != 'P')
		return 0;
	if (pld[1] == 'O' && pld[2] == 'S' && pld[3] == 'T' && pld[4] == WHITESPACE_ASCII && pld[5] == '/')
		return 1;
	return 0;
}

__device__ __inline__ int
filter_is_http_head(const uint8_t *pld)
{
	/* HEAD / */
	if (pld[0] != 'H' || pld[1] != 'E')
		return 0;
	if (pld[2] == 'A' && pld[3] == 'D' && pld[4] == WHITESPACE_ASCII && pld[5] == '/')
		return 1;
	return 0;
}

__device__ __inline__ int
filter_is_tcp_syn(const struct tcp_hdr *l4_hdr)
{
	return l4_hdr->tcp_flags & TCP_FLAG_SYN;
}

__device__ __inline__ int
filter_is_tcp_fin(const struct tcp_hdr *l4_hdr)
{
	return l4_hdr->tcp_flags & TCP_FLAG_FIN;
}

__device__ __inline__ int
filter_is_tcp_ack(const struct tcp_hdr *l4_hdr)
{
	return l4_hdr->tcp_flags & ACK_MASK;
}

#endif /* GPU_PKT_FILTERS_H */