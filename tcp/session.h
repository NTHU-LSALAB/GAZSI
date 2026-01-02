/*
 * GPU Packet Processing - TCP Session Table Header
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef TCP_SESSION_H
#define TCP_SESSION_H

#include <stdint.h>
#include <rte_common.h>
#include <rte_byteorder.h>
#include <rte_hash.h>

#define TCP_SESSION_MAX_ENTRIES 4096

/* TCP session key */
struct tcp_session_key {
	rte_be32_t src_addr;
	rte_be32_t dst_addr;
	rte_be16_t src_port;
	rte_be16_t dst_port;
};

/* TCP session entry */
struct tcp_session_entry {
	struct tcp_session_key key;
	struct doca_flow_pipe_entry *flow;
};

/* TCP session hash table parameters */
extern struct rte_hash_parameters tcp_session_ht_params;

/* TCP session table (created in flow.c) */
extern struct rte_hash *tcp_session_table;

#endif /* TCP_SESSION_H */
