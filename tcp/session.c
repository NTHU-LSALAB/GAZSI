/*
 * GPU Packet Processing - TCP Session Table
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <rte_jhash.h>

#include "session.h"

struct rte_hash_parameters tcp_session_ht_params = {
	.name = "tcp_session_ht",
	.entries = TCP_SESSION_MAX_ENTRIES,
	.key_len = sizeof(struct tcp_session_key),
	.hash_func = rte_jhash,
	.hash_func_init_val = 0,
	.extra_flag = 0, // remember: if >1 lcore is needed, make this thread-safe and update insert/delete logic
};

struct rte_hash *tcp_session_table;
