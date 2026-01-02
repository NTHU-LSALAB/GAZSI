/*
 * GPU Packet Processing - Packet Structures
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef GPU_PKT_PACKETS_H
#define GPU_PKT_PACKETS_H

#include "common.h"

#define WHITESPACE_ASCII 0x20

enum tcp_flags {
	TCP_FLAG_FIN = (1 << 0),
	/* set tcp packet with Fin flag */
	TCP_FLAG_SYN = (1 << 1),
	/* set tcp packet with Syn flag */
	TCP_FLAG_RST = (1 << 2),
	/* set tcp packet with Rst flag */
	TCP_FLAG_PSH = (1 << 3),
	/* set tcp packet with Psh flag */
	TCP_FLAG_ACK = (1 << 4),
	/* set tcp packet with Ack flag */
	TCP_FLAG_URG = (1 << 5),
	/* set tcp packet with Urg flag */
	TCP_FLAG_ECE = (1 << 6),
	/* set tcp packet with ECE flag */
	TCP_FLAG_CWR = (1 << 7),
	/* set tcp packet with CQE flag */
};

struct ether_hdr {
	uint8_t d_addr_bytes[ETHER_ADDR_LEN]; /* Destination addr bytes in tx order */
	uint8_t s_addr_bytes[ETHER_ADDR_LEN]; /* Source addr bytes in tx order */
	uint16_t ether_type;		      /* Frame type */
} __attribute__((__packed__));

struct ipv4_hdr {
	uint8_t version_ihl;	  /* version and header length */
	uint8_t type_of_service;  /* type of service */
	uint16_t total_length;	  /* length of packet */
	uint16_t packet_id;	  /* packet ID */
	uint16_t fragment_offset; /* fragmentation offset */
	uint8_t time_to_live;	  /* time to live */
	uint8_t next_proto_id;	  /* protocol ID */
	uint16_t hdr_checksum;	  /* header checksum */
	uint32_t src_addr;	  /* source address */
	uint32_t dst_addr;	  /* destination address */
} __attribute__((__packed__));

struct tcp_hdr {
	uint16_t src_port; /* TCP source port */
	uint16_t dst_port; /* TCP destination port */
	uint32_t sent_seq; /* TX data sequence number */
	uint32_t recv_ack; /* RX data acknowledgment sequence number */
	uint8_t dt_off;	   /* Data offset */
	uint8_t tcp_flags; /* TCP flags */
	uint16_t rx_win;   /* RX flow control window */
	uint16_t cksum;	   /* TCP checksum */
	uint16_t tcp_urp;  /* TCP urgent pointer, if any */
} __attribute__((__packed__));

struct eth_ip_tcp_hdr {
	struct ether_hdr l2_hdr; /* Ethernet header */
	struct ipv4_hdr l3_hdr;	 /* IP header */
	struct tcp_hdr l4_hdr;	 /* TCP header */
} __attribute__((__packed__));

#endif /* GPU_PKT_PACKETS_H */
