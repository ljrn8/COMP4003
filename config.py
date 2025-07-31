import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# dotenv constants
# dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
# load_dotenv(dotenv_path=dotenv_path)

# UNSWNB15_ROOT = Path(os.getenv("UNSWNB15_ROOT"))
# CICIDS2017_ROOT = Path(os.getenv("CICIDS2017_ROOT"))
# BOTIOT_ROOT = Path(os.getenv("BOTIOT_ROOT"))

LOG_LEVEL = os.getenv("LOG_LEVEL") or "INFO"

# Local Directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = Path(ROOT) / "models"
FIGURES_DIR = Path(ROOT) / "figures"
INTERM_DIR = Path(ROOT) / "interm"
DATA_ROOT = Path(ROOT) / "IDS datasets"
...  # others

UQ_dtypes = {
    'IPV4_SRC_ADDR': 'object',	                # IPv4 source address
    'IPV4_DST_ADDR': 'object',	                # IPv4 destination address
    'L4_SRC_PORT': 'object',	                # IPv4 source port number
    'L4_DST_PORT': 'object',	                # IPv4 destination port number

    'PROTOCOL': 'int32',	                # IP protocol identifier byte
    'L7_PROTO': 'category',	                # Layer 7 protocol (numeric)
    'IN_BYTES': 'int32',	                # Incoming number of bytes
    'OUT_BYTES': 'int32',	                # Outgoing number of bytes
    'IN_PKTS': 'int32',	                # Incoming number of packets
    'OUT_PKTS': 'int32',	                # Outgoing number of packets
    'FLOW_DURATION_MILLISECONDS': 'float32',	                # Flow duration in milliseconds
    'TCP_FLAGS': 'int32',	                # Cumulative of all TCP flags
    'CLIENT_TCP_FLAGS': 'int32',	                # Cumulative of all client TCP flags
    'SERVER_TCP_FLAGS': 'int32',	                # Cumulative of all server TCP flags
    'DURATION_IN': 'float32',	                # Client to Server stream duration (msec)
    'DURATION_OUT': 'float32',	                # Client to Server stream duration (msec)
    'MIN_TTL': 'float32',	                # Min flow TTL
    'MAX_TTL': 'float32',	                # Max flow TTL
    'LONGEST_FLOW_PKT': 'int32',	            # Longest packet (bytes) of the flow
    'SHORTEST_FLOW_PKT': 'int32',	            # Shortest packet (bytes) of the flow
    'MIN_IP_PKT_LEN': 'int32',	            # Len of the smallest flow IP packet observed
    'MAX_IP_PKT_LEN': 'int32',	            # Len of the largest flow IP packet observed
    'SRC_TO_DST_SECOND_BYTES': 'float32',	            # Src to dst Bytes/sec
    'DST_TO_SRC_SECOND_BYTES': 'float32',	            # Dst to src Bytes/sec
    'RETRANSMITTED_IN_BYTES': 'int32',	            # Number of retransmitted TCP flow bytes (src->dst)
    'RETRANSMITTED_IN_PKTS': 'int32',	            # Number of retransmitted TCP flow packets (src-...
    'RETRANSMITTED_OUT_BYTES': 'int32',         # 	Number of retransmitted TCP flow bytes (dst->src)
    'RETRANSMITTED_OUT_PKTS': 'int32',	            # Number of retransmitted TCP flow packets (dst-...
    'SRC_TO_DST_AVG_THROUGHPUT': 'float32',	            # Src to dst average thpt (bps)
    'DST_TO_SRC_AVG_THROUGHPUT': 'float32',         # 	Dst to src average thpt (bps)
    'NUM_PKTS_UP_TO_128_BYTES': 'int32',	            # Packets whose IP size <= 128
    'NUM_PKTS_128_TO_256_BYTES': 'int32',	            # Packets whose IP size > 128 and <= 256
    'NUM_PKTS_256_TO_512_BYTES': 'int32',	            # Packets whose IP size > 256 and <= 512
    'NUM_PKTS_512_TO_1024_BYTES': 'int32',	            # Packets whose IP size > 512 and <= 1024
    'NUM_PKTS_1024_TO_1514_BYTES': 'int32',	            # Packets whose IP size >��1024 and <= 1514
    'TCP_WIN_MAX_IN': 'int32',	            # Max TCP Window (src->dst)
    'TCP_WIN_MAX_OUT': 'int32',         # 	Max TCP Window (dst->src)

    # ?
    'ICMP_TYPE': 'category',	            # ICMP Type * 256 + ICMP code
    'ICMP_IPV4_TYPE': 'category',	            # ICMP Type

    'DNS_QUERY_ID': 'float32',	            #    DNS query transaction Id
    'DNS_QUERY_TYPE': 'category',	        #            DNS query type (e.g. 1=A, 2=NS..)
    'DNS_TTL_ANSWER': 'float32',	        #    TTL of the first A record (if any)
    'FTP_COMMAND_RET_CODE': 'category',	    #    FTP client command return code
    'FLOW_START_MILLISECONDS': 'float32',	#        Flow start timestamp in milliseconds
    'FLOW_END_MILLISECONDS': 'float32',	    #        Flow end timestamp in milliseconds
    'SRC_TO_DST_IAT_MIN': 'float32',	    #    Minimum Inter-Packet Arrval Time (src->dst)
    'SRC_TO_DST_IAT_MAX': 'float32',	    #        Maximum Inter-Packet Arrval Time (src->dst)
    'SRC_TO_DST_IAT_AVG': 'float32',	    #        Average Inter-Packet Arrval Time (src->dst)
    'SRC_TO_DST_IAT_STDDEV': 'float32',	    #        Sandard Deviaion Inter-Packet Arrval Time (src...
    'DST_TO_SRC_IAT_MIN': 'float32',	    #        Minimum Inter-Packet Arrval Time (dst > src)
    'DST_TO_SRC_IAT_MAX': 'float32',	    #        Minimum Inter-Packet Arrval Time (dst > src)
    'DST_TO_SRC_IAT_AVG': 'float32',	    #        Minimum Inter-Packet Arrval Time (dst > src)
    'DST_TO_SRC_IAT_STDDEV': 'float32',	    #        Minimum Inter-Packet Arrval Time (dst > src)
    
}