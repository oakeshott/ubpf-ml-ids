import os
import pickle
import time
from datetime import datetime
import parser

# Looking at the packets, many of the 'Attack times' in the paper are visibly incorrect
# (eg. Training Set NTP contains LDAP packages?  Port 636 is LDAP, not NTP and this is
# clearly script traffic, also SYNs are delayed 14:29 and after)
# So I looked at the packets myself in order to verify the 'Attack times' I use below
# These new 'Attack times' are more accurate than the dataset author's.
# Also as seen in 'Labeling proof.ipynb', I prove that all attacks come from 172.16.0.5,
# since that perfectly splits authors CSVs into benign and malicious packets
# Also, I subtract 3 hours from the timestamps, since they are in UTC and the authors
# seem to be in a UTC -3 timezone

label_rules = {
    "NTP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [1023],
    },
    "DNS": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [53],
    },
    "LDAP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "source_port": [636],
    },
    "MSSQL": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [1434],
    },
    "NetBIOS": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 12:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [137],
    },
    "SNMP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 13:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "source_port": [161, 162],
    },
    "SSDP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 10:17:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 13:00:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "source_port": [2869, 5000],
    },
    "UDP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 12:45:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 13:09:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
    },
    "UDP-Lag": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 13:13:17", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime(
            "2018-12-01 13:26:00", "%Y-%m-%d %H:%M:%S"
        ),  # From 13:11 to 13:13 there are still UDP flood packets. From 13:15 to 13:26 the same attack patters are seen
        "protocol": ["UDP"],
    },
    "WebDDoS": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 13:18:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 14:29:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["TCP"],
        "destination_port": [80],
    },
    "SYN": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-12-01 14:30:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-12-01 17:15:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["TCP"],
    },
    "TFTP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime(
            "2018-12-01 14:40:00", "%Y-%m-%d %H:%M:%S"
        ),  # From 13:35:00 to 14:40:00 and after 15:30:30, the packet sizes are off and no traffic on port 69 so I don't know what the packets are
        "end_time": datetime.strptime("2018-12-01 15:30:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
    },
    "skip": {
        "ip": ["172.16.0.5"],
    },
    "Benign": {},
}
pcaps_path = "CICDDoS2019/PCAPs/01-12/"
pcaps_name = "SAT-01-12-2018_0"
pcaps_list = []
inbound_ips = ['192.168.50.1', '192.168.50.4', '205.174.165.81', '192.168.50.8', '192.168.50.5', '192.168.50.6','192.168.50.7', '192.168.50.8', '192.168.50.9']
for i in range(0, 818):
    if i == 0:
        pcaps_list.append(pcaps_name)
    else:
        pcaps_list.append(pcaps_name + str(i))
parser.pcapsToCSVs(
        pcaps_path,
        pcaps_list,
        "csv/01-12/",
        5000000,
        label_rules,
        3,
        inbound_ips
        )
