import os
import pickle
import time
from datetime import datetime
import parser

label_rules = {
    "PortMap": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 09:43:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 09:51:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP", "TCP"],
    },
    "NetBIOS": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 10:00:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 10:09:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [137],
    },
    "LDAP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 10:21:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 10:30:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "source_port": [636],
    },
    "MSSQL": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 10:33:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 10:42:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
        "destination_port": [1434],
    },
    "UDP": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 10:53:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 11:03:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
    },
    "UDP-Lag": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 11:14:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 11:24:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["UDP"],
    },
    "SYN": {
        "ip": ["172.16.0.5"],
        "start_time": datetime.strptime("2018-11-03 11:28:00", "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.strptime("2018-11-03 17:35:00", "%Y-%m-%d %H:%M:%S"),
        "protocol": ["TCP"],
    },
    "skip": {
        "ip": ["172.16.0.5"],
    },
    "Benign": {},
}
pcaps_path = "CICDDoS2019/PCAPs/03-11/"
pcaps_name = "SAT-03-11-2018_0"
pcaps_list = []
inbound_ips = ['192.168.50.1', '192.168.50.4', '205.174.165.81', '192.168.50.8', '192.168.50.5', '192.168.50.6','192.168.50.7', '192.168.50.8', '192.168.50.9']
for i in range(0, 146):
    if i == 0:
        pcaps_list.append(pcaps_name)
    else:
        pcaps_list.append(pcaps_name + str(i))
parser.pcapsToCSVs(
        pcaps_path,
        pcaps_list,
        "csv/03-11/",
        5000000,
        label_rules,
        3,
        inbound_ips,
        )
