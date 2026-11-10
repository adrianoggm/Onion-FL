from __future__ import annotations

"""Minimal AF_PACKET sniffer of MQTT (port 1883) segments, kernel timestamps.

    python sniff_mqtt.py OUT.jsonl   (inside a container, in the namespace sniffed)

One JSON line per TCP segment to or from port 1883: kernel time, interface,
packet type, flow, seq, ack, payload length, flags, and for a segment that
holds the start of a PUBLISH of an Onion-FL frame, its msg_id and the seq of
the PUBLISH's last byte.
"""

import json
import re
import signal
import socket
import struct
import sys

ETH_P_ALL = 0x0003
SO_TIMESTAMPNS = 35
MSG_ID = re.compile(rb'"msg_id": "([^"]+)"')
TOPIC = b"onionfl/"


def varint(buf):
    value, mult = 0, 1
    for i, b in enumerate(buf):
        value += (b & 0x7F) * mult
        if b < 0x80:
            return value, i + 1
        mult *= 128
    return None, None


def publish_end(payload, seq, p):
    """seq of the byte after the PUBLISH whose topic starts at ``p``."""
    if p < 3:
        return None
    for k in range(1, 5):
        s = p - 3 - k
        if s < 0:
            break
        if payload[s] & 0xF0 != 0x30:
            continue
        rem, n = varint(payload[s + 1 : s + 1 + k])
        if n == k:
            return (seq + s + 1 + k + rem) % 2**32
    return None


def main(out_path):
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
    sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 24)
    out = open(out_path, "w", buffering=1 << 16)
    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    sock.settimeout(0.5)
    while not stop:
        try:
            data, anc, _, addr = sock.recvmsg(65536, 64)
        except TimeoutError:
            continue
        except InterruptedError:
            continue
        ts = None
        for level, kind, cdata in anc:
            if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS:
                sec, nsec = struct.unpack("qq", cdata[:16])
                ts = sec + nsec * 1e-9
        ifname, _, pkttype = addr[0], addr[1], addr[2]
        if ifname == "lo":
            off = 14  # loopback still has an Ethernet header on AF_PACKET
        else:
            off = 14
        if len(data) < off + 20 or data[off - 2 : off] != b"\x08\x00":
            continue
        ihl = (data[off] & 0x0F) * 4
        if data[off + 9] != 6:
            continue
        total = struct.unpack("!H", data[off + 2 : off + 4])[0]
        src = socket.inet_ntoa(data[off + 12 : off + 16])
        dst = socket.inet_ntoa(data[off + 16 : off + 20])
        t = off + ihl
        sport, dport, seq, ack = struct.unpack("!HHII", data[t : t + 12])
        if 1883 not in (sport, dport):
            continue
        doff = (data[t + 12] >> 4) * 4
        flags = data[t + 13]
        payload = data[t + doff : off + total]
        row = {
            "ts": ts,
            "if": ifname,
            "pt": pkttype,
            "src": f"{src}:{sport}",
            "dst": f"{dst}:{dport}",
            "seq": seq,
            "ack": ack,
            "len": len(payload),
            "fl": flags,
        }
        if payload:
            pubs = []
            p = payload.find(TOPIC)
            while p >= 0:
                m = MSG_ID.search(payload, p)
                if m:
                    pubs.append([m.group(1).decode(), publish_end(payload, seq, p), p])
                p = payload.find(TOPIC, p + 1)
            if pubs:
                row["pubs"] = pubs
        out.write(json.dumps(row) + "\n")
    out.close()


if __name__ == "__main__":
    main(sys.argv[1])
