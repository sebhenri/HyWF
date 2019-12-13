
from constants import *
import re
import random
from utils import *

import logging

logger = logging.getLogger('wtfpad')


def parse(fpath):
    # Parse a file assuming Tao's format.
    t = Trace()
    for line in open(fpath):
        line = line.replace("\n", "")
        tokens = re.split("[\t ]", line)
        timestamp = float(tokens[0])
        length = int(float(tokens[1]))
        direction = length / abs(length)
        t.append(Packet(timestamp, direction, length))
    return t


def dump(trace, fpath):
    # Write trace packet into file `fpath`.
    with open(fpath, 'w') as fo:
        for packet in trace:
            fo.write(str(packet) + NL)


class Packet(object):
    """Define a packet.

    Direction is defined in the wfpaddef const.py.
    """
    payload = None

    def __init__(self, timestamp, direction, length, dummy=False, rand_out=False, sec_channel_pre=False,
                 sec_channel=False):
        self.timestamp = timestamp
        self.direction = int(direction)
        self.length = length
        self.dummy = dummy
        self.rand_out = rand_out
        self.sec_channel_pre = sec_channel_pre
        self.sec_channel = sec_channel
        self.id = 0
        self.post_id = -1

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __str__(self):
        p_repr = '\t'.join(map(str, [self.timestamp, self.direction * abs(self.length)]))
        if self.dummy:
            p_repr += "\td"
        p_repr += "\t"+str(self.post_id)
        return p_repr

    def formatted(self):
        return "("+(', '.join(map(str, [self.timestamp, self.direction * abs(self.length), self.id])))+")"+\
               (" dummy" if self.dummy else "")

class Trace(list):
    """Define trace as a list of Packets."""

    _index = 0

    def __init__(self,  list_packets=None):
        if list_packets:
            count = 0
            for p in list_packets:
                p.id = count
                self.append(p)
                count += 1
        else:
            list.__init__(self)

    def __getslice__(self, i, j):
        return Trace(list_packets=list.__getslice__(self, i, j))

    def __add__(self, new_packet):
        t = Trace(self.pcap)
        l = list.__add__(self, new_packet)
        for e in l:
            t.append(e)
        return t

    def __mul__(self, other):
        return Trace(list.__mul__(self, other))

    def get_next_by_direction(self, i, direction):
        for j, p in enumerate(self[i + 1:]):
            if p.direction == direction and not p.rand_out and not p.sec_channel:
                return i + j + 1
        raise IndexError("No packets in this direction.")

    def next(self):
        try:
            i = self[self._index]
        except IndexError:
            raise StopIteration
        self._index += 1
        return i

    def total_size(self):
        size_trace = 0
        for p in self:
            size_trace += abs(p.length)
        return size_trace

    def size_dir(self, direction):
        len_trace = 0
        for p in self:
            if p.direction == direction:
                len_trace += abs(p.length)
        return len_trace

    def max_time(self):
        return self[-1].timestamp

    def last_real(self):
        i = len(self) - 1
        while i >= 0:
            if not self[i].dummy:
                # last packet that is not dummy
                return i
            i -= 1
        return -1

    def extract_after(self, i):
        return Trace(self[i+1:])

    def remove_first_packets(self, n):
        removed = list(self[0:n])  # works even if n > len(self)
        del self[0:n]
        return Trace(removed)

    def remove_first_time(self, t):
        removed = list([p for p in self if p.timestamp < t])
        del self[0:len(removed)]
        return Trace(removed)

    def remove_all(self):
        removed = list(self)
        del self[0:len(removed)]
        return Trace(removed)

    def extract_prob(self, prob=1, set_sec_channel=True, do_not_remove=False, cons_pack=1, cons_dist="fixed"):
        packets = []
        to_secure = []

        loc_cons = get_exp(cons_pack, cons_dist)
        num_packets = loc_cons
        send_sec = True  # initial value does not matter
        for p in self:
            # send loc_cons consecutive packets
            if num_packets == loc_cons:
                # choose packets with probability prob
                if random.random() <= prob:
                    # send on this technology
                    send_sec = False
                else:
                    # send on the other
                    send_sec = True
                num_packets = 1
                # redraw number of consecutive packets
                loc_cons = get_exp(cons_pack, cons_dist)
            else:
                # keep value send_sec, increase number of packets sent
                num_packets += 1

            if not send_sec:
                packets.append(p)
            else:
                if set_sec_channel:
                    p.sec_channel_pre = True
                to_secure.append(p)
                if do_not_remove:
                    # also keep secure channel packets
                    packets.append(p)
        return Trace(packets), Trace(to_secure)

    def extract_dir(self, direction, first_n=INF):
        packets = []
        for p in self:
            if p.direction == direction and len(packets) < first_n:
                packets.append(p)
        return Trace(packets)

    def extract_real_packets(self, first_n=INF):
        real_packets = []
        for p in self:
            if not p.dummy and len(real_packets) < first_n:
                real_packets.append(p)
        return Trace(real_packets)

    def extract_dummy_packets(self, first_n=INF):
        dummy_packets = []
        for p in self:
            if p.dummy and len(dummy_packets) < first_n:
                dummy_packets.append(p)
        return Trace(dummy_packets)

    def separate_packets(self):
        real_packets = []
        dummy_packets = []
        for p in self:
            if p.dummy:
                dummy_packets.append(p)
            else:
                real_packets.append(p)
        return Trace(real_packets), Trace(dummy_packets)

    def remove_first_incoming(self):
        # a trace should start with outgoing packet, remove first incoming
        for i, p in enumerate(self):
            if p.direction == OUT:
                return self.extract_after(i-1)

    def update_times(self):
        # remove fixed time so that trace starts at 0
        if len(self) == 0:
            return self
        first_time = self[0].timestamp
        for p in self:
            p.timestamp -= first_time

    def extend_trace(self, new_trace):
        self.extend(new_trace)
        self.sort(key=lambda x: x.timestamp)

    def __str__(self):
        return "\n".join(map(str, self))


class Flow(Trace):
    """Provides a structure to keep flow dependent variables."""

    def __init__(self, direction):
        """Initialize direction and state of the flow."""
        self.direction = direction
        name = "in" if direction == -1 else "out"
        self.machine = Machine(name, self, "snd")
        self.opp_machine = -1
        self.current_burst = 0
        self.current_sent_sec_burst = 0
        self.current_sent_sec_bias = 0
        self.burst_lengths = []
        self.token_burst = INF
        self.min_burst = INF
        # keep for compatibility
        self.name = name
        self.expired = False
        self.state = WAIT
        self.timeout = 0.0
        self.last_draw = -1.0
        self.done = True  # set to False at first packet
        self.initialized = False  # initialized at first packet
        Trace.__init__(self)

    def set_opp_machine(self, opp_flow):
        # state machine for dummy answers
        self.opp_machine = Machine("in_out" if self.direction == -1 else "out_in", opp_flow, "rcv")

class Machine(Trace):
    """State machine."""

    def __init__(self, name, flow, on):
        """Initialize state machine."""
        self.name = name
        self.expired = False
        self.state = WAIT
        self.timeout = INF
        self.flow = flow
        self.direction = flow.direction
        self.last_draw = -1.0
        self.on = on
