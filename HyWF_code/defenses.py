import argparse
import os
import re
import numpy as np
import pprint
import sys
import time
import types
import math
import pickle
import random
from constants import *
import configparser
import adaptive
import parser as prs
import logging
import hashlib
from utils import *


# This file implements various defenses, in particular AP and HyWF
# For AP, we use the code by Juarez et al. (ESORICS 2016)

def dequeue_packet(local_values, global_values, args, on_sec_channel=False, t_sent=None):
    # dequeue packet from local queue (real incoming or outgoing queue)
    if t_sent == None:
        t_sent = global_values.current_time
    packet = local_values.queue_packets.pop(0)
    global_values.delays.append(t_sent - packet[0])
    # set packet as sent in global queue (used only for enqueuing in local queue)
    id_packet = packet[2]

    local_values.total_real_bytes += abs(packet[1])
    if on_sec_channel:
        local_values.total_sec_channel_bytes += abs(packet[1])

    global_values.sent_packets[id_packet] = (global_values.sent_packets[id_packet][0], t_sent, packet[1])

    # find index in global queue
    ind = 0
    while global_values.queue_packets[ind][2] < id_packet:
        ind += 1

    args.logger.log(ALL, ("%.2f" % t_sent) +
              ": Dequeuing packet "+repr(packet)+("(on secure channel)" if on_sec_channel else ""))

    if type(local_values.rate) is str and "cs" in local_values.rate and not on_sec_channel:
        # compute intervals for updating rate as in CS-BuFLO
        # if sent on secure channel, do not consider it
        if local_values.last_packet[0] != -1:
            # check whether in same burst (true only if both are still in global queue)

            same_burst = False
            c = 0
            while not same_burst and global_values.queue_packets[c][2] < id_packet \
                    and np.sign(global_values.queue_packets[c][1]) == np.sign(packet[1]):
                if global_values.queue_packets[c][2] == local_values.last_packet[2]:
                    same_burst = True
                c += 1
            if same_burst:
                interval = packet[3] - local_values.last_packet[3]
                args.logger.log(ALL, "\tAdding interval %.3f between packets %s and %s" %
                          (interval, repr(local_values.last_packet), repr(packet)))
                local_values.intervals.append(interval)
        local_values.last_packet = packet

    # remove packets that have been sent in another burst as long as entire burst not sent
    if len(global_values.queue_packets) > ind+1 and np.sign(global_values.queue_packets[ind+1][1]) != np.sign(packet[1]):
        # next packet is of another burst, remove entire burst except this packet
        global_values.queue_packets = global_values.queue_packets[ind:]
        new_ind = 0
    else:
        # keep all packets of this burst
        new_ind = 0
        ind_loc = ind - 1
        while ind_loc >= 0 and np.sign(global_values.queue_packets[ind_loc][1]) == np.sign(packet[1]):
            ind_loc -= 1
            new_ind += 1
        if ind_loc >= 0:
            # packet at index ind has another direction, remove all packets before to keep only this burst
            global_values.queue_packets = global_values.queue_packets[ind_loc+1:]
        else:
            new_ind = ind

    ind = new_ind

    p = global_values.queue_packets[ind]
    sent_packet = (p[0], p[1], p[2], t_sent)
    global_values.queue_packets[ind] = sent_packet

    return packet


def send_packet(local_values, global_values, args, args_defense):
    send_dummy = True
    queue_packets = local_values.queue_packets  # this is just an alias
    if len(queue_packets) > 0:
        # try to send real packet
        real_time = queue_packets[0][0]
        real_size = queue_packets[0][1]
        send_dummy = False
        # compute size of queue in bytes
        queue_size = sum([abs(p[1]) for p in queue_packets])
        if abs(real_size) > args_defense.size_packets:
            # fragment packet
            local_values.total_real_bytes += args_defense.size_packets
            queue_packets[0] = (real_time, real_size - args_defense.size_packets)
            args.logger.log(VDEBUG, ("%.3f" % global_values.current_time) + ": Send real "+local_values.direction+" fragment of " +
                      repr(queue_packets[0]))
            queue_size -= args_defense.size_packets
        elif abs(real_size) < args_defense.size_packets:
            # if there are other packets in queue to send fitting in fixed size packet, send them
            can_send_real = 0
            count = 0
            last_packet = 0
            while len(queue_packets) > 0 and (can_send_real + queue_packets[0][1]) < args_defense.size_packets:
                packet = dequeue_packet(local_values, global_values, args)
                last_packet = packet
                real_time = packet[0]
                can_send_real += abs(packet[1])
                count += 1
            if len(queue_packets) > 0:
                # stopped because next incoming packet did not fit, fragment it
                real_time = queue_packets[0][0]
                real_size = queue_packets[0][1]
                queue_packets[0] = (real_time, real_size - args_defense.size_packets - can_send_real)
                local_values.total_real_bytes += (args_defense.size_packets - can_send_real)
                last_packet = queue_packets[0]
                can_send_real = args_defense.size_packets
                count += 1
            queue_size -= can_send_real
            args.logger.log(VDEBUG, ("%.3f" % global_values.current_time) + ": Send "+local_values.direction+" aggregation with " +
                      str(can_send_real) + " real bytes and " + str(count) + " packets, last " + repr(last_packet))

            local_values.total_dummy_bytes += args_defense.size_packets - can_send_real
        else:  # abs(real_size) == args_defense.size_packets: send packet with correct size
            packet = dequeue_packet(local_values, global_values, args)
            args.logger.log(VDEBUG, ("%.3f" % global_values.current_time) + ": Send real "+local_values.direction +
                            " packet " + repr(packet) + " with correct size")
            queue_size -= args_defense.size_packets
        global_values.real_time_last_packet = real_time
        global_values.defense_time_last_packet = global_values.current_time
        if not args.dont_save:
            global_values.f_out.write("%f\t%d\n" % (global_values.current_time, local_values.sign * args_defense.size_packets))
        local_values.queue_sizes.append(queue_size)

    else:
        args.logger.log(VDEBUG, ("%.3f" % global_values.current_time) + ": No "+local_values.direction+" packet in queue, send dummy")

    if send_dummy:
        # send dummy packet
        local_values.total_dummy_bytes += args_defense.size_packets
        if not args.dont_save:
            global_values.f_out.write("%f\t%d\n" % (global_values.current_time, local_values.sign * args_defense.size_packets))

    args.logger.log(ALL, "Has sent "+str(local_values.total_real_bytes)+" B out of "+str(local_values.final_bytes)+" B")

    # check whether need to reprogram
    if local_values.total_real_bytes != local_values.final_bytes:
        # there are still packets to send
        reprogram = True
    else:
        # check whether have reached final time
        total_sent = local_values.total_real_bytes + local_values.total_dummy_bytes
        if local_values.stop_time == 0 :
            # first time we reach here, compute stopping time
            if args_defense.type in args_defense.stop_types["int_padding"]:
                local_values.stop_time = args_defense.padding_param*\
                                          (1+int(global_values.current_time/args_defense.padding_param))
                args.logger.log(DEBUG, ("%.3f" % global_values.current_time) + ": Has sent "+str(total_sent) +
                          " " + local_values.direction + ", will stop at time "+str(local_values.stop_time))
            elif args_defense.type in args_defense.stop_types["stream_padding"] and local_values.padding_scheme == "P":
                # total bytes is a multiple of 2^ceil(log_2(real_bytes))
                prop = math.pow(2,math.ceil(math.log(local_values.total_real_bytes, 2)))
                global_values.stop_bytes = prop*(1+math.floor(float(total_sent)/prop))
                local_values.stop_time = 1  # just so that we know stop_bytes has been computed
                args.logger.log(DEBUG, ("%.3f" % global_values.current_time) + ": Has sent %d B %s, stop after %d B" %
                          (total_sent, local_values.direction, global_values.stop_bytes))
        if args_defense.type in args_defense.stop_types["stream_padding"]:
            if args_defense.early_termination and global_values.one_is_done:
                reprogram = False
                args.logger.log(DEBUG, ("%.3f" % global_values.current_time) +
                          ": Early termination: stop now after "+str(total_sent)+" B")
            else:
                if local_values.padding_scheme == "P":
                    reprogram = (total_sent < global_values.stop_bytes)
                else:
                    # stop if total bytes is a power of 2
                    reprogram = (total_sent != math.pow(2, round(math.log(total_sent,2))))
                    if not reprogram:
                        args.logger.log(DEBUG, ("%.3f" % global_values.current_time) +
                              ": Has sent %d B, stop " % (total_sent))
        else:
            reprogram = (global_values.current_time < local_values.stop_time)

    if reprogram:
        local_values.next = global_values.current_time + local_values.interval
    else:
        global_values.one_is_done = True
        local_values.next = INF


def enqueue_packets_all(local_values_in, local_values_out, global_values, args, args_defense):
    next_time = global_values.current_time
    while next_time == global_values.current_time:
        next_time_in = enqueue_packets(local_values_in, global_values, args, args_defense)
        next_time_out = enqueue_packets(local_values_out, global_values, args, args_defense)
        next_time = min(next_time_out, next_time_in)

    return next_time


def enqueue_packets(local_values, global_values, args, args_defense):
    args.logger.log(VDEBUG, "%.3f: Enqueing %s packets" % (global_values.current_time, local_values.direction))
    # enqueue packets in real queue
    # can enqueue all packets in same direction, or packets in other direction if time has elapsed
    packet_ids_in_queue = [p[2] for p in local_values.queue_packets] # packets already enqueued
    if len(global_values.queue_packets) == 0 or global_values.queue_packets[-1][3] >= 0:
        # no more packets to send
        return INF

    if global_values.queue_packets[0][3] == -1:  # first one might not yet have been sent at beginning
        return INF

    c = 0
    first_packet_burst = global_values.queue_packets[0]

    queue_size = sum([abs(p[1]) for p in local_values.queue_packets])
    next_time = INF  # next possible time for packet to be sent without sending other packets
    for i in range(1, len(global_values.queue_packets)): # look at packets that can be enqueued
        packet = global_values.queue_packets[i]

        if packet[3] != -1 or packet[2] in packet_ids_in_queue:
            # already sent or enqueued
            args.logger.log(ALL, "\tPacket %s already sent or enqueued" % (repr(packet)))
            continue

        if np.sign(packet[1]) != np.sign(local_values.sign):
            # not direction of this queue
            args.logger.log(ALL, "\tPacket %s has wrong direction" % (repr(packet)))
            break

        real_delay = round(packet[0] - first_packet_burst[0], 5)
        current_delay = round(global_values.current_time - first_packet_burst[3], 5)
        if real_delay > current_delay:
            # real delay between this packet (and all next ones) is larger than delay after sending: break
            args.logger.log(ALL, "\tPacket %s not yet arrived" % (repr(packet)))
            next_time = real_delay + first_packet_burst[3]
            break

        # check not waiting for other packets in other direction
        j=1
        other_dir = False
        while j < i:
            if global_values.queue_packets[j][3] == -1 \
                    and np.sign(global_values.queue_packets[j][1]) != np.sign(packet[1]):
                other_dir = True
                args.logger.log(ALL, "\tFor %s, waiting for packets in other direction" % (repr(packet)))
                break
            j += 1
        if other_dir:
            break

        # this packet can be enqueued
        # compute time at which it is enqueued (time last packet sent + real time diff)
        time_enqueue = round(first_packet_burst[3], 5) + real_delay
        time_enqueue = max(time_enqueue, global_values.current_time)  # due to rounding errors, is sometimes smaller
        local_packet = (packet[0], packet[1], packet[2], time_enqueue)
        local_values.queue_packets.append(local_packet)
        global_values.sent_packets[packet[2]] = (time_enqueue, 0, packet[1])
        queue_size += abs(packet[1])

        args.logger.log(VDEBUG, "\tEnqueuing "+local_values.direction+" packet "+repr(local_packet))

        if local_values.rate == 0 or (args_defense.type == "H" and queue_size > local_values.size_queue):
            # send immediately
            if args_defense.type == "H":
                dequeue_packet(local_values, global_values, args, on_sec_channel=True, t_sent=time_enqueue)
            else:
                send_packet(local_values, global_values, args, args_defense)
            # packet has been sent, must start again
            #print(abs(global_values.current_time - time_enqueue))
            assert abs(global_values.current_time - time_enqueue) < math.pow(10,-3)
            global_values.current_time = time_enqueue  # due to rounding errors, they are sometimes different

            return global_values.current_time

    args.logger.log(VDEBUG, "\tDone enqueuing packets")

    return next_time


def initiate_local_values(direction, final_bytes, args_defense):
    local_values = types.SimpleNamespace()

    # local queues with packets ready to be sent
    local_values.queue_packets = []

    # get interval
    if direction == IN:
        local_values.rate = args_defense.rate_in
        local_values.direction = DIR_NAMES[IN]
        local_values.max_delay = args_defense.delay_in
        if args_defense.type in args_defense.stop_types["stream_padding"]:
            local_values.padding_scheme = args_defense.padding_scheme_in
    else:
        local_values.rate = args_defense.rate_out
        local_values.direction = DIR_NAMES[OUT]
        local_values.max_delay = args_defense.delay_out
        if args_defense.type in args_defense.stop_types["stream_padding"]:
            local_values.padding_scheme = args_defense.padding_scheme_out

    if type(local_values.rate) == int:
        if local_values.rate > 0:
            local_values.interval = float(args_defense.size_packets) / local_values.rate
        else:
            local_values.interval = INF
    else:
        # CS-BuFLO rate adaptation
        local_values.interval = float(1/8)  # initial interval
        local_values.k = 1  # initial k (estimate interval at every 2^k)
        local_values.intervals = []
        local_values.interval_stats = [local_values.interval]

    if local_values.max_delay == INF:
        local_values.size_queue = INF
    else:
        rate = float(args_defense.size_packets) / local_values.interval
        local_values.size_queue = max(1, int(rate * local_values.max_delay))

    if args_defense.type in args_defense.stop_types["global_stop"]:
        local_values.stop_time = args_defense.time
    else:
        local_values.stop_time = 0  # stop time will be defined when no more packets are sent

    local_values.next = 0
    local_values.final_bytes = final_bytes
    local_values.total_real_bytes = 0
    local_values.total_dummy_bytes = 0
    local_values.total_sec_channel_bytes = 0
    local_values.sign = direction
    local_values.queue_sizes = []
    local_values.last_enqueued_time = -1
    local_values.last_packet = (-1,-1,-1,-1)

    return local_values


def get_cs_interval(local_values, args_defense):
    # CS-BuFLO algorithm
    if local_values.total_real_bytes == 0:
        return
    current_k = int(math.log(local_values.total_real_bytes, 2))
    if current_k > local_values.k:
        if len(local_values.intervals) > 0:
            # if less than 5 values, keep previous
            med = np.mean(local_values.intervals)  # CS-BuFLO uses median, but it seems to give very bad overheads
            print(med)
            if local_values.rate == "cs":
                if med < math.pow(10, -4):
                    med = math.pow(10, -4)  # min interval
                local_values.interval = math.pow(2, int(round(math.log(med, 2))))
            elif local_values.rate == "cs2":
                if med < local_values.interval:
                    local_values.interval /= 2
                elif med > local_values.interval:
                    local_values.interval *= 2
            else:
                print("Unknown cs algorithm")
                return
        local_values.k = current_k
        local_values.interval_stats.append(local_values.interval)
        if local_values.max_delay == INF:
            local_values.size_queue = INF
        else:
            rate = float(args_defense.size_packets) / local_values.interval
            local_values.size_queue = max(1, int(rate * local_values.max_delay))
        if args_defense.verbose <= DEBUG:
            very_verbose_str = ""
            if args_defense.verbose <= VDEBUG:
                very_verbose_str = "\n\n\n"
            queue_size_str = ""
            if args_defense.type == "H":
                queue_size_str = " and queue size to "+str(local_values.size_queue)
                args_defense.logger.log(DEBUG, "%sAfter %d B, setting %s interval to %.4f %s, computed on %d values%s" %
                  (very_verbose_str, local_values.total_real_bytes, local_values.direction, local_values.interval,
                   queue_size_str, len(local_values.intervals), very_verbose_str))

        # with cs, overheads seem to be much lower when not reset
        #if local_values.rate == "cs2":
        #    local_values.intervals = []


def create_repertories(out_name, args):
    last_slash = out_name.rfind("/")
    if not args.dont_save:
        if last_slash != -1:
            subdir_out = out_name[:last_slash]
            if not os.path.exists(subdir_out):
                args.logger.log(INFO, "Creating "+subdir_out)
                os.makedirs(subdir_out)


def transform_file(file_name, out_name, out_name_sec, args, args_defense):

    if out_name is None:
        args.logger.log(DEBUG, "Transforming " + file_name)
    else:
        args.logger.log(DEBUG, "Transforming " + file_name + ", writing to " + out_name)

        create_repertories(out_name, args)

    if args.only_create:
        return {}

    f_in = open(file_name, "r")
    if not args.dont_save and out_name is not None:
        f_out = open(out_name, "w")
    else:
        f_out = None

    # read real packets and queue them in global queue as (timestamp,size,id,time_sent)
    lines = f_in.readlines()
    nb_real_packets = len(lines)

    if nb_real_packets == 0:
        return {}

    queue_packets = [(0,0,0,0)]*nb_real_packets
    i = 0
    first_packet = (-1, -1, -1, -1)
    final_in = 0
    final_out = 0
    for line in lines:
        line = line.replace("\n", "")
        toks = re.split("[\t ]", line)
        timestamp = float(toks[0])
        size = int(float(toks[1]))
        if abs(size) == 1:
            # if packets already have fixed size, assume we use this size (keep direction given by sign)
            size *= args_defense.size_packets
        if size < 0:
            final_in += abs(size)
        else:
            final_out += size
        queue_packets[i] = (timestamp, size, i, -1)
        if first_packet[3] == -1:
            first_packet = (timestamp, size, i, timestamp)
        args.logger.log(ALL, "Real %d packet at %f of size %d" % (i, timestamp, size))
        i += 1

    args.logger.log(DEBUG, "Has %d real packets, in: %d B, out: %d B" % (nb_real_packets, final_in, final_out))

    max_time = queue_packets[-1][0]

    global_values = types.SimpleNamespace()
    global_values.f_out = f_out
    global_values.queue_packets = queue_packets # queue used for enqueuing packets in local (real) queue
    global_values.current_time = 0
    global_values.real_time_last_packet = 0
    global_values.defense_time_last_packet = 0
    global_values.delays = []
    global_values.sent_packets = [(0, 0, 0)]*nb_real_packets  # sent packets (ts_enq, ts_sent, size)
    global_values.one_is_done = False

    local_values_in = initiate_local_values(IN, final_in, args_defense)
    local_values_out = initiate_local_values(OUT, final_out, args_defense)

    if first_packet[1] > 0: # first packet is outgoing
        local_values_out.queue_packets.append(first_packet)
    else:
        local_values_in.queue_packets.append(first_packet)

    # continue until there are packets to send or we did not reach stop time
    # difference between consecutive real packets after defense must stay larger than real one
    end_time = 0
    while global_values.current_time < INF:
        if type(local_values_in.rate) is str and "cs" in local_values_in.rate:
            get_cs_interval(local_values_in, args_defense)
        if type(local_values_out.rate) is str and "cs" in local_values_out.rate:
            get_cs_interval(local_values_out, args_defense)
        enqueue_packets_all(local_values_in, local_values_out, global_values, args, args_defense)
        if global_values.current_time == local_values_in.next:
            #enqueue_packets(local_values_in, global_values)
            send_packet(local_values_in, global_values, args, args_defense)
        enqueue_packets_all(local_values_in, local_values_out, global_values, args, args_defense)
        if global_values.current_time == local_values_out.next:
            #enqueue_packets(local_values_out, global_values)
            send_packet(local_values_out, global_values, args, args_defense)
        next_time = enqueue_packets_all(local_values_in, local_values_out, global_values, args, args_defense)

        end_time = global_values.current_time
        global_values.current_time = min([local_values_in.next, local_values_out.next, next_time])

        args.logger.log(ALL, "Reprogramming at "+str(global_values.current_time))

    if len(global_values.delays) != nb_real_packets:
        print("WARNING: has %d delays and %d packets" % (len(global_values.delays), nb_real_packets))

    f_in.close()
    if not args.dont_save:
        f_out.close()

    results = {}

    results["real_bytes_in"] = local_values_in.total_real_bytes
    results["real_bytes_out"] = local_values_out.total_real_bytes
    results["dummy_bytes_in"] = local_values_in.total_dummy_bytes
    results["dummy_bytes_out"] = local_values_out.total_dummy_bytes
    results["avg_delay_overhead"] = np.mean(global_values.delays)
    if len(local_values_in.queue_sizes) > 0:
        results["avg_queue_size_in"] = np.mean(local_values_in.queue_sizes)
    else:
        results["avg_queue_size_in"] = 0
    if len(local_values_out.queue_sizes) > 0:
        results["avg_queue_size_out"] = np.mean(local_values_out.queue_sizes)
    else:
        results["avg_queue_size_out"] = 0
    results["max_time_after"] = end_time
    results["max_time_before"] = max_time
    if args_defense.type == "H":
        results["sec_channel_bytes_in"] = local_values_in.total_sec_channel_bytes
        results["sec_channel_bytes_out"] = local_values_out.total_sec_channel_bytes

    args.logger.log(DEBUG, "Has sent in:%d, out:%d real bytes; has sent in:%d, out:%d dummy bytes " % \
              (local_values_in.total_real_bytes, local_values_out.total_real_bytes,
               local_values_in.total_dummy_bytes, local_values_out.total_dummy_bytes))
    if len(global_values.delays) > 0: # should always be
        args.logger.log(VDEBUG, "Overhead average delay is %.3f, took %.3f more" %
                  (float(np.mean(global_values.delays)), global_values.current_time - max_time))

    return results


def pad(f_out, total_bytes, pad_param, size, interval_dist, last_time):
    sequence = [200, 800, 1600, 3000]
    pad_bytes = 0
    #random.seed(args.arg_str)
    can_stop = False
    while not can_stop:
        interval = interval_dist[random.randint(0, len(interval_dist)-1)]
        timestamp = last_time + interval
        if f_out:
            f_out.write("%f\t%d\n" % (timestamp, size))
        last_time = timestamp
        pad_bytes += abs(size)
        total_bytes += abs(size)
        args.logger.log(ALL, "After padding with packet of size %d, total_bytes=%d, time=%f " % (size, total_bytes, last_time))
        if type(pad_param) == int:
            # stop when multiple of pad_param
            can_stop = (total_bytes % pad_param) == 0
        elif pad_param == "pow":
            # stop when total bytes is a power of 2
            can_stop = total_bytes == math.pow(2, round(math.log(total_bytes,2)))
        elif pad_param == "seq":
            # stop based on sequence list (or multiple of last)
            can_stop = total_bytes in sequence or total_bytes % sequence[-1] == 0

    return pad_bytes, last_time


def get_local_p(p, s):
    if p == 0 or p == 1 or s == 0:
        loc_p = p
    else:
        if args_defense.prob_type == "uniform":
            s = round(min([s, p, 1 - p]), 2)

            loc_p = p - s + 2*s*np.random.uniform()
        elif args_defense.prob_type == "normal":
            loc_p = np.random.normal(p, s)
        else:
            # use fixed
            loc_p = p

    return min(1,max(0,loc_p))  # between 0 and 1


def transform_file_probabilistic(file_name, out_name, out_name_sec, args):

    if out_name is None:
        args.logger.log(DEBUG, "Transforming " + file_name)
    else:
        args.logger.log(DEBUG, "Transforming " + file_name + ", writing to " + out_name)

        create_repertories(out_name, args)

    if args.only_create:
        return {}

    f_in = open(file_name, "r")
    if not args.dont_save:
        f_out = open(out_name, "w")
    else:
        f_out = None

    lines = f_in.readlines()
    nb_real_packets = len(lines)

    if nb_real_packets == 0:
        return {}

    real_bytes = [0, 0]
    dummy_bytes = [0, 0]
    sec_channel_bytes = [0, 0]
    # compute interval distribution to use in padding
    interval_dist_in = []
    interval_dist_out = []
    last_in = -1
    last_out = -1
    timestamp = -1
    p_in_inst = get_local_p(args_defense.p_in, args_defense.prob_std_in)
    p_out_inst = get_local_p(args_defense.p_out, args_defense.prob_std_out)
    args.logger.log(DEBUG, "Instance probabilities are in:%f, out:%f" % (p_in_inst, p_out_inst))
    p_in = get_local_p(p_in_inst, args_defense.prob_std_in)
    p_out = get_local_p(p_out_inst, args_defense.prob_std_out)
    args.logger.log(DEBUG, "New probabilities are in:%f, out:%f" % (p_in, p_out))

    if args_defense.packet_prob_changes > 0:
        num_packets_before_prob_change = int(math.ceil(get_exp(args_defense.packet_prob_changes, args_defense.change_dist)))
        time_prob_changes = 0
        num_packets = 0
        next_stop = 0
    elif args_defense.time_prob_changes > 0:
        num_packets = 0
        num_packets_before_prob_change = 0
        time_prob_changes = args_defense.time_prob_changes
        next_stop = get_exp(time_prob_changes, args_defense.change_dist)
    else:
        num_packets_before_prob_change = 0
        time_prob_changes = 0
        num_packets = 0
        next_stop = 0

    avg_consecutive_packets = {-1: args_defense.consecutive_packets_in, 1:args_defense.consecutive_packets_out}
    consecutive_packets = {-1: int(math.ceil(get_exp(avg_consecutive_packets[-1], args_defense.num_conspack_dist))),
                           1: int(math.ceil(get_exp(avg_consecutive_packets[1], args_defense.num_conspack_dist)))}
    current_cons_packets = {-1: consecutive_packets[-1], 1:consecutive_packets[1]}
    current_is_sec = {-1:False, 1:False}

    for line in lines:
        toks = re.split("[\t ]", line)
        timestamp = float(toks[0])
        size = int(toks[1])
        # check whether time to change probability
        if num_packets == num_packets_before_prob_change > 0:
            num_packets = 0
            num_packets_before_prob_change = int(math.ceil(get_exp(args_defense.packet_prob_changes,
                                                               args_defense.change_dist)))
            p_in = get_local_p(p_in_inst, args_defense.prob_std_in)
            p_out = get_local_p(p_out_inst, args_defense.prob_std_out)
            args.logger.log(DEBUG, "New probabilities are in:%f, out:%f" % (p_in, p_out))
        elif time_prob_changes > 0 and timestamp >= next_stop:
            next_stop += get_exp(time_prob_changes, args_defense.change_dist)
            p_in = get_local_p(p_in_inst, args_defense.prob_std_in)
            p_out = get_local_p(p_out_inst, args_defense.prob_std_out)
            args.logger.log(DEBUG, "New probabilities are in:%f, out:%f" % (p_in, p_out))

        if size < 0:
            p = p_in
            if last_in != -1:
                interval_dist_in.append(timestamp-last_in)
            last_in = timestamp
        else:
            p = p_out
            if last_out != -1:
                interval_dist_out.append(timestamp-last_out)
                last_out = timestamp
        real_bytes[int(size > 0)] += abs(size)

        # check whether to send on secure channel (must send consecutive_packets on same channel)
        if current_cons_packets[np.sign(size)] < consecutive_packets[np.sign(size)]:
            current_cons_packets[np.sign(size)] += 1
            send_on_sec = current_is_sec[np.sign(size)]
        else:
            consecutive_packets[np.sign(size)] = \
                int(math.ceil(get_exp(avg_consecutive_packets[np.sign(size)], args_defense.num_conspack_dist)))
            current_cons_packets[np.sign(size)] = 1
            if random.random() <= p:
                send_on_sec = False
            else:
                send_on_sec = True
            current_is_sec[np.sign(size)] = send_on_sec
        if not send_on_sec:
            if f_out:
                f_out.write(line)
        elif args_defense.transform_in_out:
            if f_out:
                f_out.write("%f\t%d\n" % (timestamp, IN * size))
            dummy_bytes[int(size < 0)] += abs(size)
            sec_channel_bytes[int(size > 0)] += abs(size)
        else:
            sec_channel_bytes[int(size > 0)] += abs(size)

    end_time_before_pad = timestamp

    if (type(args_defense.pad_in) == str or args_defense.pad_in > 0) and len(interval_dist_in) > 0:
        pad_bytes, end_time_in = pad(f_out, real_bytes[0]+dummy_bytes[0], args_defense.pad_in, IN*args_defense.size_packets,
                              interval_dist_in, last_in)
        dummy_bytes[0] += pad_bytes
    else:
        end_time_in = timestamp
    if (type(args_defense.pad_out) == str or args_defense.pad_out > 0) and len(interval_dist_out) > 0:
        pad_bytes, end_time_out = pad(f_out, real_bytes[1]+dummy_bytes[1], args_defense.pad_out, OUT*args_defense.size_packets,
                              interval_dist_out, last_out)
        dummy_bytes[1] += pad_bytes
    else:
        end_time_out = timestamp

    end_time_after_pad = max(end_time_in, end_time_out)

    results = {}

    results["real_bytes_in"] = real_bytes[0]
    results["real_bytes_out"] = real_bytes[1]
    results["dummy_bytes_in"] = dummy_bytes[0]
    results["dummy_bytes_out"] = dummy_bytes[1]
    results["avg_delay_overhead"] = 0
    results["avg_queue_size_in"] = 0
    results["avg_queue_size_out"] = 0
    results["max_time_after"] = end_time_after_pad
    results["max_time_before"] = end_time_before_pad
    if args_defense.type == "H":
        results["sec_channel_bytes_in"] = sec_channel_bytes[0]
        results["sec_channel_bytes_out"] = sec_channel_bytes[1]

    if f_out:
        f_out.close()

    return results


def transform_file_wtfpad(file_name, out_name, out_name_sec, args):

    if out_name is None:
        args.logger.log(DEBUG, "Transforming " + file_name)
    else:
        args.logger.log(DEBUG, "Transforming " + file_name + ", writing to " + out_name + " sec to " + out_name_sec)

        create_repertories(out_name, args)
        if args_defense.type == "H":
            create_repertories(out_name_sec, args)

    if args.only_create:
        return {}

    results = {}

    if args_defense.refill:
        args_defense.wtfpad.reset_all_infinity_bins()

    p_in_inst = get_local_p(args_defense.p_in, args_defense.prob_std_in)
    p_out_inst = get_local_p(args_defense.p_out, args_defense.prob_std_out)
    args.logger.log(DEBUG, "Instance probabilities are in:%f, out:%f" % (p_in_inst, p_out_inst))

    initial_trace = prs.parse(file_name).remove_first_incoming()
    initial_trace.update_times()
    total_size_in = initial_trace.size_dir(IN)
    total_size_out = initial_trace.size_dir(OUT)
    total_size = total_size_out+total_size_in
    args_defense.wtfpad.current_burst = 0
    args_defense.wtfpad.burst_lengths = []

    if args_defense.packet_prob_changes > 0:
        num_packets_before_prob_change_in = int(math.ceil(get_exp(args_defense.packet_prob_changes, args_defense.change_dist)))
        num_packets_before_prob_change_out = int(math.ceil(get_exp(args_defense.packet_prob_changes, args_defense.change_dist)))
        time_prob_changes_in = 0
        time_prob_changes_out = 0
    elif args_defense.time_prob_changes > 0:
        num_packets_before_prob_change_in = 0
        num_packets_before_prob_change_out = 0
        time_prob_changes_in = get_exp(args_defense.time_prob_changes, args_defense.change_dist)
        time_prob_changes_out = get_exp(args_defense.time_prob_changes, args_defense.change_dist)
    else:
        num_packets_before_prob_change_in = total_size
        num_packets_before_prob_change_out = total_size
        time_prob_changes_in = 0
        time_prob_changes_out = 0

    if args_defense.type == "H":
        prob_changes_in = []
        prob_changes_out = []
        # choose the packets to be sent on secure channel
        if args_defense.pad_before:
            # pad with all traffic, split after
            trace_in = initial_trace.extract_dir(IN)
            trace_out = initial_trace.extract_dir(OUT)
            results["sec_channel_bytes_in"] = 0
            results["sec_channel_bytes_out"] = 0
        else:
            # split, then pad
            trace_in = prs.Trace()
            trace_out = prs.Trace()
            to_secure_in = prs.Trace()
            to_secure_out = prs.Trace()
            #trace_to_modify = prs.Trace(initial_trace)
            trace_to_modify_in = initial_trace.extract_dir(IN)
            trace_to_modify_out = initial_trace.extract_dir(OUT)
            while len(trace_to_modify_in) > 0 or len(trace_to_modify_out) > 0:

                p_in = get_local_p(p_in_inst, args_defense.prob_std_in)
                p_out = get_local_p(p_out_inst, args_defense.prob_std_out)

                # store packets when probability is changed
                if len(trace_to_modify_in) > 0:
                    prob_changes_in.append((trace_to_modify_in[0], p_in))
                if len(trace_to_modify_out) > 0:
                    prob_changes_out.append((trace_to_modify_out[0], p_out))

                args.logger.log(DEBUG, "New probabilities are in:%f, out:%f" % (p_in, p_out))

                if args_defense.packet_prob_changes > 0:
                    removed_in = trace_to_modify_in.remove_first_packets(num_packets_before_prob_change_in)
                    removed_out = trace_to_modify_out.remove_first_packets(num_packets_before_prob_change_out)
                    num_packets_before_prob_change_in = int(math.ceil(get_exp(args_defense.packet_prob_changes, args_defense.change_dist)))
                    num_packets_before_prob_change_out = int(math.ceil(get_exp(args_defense.packet_prob_changes, args_defense.change_dist)))
                elif args_defense.time_prob_changes > 0:
                    removed_in = trace_to_modify_in.remove_first_time(time_prob_changes_in)
                    removed_out = trace_to_modify_out.remove_first_time(time_prob_changes_out)
                    time_prob_changes_in += get_exp(args_defense.time_prob_changes, args_defense.change_dist)
                    time_prob_changes_out += get_exp(args_defense.time_prob_changes, args_defense.change_dist)
                else:
                    removed_in = trace_to_modify_in.remove_all()
                    removed_out = trace_to_modify_out.remove_all()

                trace_in_loc, to_secure_in_loc = removed_in.extract_prob(
                    p_in, do_not_remove=args_defense.all_sec, cons_pack=args_defense.consecutive_packets_in,
                    cons_dist=args_defense.num_conspack_dist)

                trace_out_loc, to_secure_out_loc = removed_out.extract_prob(
                    p_out, do_not_remove=args_defense.all_sec, cons_pack=args_defense.consecutive_packets_out,
                    cons_dist=args_defense.num_conspack_dist)

                trace_in.extend_trace(trace_in_loc)
                trace_out.extend_trace(trace_out_loc)
                to_secure_in.extend_trace(to_secure_in_loc)
                to_secure_out.extend_trace(to_secure_out_loc)

            if args_defense.all_sec:
                results["sec_channel_bytes_in"] = 0
            else:
                results["sec_channel_bytes_in"] = to_secure_in.total_size()
            if args_defense.all_sec:
                results["sec_channel_bytes_out"] = 0
            else:
                results["sec_channel_bytes_out"] = to_secure_out.total_size()

        trace_in.extend_trace(trace_out)

        if args_defense.tune_hybrid:
            # prob changes is used to tune factor of interarrivals
            prob_changes = {IN: prob_changes_in, OUT: prob_changes_out}
        else:
            prob_changes = {}


        wtfpad_trace, to_secure_sim = args_defense.wtfpad.simulate(prs.Trace(trace_in), prob_changes=prob_changes)

        if args_defense.pad_before:
            trace_to_modify_in = wtfpad_trace.extract_dir(IN)
            trace_to_modify_out = wtfpad_trace.extract_dir(OUT)
            wtfpad_trace = prs.Trace()
            while len(trace_to_modify_in) > 0 or len(trace_to_modify_out) > 0:
                p_in = get_local_p(p_in_inst, args_defense.prob_std_in)
                p_out = get_local_p(p_out_inst, args_defense.prob_std_out)
                args.logger.log(DEBUG, "New probabilities are in:%f, out:%f" % (p_in, p_out))

                if args_defense.packet_prob_changes > 0:
                    removed_in = trace_to_modify_in.remove_first_packets(num_packets_before_prob_change_in)
                    removed_out = trace_to_modify_out.remove_first_packets(num_packets_before_prob_change_out)
                    num_packets_before_prob_change_in = int(math.ceil(get_exp(args_defense.packet_prob_changes, args_defense.change_dist)))
                    num_packets_before_prob_change_out = int(math.ceil(get_exp(args_defense.packet_prob_changes, args_defense.change_dist)))
                elif args_defense.time_prob_changes > 0:
                    removed_in = trace_to_modify_in.remove_first_time(time_prob_changes_in)
                    removed_out = trace_to_modify_out.remove_first_time(time_prob_changes_out)
                    time_prob_changes_in += get_exp(args_defense.time_prob_changes, args_defense.change_dist)
                    time_prob_changes_out += get_exp(args_defense.time_prob_changes, args_defense.change_dist)
                else:
                    removed_in = trace_to_modify_in.remove_all()
                    removed_out = trace_to_modify_out.remove_all()

                # remove random proportion of packet (sent to other channel)
                real_packets_in, dummy_packets_in_unsec = removed_in.separate_packets()
                real_packets_out, dummy_packets_out_unsec = removed_out.separate_packets()
                real_packets_in_unsec, real_packets_in_sec = real_packets_in.extract_prob(
                    p_in, cons_pack=args_defense.consecutive_packets_in, cons_dist=args_defense.num_conspack_dist)
                real_packets_out_unsec, real_packets_out_sec = real_packets_out.extract_prob(
                    p_out, cons_pack=args_defense.consecutive_packets_out, cons_dist=args_defense.num_conspack_dist)
                dummy_packets_in_sec = []
                dummy_packets_out_sec = []
                if not args_defense.all_sec or args_defense.pad_secure:
                    dummy_packets_in_unsec, dummy_packets_in_sec = dummy_packets_in_unsec.extract_prob(
                        p_in, cons_pack=args_defense.consecutive_packets_in, cons_dist=args_defense.num_conspack_dist)
                    dummy_packets_out_unsec, dummy_packets_out_sec = dummy_packets_out_unsec.extract_prob(
                        p_out, cons_pack=args_defense.consecutive_packets_out, cons_dist=args_defense.num_conspack_dist)
                to_secure_sim.extend_trace(real_packets_in_sec)
                to_secure_sim.extend_trace(real_packets_out_sec)
                wtfpad_trace.extend_trace(real_packets_in_unsec)
                wtfpad_trace.extend_trace(real_packets_out_unsec)
                wtfpad_trace.extend_trace(dummy_packets_in_unsec)
                wtfpad_trace.extend_trace(dummy_packets_out_unsec)
                if args_defense.pad_secure:
                    to_secure_sim.extend_trace(dummy_packets_in_sec)
                    to_secure_sim.extend_trace(dummy_packets_out_sec)

        results["sec_channel_bytes_in"] += to_secure_sim.size_dir(IN)
        results["sec_channel_bytes_out"] += to_secure_sim.size_dir(OUT)
    else:
        wtfpad_trace, to_secure_sim = args_defense.wtfpad.simulate(prs.Trace(initial_trace))

    if not args.dont_save:
        prs.dump(wtfpad_trace, out_name)
        if args_defense.type == "H":
            prs.dump(to_secure_sim, out_name_sec)


    if len(args_defense.wtfpad.flows[IN].burst_lengths) > 0:
        results["mean_burst_length_in"] = np.mean(args_defense.wtfpad.flows[IN].burst_lengths)
    else:
        results["mean_burst_length_in"] = 0
    if len(args_defense.wtfpad.flows[OUT].burst_lengths) > 0:
        results["mean_burst_length_out"] = np.mean(args_defense.wtfpad.flows[OUT].burst_lengths)
    else:
        results["mean_burst_length_out"] = 0

    results["real_bytes_in"] = total_size_in
    results["real_bytes_out"] = total_size_out
    results["dummy_bytes_in"] = wtfpad_trace.size_dir(IN) - initial_trace.size_dir(IN)
    if args_defense.type == "H":
        results["dummy_bytes_in"] = wtfpad_trace.size_dir(IN) + results["sec_channel_bytes_in"] - initial_trace.size_dir(IN)
    results["dummy_bytes_out"] = wtfpad_trace.size_dir(OUT) - initial_trace.size_dir(OUT)
    if args_defense.type == "H":
        results["dummy_bytes_out"] = wtfpad_trace.size_dir(OUT) + results["sec_channel_bytes_out"] - initial_trace.size_dir(OUT)
    results["avg_delay_overhead"] = 0
    results["avg_queue_size_in"] = 0
    results["avg_queue_size_out"] = 0
    ind_last_real = wtfpad_trace.last_real()
    if ind_last_real >= 0:
        results["max_time_after"] = max(initial_trace.max_time(),wtfpad_trace[ind_last_real].timestamp)  # time of last real packet
    else:  # can happen in very rare cases
        results["max_time_after"] = initial_trace.max_time()
    results["max_time_before"] = initial_trace.max_time()
    extracted_trace = wtfpad_trace.extract_after(ind_last_real)
    results["dummy_bytes_after_in"] = extracted_trace.size_dir(IN)
    results["dummy_bytes_after_out"] = extracted_trace.size_dir(OUT)

    return results


def defenses(args, args_defense):

    all_files = []
    for root, dirs, files in os.walk(args.in_directory):
        files_loc = [root+"/"+file_name for file_name in files if re.match(args.arg, file_name) and file_name[0] != "."
                     and (args.max_inst == -1 or args.delimiter not in file_name or
                          int(file_name.split(args.delimiter)[1].split(".")[0]) <= args.max_inst)
                     and (args.equal_arg == "" or file_name == args.equal_arg)]
        all_files.extend(files_loc)

    if len(all_files) == 0:
        return {}


    results = {"f_names": []}

    for count_exp in range(args.num_times):
        count_files = 1
        for file_name in all_files:
            args.logger.log(DEBUG, "Experiment for file %s", file_name)
            if not args.dont_save:
                out_name = file_name.replace(args.in_directory, args.out_directory)
                if ".cell" in out_name:
                    out_name = out_name.replace(".cell", ","+str(count_exp+args.first_index)+".cell")
                else:
                    out_name = out_name+","+str(count_exp+args.first_index)
                # write arguments
                last_slash = out_name.rfind("/")
                out_name = out_name[:last_slash]+"/"+args.arg_str+out_name[last_slash:]
                last_slash = out_name.rfind("/")
                out_name_sec = out_name[:last_slash]+"_sec/"+out_name[last_slash:]
            else:
                out_name = None
                out_name_sec = None
            if args.dont_remove and out_name is not None and os.path.exists(out_name):
                # file already exists, continue
                continue

            if args_defense.type in args_defense.stop_types["wtfpad"]:
                results_loc = transform_file_wtfpad(file_name, out_name, out_name_sec, args)
            elif args_defense.probabilistic:
                results_loc = transform_file_probabilistic(file_name, out_name, out_name_sec, args)
            else:
                results_loc = transform_file(file_name, out_name, out_name_sec, args, args_defense)
            if args.only_create:
                return {}
            for key in results_loc:
                if key not in results:
                    results[key] = []
                if isinstance(results_loc[key], list):
                    results[key].extend(results_loc[key])
                else:
                    results[key].append(results_loc[key])
            if len(results_loc) > 0:
                results["f_names"].append(file_name)

            if count_files % 10 == 0:
                if args.verbose == INFO:
                    sys.stdout.write("\r------ Have done %d out of %d -------" % (count_files, len(all_files)))
                    sys.stdout.flush()
                elif args.verbose < INFO:
                    print("------ Have done %d out of %d -------" % (count_files, len(all_files)))
            count_files += 1

        if args.verbose == INFO:
            sys.stdout.write("\nHave done exp #%d\n" % count_exp)
            sys.stdout.flush()

    if len(results["f_names"]) == 0:
        return {}

    return results


def get_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-id', '--in_directory', required=True, help="Directory with files of original sequences. "
                                                                     "Consider recursively all files with arg in name.")
    parser.add_argument('-od', '--out_directory', help="Directory in which modified files will be saved.")
    parser.add_argument('-o', '--options', nargs='+', help="options overriding option file (name_option=value_option)")
    parser.add_argument('-a', '--arg', default="", help="Modify only files with this regular expression in file name")
    parser.add_argument('-ea', '--equal_arg', default="", help="Modify only this file name")
    parser.add_argument('-t', '--type', required=True, help="Define method to use. Can be:  B (BuFLO), T (Tamaraw), "
                                                            "H (Hybrid), C (CS-BuFLO), W (WTF-PAD)")
    parser.add_argument('-v', '--verbose', default="INFO", help="Verbosity level (NONE, INFO, DEBUG, VDEBUG, ALL)")
    parser.add_argument('-c', '--config', help="WTF-PAD configuration.", default="normal_rcv")
    parser.add_argument('-n', '--num_times', type=int, default=1, help="Number of times each file is defended.")
    parser.add_argument('-fi', '--first_index', type=int, default=0, help="First index of files.")
    parser.add_argument('-ds', '--dont_save', action="store_true", help="Don't save in files")
    parser.add_argument('-dr', '--dont_remove', action="store_true", help="Don't do experiment if file already exists")
    parser.add_argument('--old', action="store_true", help="Use original code")
    parser.add_argument('--only_create', action="store_true", help="Only create folders")
    parser.add_argument('-m', '--max_inst', type=int, default=-1, help="Max instance number")

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    # logging
    logging.addLevelName(NONE, "NONE")
    logging.addLevelName(INFO, "INFO")
    logging.addLevelName(DEBUG, "DEBUG")
    logging.addLevelName(VDEBUG, "VDEBUG")
    logging.addLevelName(ALL, "ALL")
    logger = logging.getLogger('defenses')
    ch = logging.StreamHandler(sys.stdout)
    # Set logging format
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)
    # Set level format
    logger.setLevel(args.verbose)
    args.verbose = DICT_LOGS[args.verbose]
    args.logger = logger

    # default options
    args_defense = types.SimpleNamespace()
    args_defense.type = args.type
    args_defense.rate_out = int(1e3)  # rate at which outgoing packets are sent (in bytes per milliseconds)
    args_defense.rate_in = int(1e3)  # rate at which ingoing packets are sent (in bytes per milliseconds)
    args_defense.size_packets = 1  # size of packets (in bytes)
    args_defense.time = 10  # with BuFLO: time until which packets are sent (in milliseconds)
    args_defense.padding_param = 5  # with Tamaraw or hybrid: L parameter, in time unit
    args_defense.delay_in = 0  # with Hybrid: maximum tolerated delay, used to compute queue size
    args_defense.delay_out = 0  # with Hybrid: maximum tolerated delay, used to compute queue size
    args_defense.padding_scheme_in = "P"  # padding scheme with CS-BuFLO; can be P (payload) or T (total)
    args_defense.padding_scheme_out = "P"  # padding scheme with CS-BuFLO; can be P (payload) or T (total)
    args_defense.early_termination = True  # if early termination with CS-BuFLO
    args_defense.probabilistic = False
    args_defense.prob_std_in = 1  # std info for non fixed p
    args_defense.prob_std_out = 1
    args_defense.prob_type = "uniform"  # can be fixed, uniform (between p \pm prob_std), normal (mean p, std prob_std)
    args_defense.p_in = 0.5
    args_defense.p_out = 0.5
    args_defense.packet_prob_changes = 0
    args_defense.time_prob_changes = 0
    args_defense.change_dist = "exp"
    args_defense.consecutive_packets_in = 20
    args_defense.consecutive_packets_out = 20
    args_defense.num_conspack_dist = "exp"
    args_defense.pad_before = True
    args_defense.all_sec = False
    args_defense.transform_in_out = False  # if True, replace each outgoing packet sent on secure channel with incoming
    args_defense.pad_in = 0
    args_defense.pad_out = 0
    args_defense.refill = False
    args_defense.stop_on_real = True
    args_defense.percentile = 0.5
    args_defense.percentile_burst = 0
    args_defense.thr_burst = 0
    args_defense.prob_burst = 0.9
    args_defense.pad_secure = False
    args_defense.thr = 0  # threshold for hybrid scheme (see is_biased of histograms.py)
    args_defense.tune_hybrid = False
    args_defense.stop_types = {"no_pad": [], "global_stop": ["B"], "int_padding": ["T"], "stream_padding": ["C"],
                               "wtfpad": ["W", "H"]}
    args_defense.logger = logger

    if args.in_directory[-1] != "/":
        args.in_directory += "/"
    if args.out_directory is not None:
        if args.out_directory[-1] != "/":
            args.out_directory += "/"

    # overrides default options with arguments
    if args.options is not None:

        add_options = []
        for i in range(0,len(args.options)):
            opt_tok = args.options[i].split('=')
            if opt_tok[0] == "rate" or opt_tok[0] == "delay" or opt_tok[0] == "padding_scheme":
                # both in an out
                args.options[i] = opt_tok[0]+"_in="+opt_tok[1]
                add_options.append(opt_tok[0]+"_out="+opt_tok[1])
        args.options.extend(add_options)

        for opt in args.options:
            opt_tok = opt.split('=')
            if opt_tok[0] == "stop_type": # replace stop type
                for key in args_defense.stop_types:
                    if args_defense.type in args_defense.stop_types[key]:
                        args_defense.stop_types[key].remove(args_defense.type)
                    if key == opt_tok[1]:
                        args_defense.stop_types[key].append(args_defense.type)
                continue
            type_field = type(getattr(args_defense, opt_tok[0]))
            if opt_tok[1] == "Inf":
                setattr(args_defense, opt_tok[0], INF)
            else:
                if type_field is bool:
                    setattr(args_defense, opt_tok[0], opt_tok[1]=="True" or opt_tok[1] == "1")
                else:
                    try:
                        setattr(args_defense, opt_tok[0], type_field(opt_tok[1]))
                    except ValueError:
                        # type parsing failed, set as a float (if int failed)
                        try:
                            setattr(args_defense, opt_tok[0], float(opt_tok[1]))
                        except ValueError:
                            # type parsing failed again, set as string
                            setattr(args_defense, opt_tok[0], opt_tok[1])

    if args_defense.packet_prob_changes > 0 and args_defense.time_prob_changes > 0:
        print("WARNING: cannot make probability changes based both on number of packets and time. "
              "Use time")
        args_defense.packet_prob_changes = 0

    delimiter = find_delimiter(args.in_directory)
    args.delimiter = delimiter

    args.arg_str = "size-%d-%s" % (args_defense.size_packets, args_defense.type)

    if args_defense.type == "H":
        args.arg_str += "-"+args_defense.prob_type[0]

    if args.old:
        args.arg_str += "-old"

    if args_defense.type in args_defense.stop_types["wtfpad"]:
        pctl_str = str(int(args_defense.percentile*100)).zfill(3)
        args.arg_str += "_"+args.config.replace("_","-")+("-stop" if args_defense.stop_on_real else "")+\
                        ("-tuned"+pctl_str if args_defense.percentile != 0.5 else "")+\
                        ("-rf" if args_defense.refill else "")

    if args_defense.probabilistic:
        p_in_str = str(int(args_defense.p_in*100)).zfill(3)
        if args_defense.p_in != args_defense.p_out:
            p_out_str = str(int(args_defense.p_out * 100)).zfill(3)
            args.arg_str += "_p%s-%s" % (p_in_str, p_out_str)
        else:
            args.arg_str += "_p%s" % p_in_str
        if not args_defense.prob_type == "fixed":
            s_in_str = str(int(args_defense.prob_std_in * 100)).zfill(3)
            if args_defense.prob_std_in != args_defense.prob_std_out:
                s_out_str = str(int(args_defense.prob_std_out * 100)).zfill(3)
                args.arg_str += "_s%s-%s" % (s_in_str, s_out_str)
            else:
                args.arg_str += "_s%s" % s_in_str
        if args_defense.transform_in_out:
            args.arg_str += "_t1"
        if args_defense.pad_in != args_defense.pad_out:
            args.arg_str += "_pad-%s-%s" % (str(args_defense.pad_in), str(args_defense.pad_out))
        else:
            args.arg_str += "_pad-%s" % str(args_defense.pad_in)
    else:
        if args_defense.type not in args_defense.stop_types["wtfpad"]:
            if args_defense.rate_in != args_defense.rate_out:
                args.arg_str += "_rate-%s-%s" % (str(args_defense.rate_in), str(args_defense.rate_out))
            else:
                args.arg_str += "_rate-%s" % (str(args_defense.rate_in))
            if args_defense.type in args_defense.stop_types["global_stop"]:
                args.arg_str += "_time-" + str(args_defense.time)
            elif args_defense.type in args_defense.stop_types["int_padding"]:
                args.arg_str += "_L-" + str(args_defense.padding_param)
            elif args_defense.type in args_defense.stop_types["stream_padding"]:
                args.arg_str += ("_"+args_defense.padding_scheme_in+args_defense.padding_scheme_out)+\
                                ("1" if args_defense.early_termination else "0")

        if args_defense.type == "H":
            p_in_str = str(int(args_defense.p_in * 100)).zfill(3)
            if args_defense.p_in != args_defense.p_out:
                p_out_str = str(int(args_defense.p_out * 100)).zfill(3)
                args.arg_str += "_p%s-%s" % (p_in_str, p_out_str)
            else:
                args.arg_str += "_p%s" % p_in_str
            if not args_defense.prob_type == "fixed":
                s_in_str = str(int(args_defense.prob_std_in * 100)).zfill(3)
                if args_defense.prob_std_in != args_defense.prob_std_out:
                    s_out_str = str(int(args_defense.prob_std_out * 100)).zfill(3)
                    args.arg_str += "_s%s-%s" % (s_in_str, s_out_str)
                else:
                    args.arg_str += "_s%s" % s_in_str
            if "H" in args_defense.stop_types["wtfpad"]:
                if args_defense.pad_before:
                    args.arg_str += "-ra"
                if args_defense.all_sec:
                    args.arg_str += "-as"
                if args_defense.tune_hybrid:
                    args.arg_str += "-th"
                if args_defense.thr > 0:
                    args.arg_str += "_thr%s" % str(args_defense.thr).replace(".","")
                if args_defense.percentile_burst > 0:
                    pctl_str = str(int(args_defense.percentile_burst * 100)).zfill(3)
                    args.arg_str += "_bb"+pctl_str


            if args_defense.delay_in > 0 or args_defense.delay_out > 0:
                if args_defense.delay_in != args_defense.delay_out:
                    args.arg_str += ("_delay-%s-%s" % (str(args_defense.delay_in), str(args_defense.delay_out)))
                else:
                    args.arg_str += ("_delay-%s" % str(args_defense.delay_in))

    if args_defense.type == "H":
        args.arg_str += "_ch" + ("p" + str(args_defense.packet_prob_changes) if args_defense.packet_prob_changes > 0
                                 else "t" + str(args_defense.time_prob_changes))
        if args_defense.consecutive_packets_in != args_defense.consecutive_packets_out:
            args.arg_str += "-cp-%s-%s" % (str(args_defense.consecutive_packets_in), str(args_defense.consecutive_packets_out))
        else:
            args.arg_str += "-cp-%s" % (str(args_defense.consecutive_packets_in))
        args.arg_str += "-" + args_defense.num_conspack_dist[0]

    if args_defense.type in args_defense.stop_types["wtfpad"] and not args.only_create:

        # Get section in config file
        # get options of
        if "clusters_" in args.config:
            conf_parser = configparser.ConfigParser()
            conf_parser[args.config] = {}
            name_config = args.config.replace("clusters_", "")
            fold = name_config[:name_config.index("_n")]
            conf_parser[args.config]["cluster_path"] = "dump_"+fold+"/"+name_config+".clusters"
        else:
            conf_parser = configparser.RawConfigParser()
            conf_parser.read(CONFIG_FILE)
        config = conf_parser._sections[args.config]
        config.stop_on_real = args_defense.stop_on_real
        config.percentile = args_defense.percentile
        config.percentile_burst = args_defense.percentile_burst
        config.size_packets = args_defense.size_packets
        config.logger = logger
        config.seed = int(hashlib.md5(args.arg.encode()).hexdigest(), 16) % np.power(2,16)  # hash arg to set seed
        config.thr = args_defense.thr
        #config.thr_burst_in = args_defense.thr_burst_in
        #config.thr_burst_out = args_defense.thr_burst_out
        #config.prob_burst_in = args_defense.prob_burst_in
        #config.prob_burst_out = args_defense.prob_burst_out
        config.hybrid = (args_defense.type == "H") and \
                        (args_defense.thr > 0 or args_defense.percentile_burst > 0 or args_defense.thr_burst > 0)

        if args_defense.percentile_burst > 0 and args_defense.thr_burst > 0:
            print("WARNING: both percentile_burst and thr_burst are > 0")

        if "clusters" in args.config:
            # clustered, means that args.arg is set
            config.website = re.split("[-_]", args.arg)[0]
        else:
            config.website = ""
        args_defense.config = config

        wtfpad = adaptive.AdaptiveSimulator(config)
        args_defense.wtfpad = wtfpad

    args.args_defense = args_defense

    args.name = (args.out_directory if args.out_directory is not None else "") + args.arg_str +\
           ("_arg:" + args.arg if args.arg != "" else "")+("_arg:" + args.equal_arg if args.equal_arg != "" else "")

    return args, args_defense


if __name__ == "__main__":

    args, args_defense = get_args()

    if not args.dont_save and args.out_directory is None:
        print("Must provide an out directory")
        sys.exit(0)

    t_beg = time.time()
    args.logger.log(INFO, "Doing experiment " + args.name)

    if args.dont_save:
        print("WARNING: Will not save anything of this experiment")

    results = defenses(args, args_defense)

    print("Done in " + str(time.time() - t_beg) + " seconds")

    if len(results) > 0:

        args.logger.log(INFO, "Results for "+args.name)
        if args.verbose <= INFO:
            save_and_print_results(results, args, args.name)



