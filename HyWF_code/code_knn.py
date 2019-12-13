from utils import *


# Features for Wang et al. (Usenix 14)
def extract_Wang(times, sizes):

    features = []

    iscell = len(list(set(np.abs(sizes)))) <= 1

    if not iscell:
        print("No cell??")

    # Transmission size features
    features.append(len(sizes))
    features += [0]*3
    count_f = len(features)
    if not iscell:
        # for unique packet lengths
        features += [0]*3001
        count_plen = 3001
    else:
        count_plen = 0
    features += [-1]*1000
    features += [0]*100

    count_pos = 0
    prevloc = 0
    count_dist = 0
    bursts = []
    curburst = 0
    itimes = [0] * (len(sizes) - 1)

    if not iscell:
        # Unique packet lengths (useless when tor cells)
        for i in range(-1500, 1501):
            if i in sizes:
                features[count_f+1500+i] = 1
            else:
                features[count_f+1500+i] = 0

    for i in range(0, len(sizes)):
        x = sizes[i]
        if i > 0 and len(times) > 0:
            itimes[i - 1] = times[i] - times[i - 1]
        if x > 0:
            count_pos += 1

            if count_pos <= 500:
                # Transpositions (similar to good distance scheme)
                features[count_f+count_plen+count_pos] = i
                features[count_f+500+count_plen+count_pos] = i - prevloc
                prevloc = i

            # Packet distributions (where are the outgoing packets concentrated)
            if i % 30 != 29:
                count_dist += 1

            # Bursts
            if len(bursts) > 0:
                if bursts[-1] != curburst:
                    bursts.append(curburst)
            else:
                bursts.append(curburst)
        else:
            # Bursts
            curburst -= x

        if i < min(len(sizes), 3000):
            if i % 30 == 29:
                features[count_f+1000+int(i/30)] = count_dist
                count_dist = 0

    features[count_f-3] = count_pos
    features[count_f-2] = len(sizes)-count_pos

    if len(times) > 0:
        features[count_f-1] = times[-1] - times[0]

    if len(bursts) > 0:
        features.append(max(bursts))
        features.append(np.mean(bursts))
        features.append(len(bursts))
    else:
        features.append(-1)
        features.append(-1)
        features.append(-1)

    # print bursts
    counts = [0, 0, 0, 0, 0, 0]
    for x in bursts:
        if x > 2:
            counts[0] += 1
        if x > 5:
            counts[1] += 1
        if x > 10:
            counts[2] += 1
        if x > 15:
            counts[3] += 1
        if x > 20:
            counts[4] += 1
        if x > 50:
            counts[5] += 1
    features.append(counts[0])
    features.append(counts[1])
    features.append(counts[2])
    features.append(counts[3])
    features.append(counts[4])
    features.append(counts[5])
    for i in range(0, 100):
        try:
            features.append(bursts[i])
        except:
            features.append(-1)

    for i in range(0, 10):
        try:
            features.append(sizes[i] + 1500)
        except:
            features.append(-1)

    if len(itimes) > 0:
        features.append(np.mean(itimes))
        features.append(np.std(itimes))

    return features
