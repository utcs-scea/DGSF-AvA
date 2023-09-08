import argparse
import numpy as np


def load_stats(file_name, stats):
    with open(file_name, 'r') as fin:
        for line in fin:
            sp = line.strip().split(",")
            time = sp[1].strip()
            namesp = sp[0].split(" ")
            name = namesp[1]
            if name not in stats:
                stats[name] = [0, []]
            stats[name][0] += 1
            stats[name][1].append(int(time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gstats", required=True, type=str, help="guest stats file")
    parser.add_argument("--wstats", required=True, help="worker stats file")
    args = parser.parse_args()

    guest_stats = {}
    worker_stats = {}
    load_stats(args.gstats, guest_stats)
    load_stats(args.wstats, worker_stats)

    keys = sorted(guest_stats.keys())
    for n in keys:
        if n[-6:] == "_async":
            name = n[:-6]
        else:
            name = n
        if name in worker_stats:
            g_exec_time = np.array(guest_stats[n][1])
            w_exec_time = np.array(worker_stats[name][1])
            g_exec_time = g_exec_time / 1000000.0
            w_exec_time = w_exec_time / 1000000.0
            g_total = np.sum(g_exec_time)
            w_total = np.sum(w_exec_time)
            print(str(n), round(g_total, 3), round(w_total, 3),
                  round(g_total - w_total, 3))


if __name__ == '__main__':
    main()
