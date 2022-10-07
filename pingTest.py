'''ping a server and return the result'''
import subprocess
import sys
import os
import time
import datetime
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

#from matplotlib import rc
#rc('text', usetex=True)
#rc('font', family='serif')

def pingTest(host, count=10, interval=1):
    '''Ping a server and return the result'''
    cmd = ['ping', '-c', str(count), '-i', str(interval), host]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print('Error: {}'.format(result.stderr.decode('utf-8')))
        return None
    else:
        return result.stdout.decode('utf-8')

def parsePingResult(result):
    '''Parse the ping result and return the average ping time'''
    lines = result.splitlines()
    print(lines)
    if len(lines) < 2:
        return None
    else:
        #line = lines[2]
        line = lines[-1]
        # regex to check if the result is in the format "dd bytes from dd.dd.dd.dd: icmp_seq=dd ttl=dd time=dd.dd ms"
        print(line)
        # regex to check if the result is in the format "round-trip min/avg/max/stddev = dd.dd/dd.dd/dd.dd/dd.dd ms"
        match = re.search(r'round-trip min/avg/max/stddev = (\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+) ms', line)
        #match = re.search(r'\d+ bytes from \d+\.\d+\.\d+\.\d+: icmp_seq=\d+ ttl=\d+ time=(\d+\.\d+) ms', line)
        if match:
            return float(match.group(2))
        else:
            return None

def pingTestPlot(host, count=10, interval=1, plot=True):
    '''Ping a server and plot the result'''
    result = pingTest(host, count, interval)
    if result:
        pingTimes = []
        for i in range(count):
            result = pingTest(host, count, interval)
            pingTimes.append(parsePingResult(result))
            time.sleep(interval)
        if plot:
            plt.plot(pingTimes)
            plt.xlabel('Ping number')
            plt.ylabel('Ping time (ms)')
            plt.title('Ping time for {}'.format(host))
            plt.show()
        return pingTimes
    else:
        return None


def main():
    '''Main function'''
    parser = argparse.ArgumentParser(description='Ping a server and plot the result')
    parser.add_argument('host', help='host to ping')
    parser.add_argument('-c', '--count', type=int, default=10, help='number of pings')
    parser.add_argument('-i', '--interval', type=float, default=1, help='interval between pings')
    args = parser.parse_args()
    pingTestPlot(args.host, args.count, args.interval)

if __name__ == '__main__':
    print(sys.argv)
    main()