import os
from psutil import Process
from collections import namedtuple
from itertools import groupby

_proc = Process(os.getpid())


def get_consumed_ram():
    return _proc.memory_info().rss


START = 'START'
END = 'END'
ConsumedRamLogEntry = namedtuple(
    'ConsumedRamLogEntry', ('nodeid', 'on', 'consumed_ram'))
consumed_ram_log = []


def pytest_runtest_setup(item):
    log_entry = ConsumedRamLogEntry(item.nodeid, START, get_consumed_ram())
    consumed_ram_log.append(log_entry)


def pytest_runtest_teardown(item):
    log_entry = ConsumedRamLogEntry(item.nodeid, END, get_consumed_ram())
    consumed_ram_log.append(log_entry)


LEAK_LIMIT = 10 * 1024 * 1024


def pytest_terminal_summary(terminalreporter):
    grouped = groupby(consumed_ram_log, lambda entry: entry.nodeid)
    for nodeid, entries in grouped:
        try:
            start_entry, end_entry = entries
            leaked = end_entry.consumed_ram - start_entry.consumed_ram
            if leaked > LEAK_LIMIT:
                terminalreporter.write('LEAKED {}MB in {}\n'.format(
                    leaked / 1024 / 1024, nodeid))
                terminalreporter.write('MEMORY ENDED AT: {}MB in {}\n'.format(
                    end_entry.consumed_ram / 1024 / 1024, nodeid))
        except:
            pass
