from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import threading
import sys
import queue


class Worker(threading.Thread):
    """Worker
    :param task:
    :type task: Callable
    """

    _queue = queue.Queue()

    def __init__(self, task):
        threading.Thread.__init__(self)
        self.daemon = True
        self._task = task
        self.values = {}

    def run(self):
        while True:
            record = Worker._queue.get()
            # last queued item is None
            if record is None:
                Worker._queue.task_done()
                break
            self._task(record, self.values)
            Worker._queue.task_done()

    @classmethod
    def put(cls, record):
        Worker._queue.put(record)

    @classmethod
    def wait(cls):
        Worker._queue.join()


def run(task, records, num_threads=3, show_progress=None, max_queue=100000):
    """run

    :param num_threads:
    :int num_threads:
    :param task: f(record, values)
    :type task: Callable
    :param records:
    :type records: iterable
    :param show_progress:
    :type show_progress: int or None
    :param max_queue:
    :type max_queue: int
    """
    threads = []
    for i in range(num_threads):
        t = Worker(task)
        t.start()
        threads.append(t)

    for i, record in enumerate(records):
        if show_progress is not None and (i % show_progress) == 0:
            print("Processing {0} items... ".format(i), file=sys.stderr)
        if (i % max_queue) == (max_queue - 1):
            print("Wait until queued tasks are processed...", file=sys.stderr)
            Worker.wait()
        Worker.put(record)

    for t in threads:
        Worker.put(None)

    Worker.wait()

    # collect values
    return [(t.name, t.values) for t in threads]
