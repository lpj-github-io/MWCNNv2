import sys
import threading
import queue
import random
import collections

import torch
import torch.multiprocessing as multiprocessing

from torch._C import _set_worker_signal_handlers, _update_worker_pids, \
    _remove_worker_pids, _error_if_any_worker_fails
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _DataLoaderIter

from torch.utils.data.dataloader import ExceptionWrapper
from torch.utils.data.dataloader import _use_shared_memory
# from torch.utils.data.dataloader import _worker_manager_loop
from torch.utils.data.dataloader import numpy_type_map
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import pin_memory_batch
from torch.utils.data.dataloader import _SIGCHLD_handler_set
from torch.utils.data.dataloader import _set_SIGCHLD_handler
from torch.utils.data.dataloader import *
from torch.utils.data.dataloader import _pin_memory_loop

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, init_fn, worker_id):
    global _use_shared_memory
    _use_shared_memory = True
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            idx_scale = 0
            if len(scale) > 1 and dataset.train:
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            samples.append(idx_scale)

        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))

# class _MSDataLoaderIter(_DataLoaderIter):
#     def __init__(self, loader):
#         self.dataset = loader.dataset
#         self.scale = loader.scale
#         self.collate_fn = loader.collate_fn
#         self.batch_sampler = loader.batch_sampler
#         self.num_workers = loader.num_workers
#         self.pin_memory = loader.pin_memory and torch.cuda.is_available()
#         self.timeout = loader.timeout
#         self.done_event = threading.Event()
#
#         self.sample_iter = iter(self.batch_sampler)
#
#         if self.num_workers > 0:
#             self.worker_init_fn = loader.worker_init_fn
#             self.index_queues = [
#                 multiprocessing.Queue() for _ in range(self.num_workers)
#             ]
#             self.worker_queue_idx = 0
#             self.worker_result_queue = multiprocessing.SimpleQueue()
#             self.batches_outstanding = 0
#             self.worker_pids_set = False
#             self.shutdown = False
#             self.send_idx = 0
#             self.rcvd_idx = 0
#             self.reorder_dict = {}
#
#             base_seed = torch.LongTensor(1).random_()[0]
#             self.workers = [
#                 multiprocessing.Process(
#                     target=_ms_loop,
#                     args=(
#                         self.dataset,
#                         self.index_queues[i],
#                         self.worker_result_queue,
#                         self.collate_fn,
#                         self.scale,
#                         base_seed + i,
#                         self.worker_init_fn,
#                         i
#                     )
#                 )
#                 for i in range(self.num_workers)]
#
#             if self.pin_memory or self.timeout > 0:
#                 self.data_queue = queue.Queue()
#
#                 if self.pin_memory:
#                     maybe_device_id = torch.cuda.current_device()
#                 else:
#                     # do not initialize cuda context if not necessary
#                     maybe_device_id = None
#                 self.pin_memory_thread = threading.Thread(
#                     target=_pin_memory_loop,
#                     args=(self.worker_result_queue, self.data_queue, maybe_device_id, self.done_event))
#                     # args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
#                     #       maybe_device_id))
#
#                 self.pin_memory_thread.daemon = True
#                 self.pin_memory_thread.start()
#                 # self.pin_memory_thread = pin_memory_thread
#
#                 # self.worker_manager_thread.daemon = True
#                 # self.worker_manager_thread.start()
#             else:
#                 self.data_queue = self.worker_result_queue
#
#             for w in self.workers:
#                 w.daemon = True  # ensure that the worker exits on process exit
#                 w.start()
#
#             _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
#             _set_SIGCHLD_handler()
#             self.worker_pids_set = True
#
#             # prime the prefetch loop
#             for _ in range(2 * self.num_workers):
#                 self._put_indices()

IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


# NOTE [ Python Traceback Reference Cycle Problem ]
#
# When using sys.exc_info(), it is important to **not** store the exc_info[2],
# which is the traceback, because otherwise you will run into the traceback
# reference cycle problem, i.e., the traceback holding reference to the frame,
# and the frame (which holds reference to all the object in its temporary scope)
# holding reference the traceback.


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""
    def __init__(self, exc_info):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""

MP_STATUS_CHECK_INTERVAL = 5.0
r"""Interval (in seconds) to check status of processes to avoid hanging in
    multiprocessing data loading. This is mainly used in getting data from
    another process, in which case we need to periodically check whether the
    sender is alive to prevent hanging."""

if IS_WINDOWS:
    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()

            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
            return not self.manager_dead
else:
    class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


def _worker_loop(dataset, index_queue, data_queue, done_event, collate_fn, seed, init_fn, worker_id):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.

    try:
        global _use_shared_memory
        _use_shared_memory = True

        # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal happened again already.
        # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
        _set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        data_queue.cancel_join_thread()

        if init_fn is not None:
            init_fn(worker_id)

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if r is None:
                # Received the final signal
                assert done_event.is_set()
                return
            elif done_event.is_set():
                # Done event is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, batch_indices = r
            try:
                samples = collate_fn([dataset[i] for i in batch_indices])
            except Exception:
                # It is important that we don't store exc_info in a variable,
                # see NOTE [ Python Traceback Reference Cycle Problem ]
                data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            else:
                data_queue.put((idx, samples))
                del samples
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass


def _pin_memory_loop(in_queue, out_queue, device_id, done_event):
    torch.cuda.set_device(device_id)

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while True:
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        except Exception:
            if done_event.is_set():
                # Weird things can happen when shutting down, e.g., fd being
                # closed when tensors are shared via fds.
                break
            raise
        if r is None:
            assert done_event.is_set()
            return
        elif done_event.is_set():
            # Haven't seen the final signal yet. Keep getting until None.
            continue
        elif isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
        else:
            idx, batch = r
            try:
                batch = pin_memory_batch(batch)
            except Exception:
                out_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            else:
                out_queue.put((idx, batch))

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def pin_memory_batch(batch):
    if isinstance(batch, torch.Tensor):
        return batch.pin_memory()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, container_abcs.Mapping):
        return {k: pin_memory_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, container_abcs.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch


_SIGCHLD_handler_set = False
r"""Whether SIGCHLD handler is set for DataLoader worker failures. Only one
handler needs to be set for all DataLoaders in a process."""


def _set_SIGCHLD_handler():
    # Windows doesn't support SIGCHLD handler
    if sys.platform == 'win32':
        return
    # can't set signal in child threads
    if not isinstance(threading.current_thread(), threading._MainThread):
        return
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return
    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        # This doesn't catch default handler, but SIGCHLD default handler is a
        # no-op.
        previous_handler = None

    def handler(signum, frame):
        # This following call uses `waitid` with WNOHANG from C side. Therefore,
        # Python can still get and update the process status successfully.
        _error_if_any_worker_fails()
        if previous_handler is not None:
            previous_handler(signum, frame)

    signal.signal(signal.SIGCHLD, handler)
    _SIGCHLD_handler_set = True


_python_exit_status = False

def _set_python_exit_flag():
    global _python_exit_status
    _python_exit_status = True

atexit.register(_set_python_exit_flag)

class _MSDataLoaderIter(object):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""


    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout

        self.sample_iter = iter(self.batch_sampler)

        base_seed = torch.LongTensor(1).random_().item()

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}
            self.done_event = multiprocessing.Event()

            self.index_queues = []
            self.workers = []
            for i in range(self.num_workers):
                index_queue = multiprocessing.Queue()
                index_queue.cancel_join_thread()
                w = multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, index_queue,
                          self.worker_result_queue, self.done_event,
                          self.collate_fn, base_seed + i,
                          self.worker_init_fn, i))
                w.daemon = True
                # NB: Process.start() actually take some time as it needs to
                #     start a process and pass the arguments over via a pipe.
                #     Therefore, we only add a worker to self.workers list after
                #     it started, so that we do not call .join() if program dies
                #     before it starts, and __del__ tries to join but will get:
                #     AssertionError: can only join a started process.
                w.start()
                self.index_queues.append(index_queue)
                self.workers.append(w)

            if self.pin_memory:
                self.data_queue = queue.Queue()
                pin_memory_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(self.worker_result_queue, self.data_queue,
                          torch.cuda.current_device(), self.done_event))
                pin_memory_thread.daemon = True
                pin_memory_thread.start()
                # Similar to workers (see comment above), we only register
                # pin_memory_thread once it is started.
                self.pin_memory_thread = pin_memory_thread
            else:
                self.data_queue = self.worker_result_queue

            _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self):
        return len(self.batch_sampler)

    def _get_batch(self):
        # In the non-timeout case, worker exit is covered by SIGCHLD handler.
        # But if `pin_memory=True`, we still need account for the possibility
        # that `pin_memory_thread` dies.
        if self.timeout > 0:
            try:
                return self.data_queue.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self.timeout))
        elif self.pin_memory:
            while self.pin_memory_thread.is_alive():
                try:
                    return self.data_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
                except queue.Empty:
                    continue
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self.data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            return self.data_queue.get()

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("_DataLoaderIter cannot be pickled")

    def _shutdown_workers(self):
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        if _python_exit_status is True or _python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self.shutdown:
            self.shutdown = True
            # Removes pids from the C side data structure first so worker
            # termination afterwards won't trigger false positive error report.
            if self.worker_pids_set:
                _remove_worker_pids(id(self))
                self.worker_pids_set = False

            self.done_event.set()

            # Exit `pin_memory_thread` first because exiting workers may leave
            # corrupted data in `worker_result_queue` which `pin_memory_thread`
            # reads from.
            if hasattr(self, 'pin_memory_thread'):
                # Use hasattr in case error happens before we set the attribute.
                # First time do `worker_result_queue.put` in this process.

                # `cancel_join_thread` in case that `pin_memory_thread` exited.
                self.worker_result_queue.cancel_join_thread()
                self.worker_result_queue.put(None)
                self.pin_memory_thread.join()
                # Indicate that no more data will be put on this queue by the
                # current process. This **must** be called after
                # `pin_memory_thread` is joined because that thread shares the
                # same pipe handles with this loader thread. If the handle is
                # closed, Py3 will error in this case, but Py2 will just time
                # out even if there is data in the queue.
                self.worker_result_queue.close()

            # Exit workers now.
            for q in self.index_queues:
                q.put(None)
                # Indicate that no more data will be put on this queue by the
                # current process.
                q.close()
            for w in self.workers:
                w.join()

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()

# class MSDataLoader(DataLoader):
#     def __init__(
#         self, args, dataset, batch_size=1, shuffle=False,
#         sampler=None, batch_sampler=None,
#         collate_fn=default_collate, pin_memory=False, drop_last=False,
#         timeout=0, worker_init_fn=None):
#
#         super(MSDataLoader, self).__init__(
#             dataset, batch_size=batch_size, shuffle=shuffle,
#             sampler=sampler, batch_sampler=batch_sampler,
#             num_workers=args.n_threads, collate_fn=collate_fn,
#             pin_memory=pin_memory, drop_last=drop_last,
#             timeout=timeout, worker_init_fn=worker_init_fn)
#
#         self.scale = args.scale
#
#     def __iter__(self):
#         return _MSDataLoaderIter(self)

# noinspection PyInterpreter
class MSDataLoader(DataLoader):
    r"""

    """

    __initialized = False

    def __init__(self, args, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        # super(MSDataLoader, self).__init__(
        #     dataset, batch_size=batch_size, shuffle=shuffle,
        #     sampler=sampler, batch_sampler=batch_sampler,
        #     num_workers=args.n_threads, collate_fn=collate_fn,
        #     pin_memory=pin_memory, drop_last=drop_last,
        #     timeout=timeout, worker_init_fn=worker_init_fn)

        self.scale = args.scale

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if self.num_workers < 0:
            raise ValueError('num_workers option cannot be negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(MSDataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        return _MSDataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)
