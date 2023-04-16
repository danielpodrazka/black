"""
Formatting many files at once via multiprocessing. Contains entrypoint and utilities.

NOTE: this module is only imported if we need to format several files at once.
"""

import asyncio
import logging
import os
import signal
import sys
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Iterable, Optional, Set

from mypy_extensions import mypyc_attr

from black import WriteBack, format_file_in_place
from black.cache import Cache, filter_cached, read_cache, write_cache
from black.mode import Mode
from black.output import err
from black.report import Changed, Report
from typing import Optional
from typing import Iterable, Any
from typing import List, Tuple
from typing import Set, Optional
from black import WriteBack, Mode
from black.report import Report
from asyncio import AbstractEventLoop
from concurrent.futures import Executor

from black import WriteBack, format_file_in_place, Mode

from black_compat import cancel, maybe_install_uvloop, reformat_many, shutdown


def prepare_cache(
    sources: Set[Path], write_back: WriteBack, mode: Mode, report: Report
) -> Tuple[Cache, Set[Path]]:
    cache = {}
    if write_back not in (WriteBack.DIFF, WriteBack.COLOR_DIFF):
        cache = read_cache(mode)
        sources, cached = filter_cached(cache, sources)
        for src in sorted(cached):
            report.done(src, Changed.CACHED)
    return cache, sources


def create_lock(write_back: WriteBack) -> Optional[Any]:
    lock = None
    if write_back in (WriteBack.DIFF, WriteBack.COLOR_DIFF):
        manager = Manager()
        lock = manager.Lock()
    return lock


def schedule_tasks(
    loop: asyncio.AbstractEventLoop,
    executor: Executor,
    sources: Set[Path],
    fast: bool,
    mode: Mode,
    write_back: WriteBack,
    lock: Any,
) -> Dict[asyncio.Future, Path]:
    tasks = {
        asyncio.ensure_future(
            loop.run_in_executor(
                executor, format_file_in_place, src, fast, mode, write_back, lock
            )
        ): src
        for src in sorted(sources)
    }
    return tasks


async def process_formatter_results(
    tasks: Dict[asyncio.Future, Path],
    write_back: WriteBack,
    cache: Cache,
    mode: Mode,
    report: Report,
) -> None:
    cancelled = []
    sources_to_cache = []
    pending = tasks.keys()

    while pending:
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            src = tasks.pop(task)
            if task.cancelled():
                cancelled.append(task)
            elif task.exception():
                report.failed(src, str(task.exception()))
            else:
                changed = Changed.YES if task.result() else Changed.NO
                if write_back is WriteBack.YES or (
                    write_back is WriteBack.CHECK and changed is Changed.NO
                ):
                    sources_to_cache.append(src)
                report.done(src, changed)
    if cancelled:
        await asyncio.gather(*cancelled, return_exceptions=True)
    if sources_to_cache:
        write_cache(cache, sources_to_cache, mode)


async def schedule_formatting(
    sources: Set[Path],
    fast: bool,
    write_back: WriteBack,
    mode: Mode,
    report: Report,
    loop: asyncio.AbstractEventLoop,
    executor: Executor,
) -> None:
    cache, sources = prepare_cache(sources, write_back, mode, report)
    if not sources:
        return

    lock = create_lock(write_back)
    tasks = schedule_tasks(loop, executor, sources, fast, mode, write_back, lock)

    try:
        loop.add_signal_handler(signal.SIGINT, cancel, tasks.keys())
        loop.add_signal_handler(signal.SIGTERM, cancel, tasks.keys())
    except NotImplementedError:
        pass

    await process_formatter_results(tasks, write_back, cache, mode, report)


def maybe_install_uvloop() -> None:
    """
    Check if uvloop is installed and use it if available.

    This function tries to import the uvloop package and, if successful, installs it as
    the event loop policy for improved performance. It is called only from command-line
    entry points to avoid interfering with the parent process if Black is used as a
    library.

    This should be called at the beginning of the __main__ entry points of command-line
    scripts that use the asyncio event loop for better performance.
    """
    uvloop_module = attempt_uvloop_import()
    if uvloop_module is not None:
        install_uvloop(uvloop_module)


def create_executor(workers: Optional[int]) -> Executor:
    """Create an executor based on the number of workers passed."""
    if workers is None:
        workers = os.cpu_count() or 1
    if sys.platform == "win32":
        workers = min(workers, 60)

    try:
        return ProcessPoolExecutor(max_workers=workers)
    except (ImportError, NotImplementedError, OSError):
        return ThreadPoolExecutor(max_workers=1)


def setup_event_loop() -> asyncio.AbstractEventLoop:
    """Set up the event loop and return it."""
    maybe_install_uvloop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def close_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Close the event loop and remove it."""
    try:
        shutdown(loop)
    finally:
        asyncio.set_event_loop(None)


def run_event_loop(
    loop: asyncio.AbstractEventLoop,
    sources: Set[Path],
    fast: bool,
    write_back: WriteBack,
    mode: Mode,
    report: Report,
    executor: Executor,
) -> None:
    """Run the event loop using the necessary arguments."""
    loop.run_until_complete(
        schedule_formatting(
            sources=sources,
            fast=fast,
            write_back=write_back,
            mode=mode,
            report=report,
            loop=loop,
            executor=executor,
        )
    )


async def reformat_many(
    sources: Set[Path],
    fast: bool,
    write_back: WriteBack,
    mode: Mode,
    report: Report,
    workers: Optional[int],
) -> None:
    """
    Reformat multiple files using a ProcessPoolExecutor.

    Args:
        sources: A set of file paths to be reformatted.
        fast: If true, skip the normal formatting process for files that have writable caches.
        write_back: An instance of WriteBack to handle the writing of files.
        mode: An instance of Mode, which contains formatting settings.
        report: An instance of Report for recording files that have been changed.
        workers: The number of workers to be used for formatting. Uses os.cpu_count() if not specified.
    """

    executor = create_executor(workers)
    loop = setup_event_loop()
    try:
        run_event_loop(loop, sources, fast, write_back, mode, report, executor)
    finally:
        close_event_loop(loop)
        if executor is not None:
            executor.shutdown()


def gather_tasks(loop: asyncio.AbstractEventLoop) -> List[asyncio.Task]:
    """
    Gather all pending tasks from the given `loop`.

    Args:
        loop (asyncio.AbstractEventLoop): The event loop containing the tasks.

    Returns:
        List[asyncio.Task]: A list of all pending tasks.
    """
    return [task for task in asyncio.all_tasks(loop) if not task.done()]


def cancel_tasks(tasks: List[asyncio.Task]) -> None:
    """
    Cancel all tasks in the given list.

    Args:
        tasks (List[asyncio.Task]): A list of tasks to cancel.

    Returns:
        None
    """
    for task in tasks:
        task.cancel()


def silence_concurrent_futures() -> None:
    """
    Silence the logging of concurrent.futures to avoid spewing about the
    event loop being closed.

    Returns:
        None
    """
    cf_logger = logging.getLogger("concurrent.futures")
    cf_logger.setLevel(logging.CRITICAL)


def shutdown(loop: asyncio.AbstractEventLoop) -> None:
    """
    Cancel all pending tasks on the given `loop`, wait for them, and close the loop.

    This function is responsible for gracefully shutting down the asyncio event loop,
    ensuring that all tasks are cancelled and completed before closing the loop.
    It is used to prevent hanging tasks when stopping the program.

    Args:
        loop (asyncio.AbstractEventLoop): The event loop containing the tasks to
                                          cancel and shut down.

    Returns:
        None
    """
    try:
        to_cancel = gather_tasks(loop)
        if not to_cancel:
            return

        cancel_tasks(to_cancel)
        loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))
    finally:
        silence_concurrent_futures()
        loop.close()


def abort() -> None:
    """Print an abort message to stderr."""
    err("Aborted!")


def cancel_all_tasks(tasks: Iterable["asyncio.Task[Any]"]) -> None:
    """
    Cancel all given asyncio tasks.

    Args:
        tasks: Iterable of asyncio tasks to be canceled.
    """
    for task in tasks:
        task.cancel()


def cancel(tasks: Iterable["asyncio.Task[Any]"]) -> None:
    """
    asyncio signal handler that cancels all given tasks and reports the abort message to stderr.

    Args:
        tasks: Iterable of asyncio tasks to be canceled.
    """
    abort()
    cancel_all_tasks(tasks)


def attempt_uvloop_import() -> Optional[Any]:
    """Attempt to import uvloop and return the module if successful, None otherwise."""
    try:
        import uvloop

        return uvloop
    except ImportError:
        return None


def install_uvloop(uvloop_module: Any) -> None:
    """Install uvloop as the event loop policy."""
    uvloop_module.install()
