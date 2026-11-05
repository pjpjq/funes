import threading
import unittest
from unittest.mock import patch

import space.server as bridge


class NativeMemoryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.constructed = threading.Event()
        self.polled = threading.Event()
        self.errors = []
        self.threads = []
        test = self

        class Process:
            stdin = stdout = stderr = None
            alive = True

            def poll(self):
                test.polled.set()
                return None if self.alive else 0

            def wait(self, timeout=None):
                self.alive = False
                return 0

        class Worker(bridge.NativeMcpWorker):
            fail_probe = False

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._process = Process()
                self.closed = False
                self.write_lock_held = []
                test.constructed.set()

            def recall(self, query, **kwargs):
                self.write_lock_held.append(bridge.NATIVE_WRITE_LOCK.locked())
                return "recall error: probe failed" if self.fail_probe else "ok"

            def close(self):
                self.closed = True
                self._process = None

        self.Worker = Worker
        self.patcher = patch.multiple(
            bridge, NativeMcpWorker=Worker, MCP_WORKER=None, _MCP_WORKER_CONFIG=None,
            NATIVE_WRITE_LOCK=threading.Lock(), INDEX_LOCK=threading.Lock(),
            SOURCE_APP=None,
        )
        self.patcher.start()

    def tearDown(self):
        try:
            for thread in self.threads:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive(), "refresh thread leaked")
        finally:
            self.patcher.stop()

    def active(self):
        old = self.Worker("unused", "local", bridge.HOME)
        bridge.MCP_WORKER = old
        bridge._MCP_WORKER_CONFIG = bridge._native_worker_config()
        self.constructed.clear()
        return old

    def start_refresh(self):
        def run():
            try:
                bridge._refresh_native_worker()
            except Exception as error:
                self.errors.append(error)
        thread = threading.Thread(target=run)
        self.threads.append(thread)
        thread.start()
        return thread

    def finish(self, thread):
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.errors, [])

    def test_healthy_worker_serves_while_replacement_waits_for_writer(self):
        old = self.active()
        with bridge.NATIVE_WRITE_LOCK:
            thread = self.start_refresh()
            self.assertTrue(self.polled.wait(1))
            self.assertFalse(self.constructed.is_set())
            self.assertEqual(bridge.recall("foreground"), "ok")
            self.assertFalse(self.constructed.is_set())
        self.finish(thread)
        self.assertIsNot(bridge.MCP_WORKER, old)
        self.assertTrue(old.closed)
        self.assertEqual(bridge.MCP_WORKER.write_lock_held, [True])
        self.assertFalse(bridge.NATIVE_WRITE_LOCK.locked())

    def test_initial_warm_bypasses_writer_and_recall_locks(self):
        with bridge.NATIVE_WRITE_LOCK, bridge.INDEX_LOCK:
            thread = self.start_refresh()
            self.assertTrue(self.constructed.wait(1))
            self.assertTrue(bridge.NATIVE_WRITE_LOCK.locked())
        self.finish(thread)

    def test_dead_worker_recovers_without_waiting_for_writer_or_respawning_old(self):
        old = self.active()
        with bridge.NATIVE_WRITE_LOCK:
            thread = self.start_refresh()
            self.assertTrue(self.polled.wait(1))
            old.process.alive = False
            self.finish(thread)
            self.assertTrue(bridge.NATIVE_WRITE_LOCK.locked())
            self.assertTrue(old._retired)
            with patch.object(old, "_start_locked") as start:
                with self.assertRaises(bridge.NativeMcpBusyError):
                    old._ensure_started_locked()
                start.assert_not_called()

    def test_busy_dead_worker_is_not_retired_during_a_foreground_retry(self):
        old = self.active()
        old.process.alive = False
        result = []
        with old._lock:
            thread = threading.Thread(target=lambda: result.append(old._retire_if_stopped()))
            thread.start()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
        self.assertEqual(result, [False])
        self.assertFalse(old._retired)

    def test_probe_failure_releases_slot_and_keeps_healthy_old(self):
        old = self.active()
        self.Worker.fail_probe = True
        with self.assertRaises(bridge.NativeMcpError):
            bridge._refresh_native_worker()
        self.assertIs(bridge.MCP_WORKER, old)
        self.assertFalse(old.closed)
        self.assertFalse(bridge.NATIVE_WRITE_LOCK.locked())


if __name__ == "__main__":
    unittest.main()
