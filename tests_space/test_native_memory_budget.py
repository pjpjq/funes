import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def final_optimize_app(self):
        store = Mock()
        store.native_optimize_checkpoint.return_value = {}
        store.native_index_checkpoint.return_value = {
            "complete": True, "indexed": 1, "index_fingerprint": "final-index",
        }
        syncer = Mock()
        syncer.upload.return_value = {"durable": True}
        return SimpleNamespace(store=store, syncer=syncer)

    def start_final_optimize(self, app):
        results = []

        def run():
            try:
                results.append(bridge.optimize_canonical_index(
                    app, {"fingerprint": "final-profile"}, "test/final-memory",
                ))
            except Exception as error:
                self.errors.append(error)

        thread = threading.Thread(target=run)
        self.threads.append(thread)
        thread.start()
        return thread, results

    def test_final_optimize_waits_for_existing_writer_and_checkpoints_once(self):
        app = self.final_optimize_app()
        slot = threading.Lock()
        attempted_or_started = threading.Event()
        order = []

        class ObservedSlot:
            def __enter__(self):
                order.append("acquire")
                attempted_or_started.set()
                slot.acquire()

            def __exit__(self, *args):
                slot.release()

        def optimize(*args):
            order.append("optimize")
            attempted_or_started.set()
            return True

        with patch.object(bridge, "NATIVE_WRITE_LOCK", ObservedSlot()), patch.object(
            bridge, "optimize_native_index", side_effect=optimize,
        ) as native:
            thread = None
            try:
                with slot:
                    thread, results = self.start_final_optimize(app)
                    self.assertTrue(attempted_or_started.wait(1))
                    self.assertEqual(order, ["acquire"])
                    app.syncer.upload.assert_not_called()
            finally:
                if thread is not None:
                    self.finish(thread)
        self.assertEqual(results, [True])
        self.assertEqual(order, ["acquire", "optimize"])
        native.assert_called_once_with("test/final-memory", {"fingerprint": "final-profile"})
        app.syncer.upload.assert_called_once()
        marker = app.syncer.upload.call_args.args[0][0]
        self.assertEqual(marker["status"], "optimized")
        app.store.set_native_optimize_checkpoint.assert_called_once_with(marker)

    def test_final_optimize_blocks_replacement_warm_but_not_healthy_queries(self):
        old = self.active()
        app = self.final_optimize_app()
        entered = threading.Event()
        release = threading.Event()
        lock_states = []

        def optimize(*args):
            lock_states.append(bridge.NATIVE_WRITE_LOCK.locked())
            entered.set()
            if not release.wait(3):
                raise AssertionError("final optimize test was not released")
            return True

        with patch.object(bridge, "optimize_native_index", side_effect=optimize):
            thread, results = self.start_final_optimize(app)
            refresh = None
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(lock_states, [True])
                refresh = self.start_refresh()
                self.assertTrue(self.polled.wait(1))
                self.assertFalse(self.constructed.is_set())
                self.assertIs(bridge.MCP_WORKER, old)
                self.assertEqual(bridge.recall("foreground"), "ok")
            finally:
                release.set()
                self.finish(thread)
                if refresh is not None:
                    self.finish(refresh)
        self.assertEqual(results, [True])
        self.assertIsNot(bridge.MCP_WORKER, old)
        self.assertTrue(old.closed)
        self.assertFalse(bridge.NATIVE_WRITE_LOCK.locked())

    def test_final_optimize_releases_writer_slot_after_native_error(self):
        app = self.final_optimize_app()
        lock_states = []

        def fail(*args):
            lock_states.append(bridge.NATIVE_WRITE_LOCK.locked())
            raise RuntimeError("test native error")

        with patch.object(bridge, "optimize_native_index", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "test native error"):
                bridge.optimize_canonical_index(
                    app, {"fingerprint": "final-profile"}, "test/final-memory",
                )
        self.assertEqual(lock_states, [True])
        self.assertFalse(bridge.NATIVE_WRITE_LOCK.locked())
        self.assertFalse(bridge.WRITE_LOCK.locked())
        app.syncer.upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
