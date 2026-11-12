import asyncio
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
from orca_gym.core import orca_gym_local
from mtool.model_sources import (
    cleanup_orca_fetch,
    download_orca_xml,
    fetch_orca_xml,
    poll_orca_states,
    stream_orca_states,
)


class DownloadOrcaXmlTests(unittest.IsolatedAsyncioTestCase):
    @patch("mtool.model_sources.OrcaGymLocal")
    @patch("mtool.model_sources.GrpcServiceStub")
    @patch("mtool.model_sources.grpc.aio.insecure_channel")
    async def test_download_passes_arguments_and_closes_channel(
        self, insecure_channel, stub_class, gym_class
    ):
        channel = insecure_channel.return_value
        channel.close = AsyncMock()
        stub = stub_class.return_value
        gym = gym_class.return_value
        gym.load_model_xml = AsyncMock(return_value="/tmp/cache/scene.xml")
        directory = Path("/tmp/cache")

        with patch("mtool.model_sources.asyncio.wait_for", wraps=asyncio.wait_for) as wait_for:
            result = await download_orca_xml(
                "localhost:50051", directory, timeout=12.5
            )

        self.assertEqual(result, Path("/tmp/cache/scene.xml"))
        insecure_channel.assert_called_once_with(
            "localhost:50051",
            options=(("grpc.max_receive_message_length", 64 * 1024 * 1024),),
        )
        stub_class.assert_called_once_with(channel)
        gym_class.assert_called_once_with(stub, xml_assets_dir=str(directory))
        gym.load_model_xml.assert_awaited_once_with()
        wait_for.assert_awaited_once()
        self.assertEqual(wait_for.await_args.kwargs, {"timeout": 12.5})
        channel.close.assert_awaited_once_with()

    @patch("mtool.model_sources.OrcaGymLocal")
    @patch("mtool.model_sources.GrpcServiceStub")
    @patch("mtool.model_sources.grpc.aio.insecure_channel")
    async def test_file_lock_avoids_main_thread_signals(
        self, insecure_channel, _stub_class, gym_class
    ):
        insecure_channel.return_value.close = AsyncMock()
        original_lock = MagicMock()
        real_lock = orca_gym_local.file_lock
        orca_gym_local.file_lock = original_lock
        self.addCleanup(setattr, orca_gym_local, "file_lock", real_lock)

        async def load_model_xml():
            orca_gym_local.file_lock("scene.xml", timeout=30)
            return "/tmp/cache/scene.xml"

        gym_class.return_value.load_model_xml = load_model_xml
        await download_orca_xml("localhost:50051", "/tmp/cache")

        original_lock.assert_called_once_with(
            "scene.xml", timeout=30, non_blocking=True
        )
        self.assertIs(orca_gym_local.file_lock, original_lock)

    @patch("mtool.model_sources.OrcaGymLocal")
    @patch("mtool.model_sources.GrpcServiceStub")
    @patch("mtool.model_sources.grpc.aio.insecure_channel")
    async def test_timeout_closes_channel(self, insecure_channel, _stub_class, gym_class):
        channel = insecure_channel.return_value
        channel.close = AsyncMock()
        gym = gym_class.return_value

        async def slow_load():
            await asyncio.sleep(0.05)

        gym.load_model_xml = slow_load

        with self.assertRaises(asyncio.TimeoutError):
            await download_orca_xml("localhost:50051", "/tmp/cache", timeout=0.001)

        channel.close.assert_awaited_once_with()

    @patch("mtool.model_sources.OrcaGymLocal")
    @patch("mtool.model_sources.GrpcServiceStub")
    @patch("mtool.model_sources.grpc.aio.insecure_channel")
    async def test_load_error_propagates_and_closes_channel(
        self, insecure_channel, _stub_class, gym_class
    ):
        channel = insecure_channel.return_value
        channel.close = AsyncMock()
        error = RuntimeError("load failed")
        gym_class.return_value.load_model_xml = AsyncMock(side_effect=error)

        with self.assertRaises(RuntimeError) as caught:
            await download_orca_xml("localhost:50051", "/tmp/cache")

        self.assertIs(caught.exception, error)
        channel.close.assert_awaited_once_with()


class PollOrcaStatesTests(unittest.IsolatedAsyncioTestCase):
    @patch("mtool.model_sources.OrcaGymRemote")
    @patch("mtool.model_sources.GrpcServiceStub")
    @patch("mtool.model_sources.grpc.aio.insecure_channel")
    async def test_poll_publishes_state_and_closes_channel(
        self, insecure_channel, stub_class, gym_class
    ):
        channel = insecure_channel.return_value
        channel.close = AsyncMock()
        state = (
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
        )
        gym_class.return_value.query_all_qpos_qvel_qacc = AsyncMock(
            return_value=state
        )
        stop_event = threading.Event()
        published = []

        def publish(received_state):
            published.append(received_state)
            stop_event.set()

        await poll_orca_states(
            "localhost:50051", publish, stop_event, interval=0.0
        )

        self.assertEqual(len(published), 1)
        for received, expected in zip(published[0], state):
            np.testing.assert_array_equal(received, expected)
        gym_class.assert_called_once_with(stub_class.return_value)
        channel.close.assert_awaited_once_with()

    @patch("mtool.model_sources.OrcaGymRemote")
    @patch("mtool.model_sources.GrpcServiceStub")
    @patch("mtool.model_sources.grpc.aio.insecure_channel")
    async def test_query_error_propagates_and_closes_channel(
        self, insecure_channel, _stub_class, gym_class
    ):
        channel = insecure_channel.return_value
        channel.close = AsyncMock()
        error = RuntimeError("query failed")
        gym_class.return_value.query_all_qpos_qvel_qacc = AsyncMock(
            side_effect=error
        )

        with self.assertRaises(RuntimeError) as caught:
            await poll_orca_states(
                "localhost:50051", lambda _state: None, threading.Event()
            )

        self.assertIs(caught.exception, error)
        channel.close.assert_awaited_once_with()


class StreamOrcaStatesTests(unittest.IsolatedAsyncioTestCase):
    @patch("mtool.model_sources.OrcaGymLocal")
    @patch("mtool.model_sources.OrcaGymRemote")
    @patch("mtool.model_sources.GrpcServiceStub")
    @patch("mtool.model_sources.grpc.aio.insecure_channel")
    async def test_stream_pauses_once_and_pushes_visual_state(
        self, insecure_channel, stub_class, gym_class, visual_gym_class
    ):
        channel = insecure_channel.return_value
        channel.close = AsyncMock()
        gym = gym_class.return_value
        visual_gym = visual_gym_class.return_value
        events = []
        gym.pause_simulation = AsyncMock(side_effect=lambda: events.append("pause"))
        gym.set_qpos = AsyncMock()
        gym.set_qvel = AsyncMock()
        gym.mj_forward = AsyncMock()
        visual_gym.push_visual_state = AsyncMock(
            side_effect=lambda qpos, time: events.append(("visual", qpos, time))
        )
        qpos = np.array([1.0, 2.0])
        state_queue = queue.Queue()
        state_queue.put((7, qpos, 1.25))
        stop_event = threading.Event()
        pushed = []

        def on_pushed(request_id):
            pushed.append(request_id)
            stop_event.set()

        await stream_orca_states(
            "localhost:50051",
            state_queue,
            stop_event,
            on_pushed,
            timeout=3.5,
        )

        self.assertEqual(events[0], "pause")
        self.assertEqual(events[1][0], "visual")
        self.assertIs(events[1][1], qpos)
        self.assertEqual(events[1][2], 1.25)
        self.assertEqual(len(events), 2)
        self.assertEqual(pushed, [7])
        gym.pause_simulation.assert_awaited_once_with()
        gym.set_qpos.assert_not_awaited()
        gym.set_qvel.assert_not_awaited()
        gym.mj_forward.assert_not_awaited()
        insecure_channel.assert_called_once_with(
            "localhost:50051",
            options=(("grpc.max_receive_message_length", 64 * 1024 * 1024),),
        )
        gym_class.assert_called_once_with(stub_class.return_value)
        visual_gym_class.assert_called_once_with(stub_class.return_value)
        channel.close.assert_awaited_once_with()

    @patch("mtool.model_sources.OrcaGymLocal")
    @patch("mtool.model_sources.OrcaGymRemote")
    @patch("mtool.model_sources.GrpcServiceStub")
    @patch("mtool.model_sources.grpc.aio.insecure_channel")
    async def test_stream_error_propagates_and_closes_channel(
        self, insecure_channel, _stub_class, gym_class, visual_gym_class
    ):
        channel = insecure_channel.return_value
        channel.close = AsyncMock()
        gym = gym_class.return_value
        error = RuntimeError("push failed")
        gym.pause_simulation = AsyncMock()
        visual_gym_class.return_value.push_visual_state = AsyncMock(side_effect=error)
        state_queue = queue.Queue()
        state_queue.put((7, np.zeros(2), 1.25))

        with self.assertRaises(RuntimeError) as caught:
            await stream_orca_states(
                "localhost:50051",
                state_queue,
                threading.Event(),
                lambda _request_id: None,
            )

        self.assertIs(caught.exception, error)
        channel.close.assert_awaited_once_with()


class FetchOrcaXmlTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.home = Path(self.temporary_directory.name)
        self.cache_root = self.home / ".cache" / "mujoco-check" / "orcalab"

    def test_success_uses_independent_directories_and_preserves_them_until_cleanup(self):
        async def fake_download(_address, directory, timeout):
            self.assertEqual(timeout, 4.5)
            xml_path = Path(directory) / "scene.xml"
            xml_path.write_text("<mujoco/>", encoding="utf-8")
            return xml_path

        with (
            patch("mtool.model_sources.Path.home", return_value=self.home),
            patch(
                "mtool.model_sources.download_orca_xml", side_effect=fake_download
            ) as download,
        ):
            first_xml, first_dir = fetch_orca_xml("localhost:50051", timeout=4.5)
            second_xml, second_dir = fetch_orca_xml("localhost:50052", timeout=4.5)

            self.assertNotEqual(first_dir, second_dir)
            self.assertEqual(first_dir.parent, self.cache_root)
            self.assertEqual(second_dir.parent, self.cache_root)
            self.assertEqual(first_xml, first_dir / "scene.xml")
            self.assertEqual(second_xml, second_dir / "scene.xml")
            self.assertTrue(first_xml.is_file())
            self.assertTrue(second_xml.is_file())
            download.assert_has_awaits(
                [
                    call("localhost:50051", first_dir, timeout=4.5),
                    call("localhost:50052", second_dir, timeout=4.5),
                ]
            )
            self.assertFalse((self.home / ".orcagym" / "tmp").exists())

            cleanup_orca_fetch(first_dir)
            self.assertFalse(first_dir.exists())
            self.assertTrue(second_dir.exists())
            cleanup_orca_fetch(second_dir)
            self.assertFalse(second_dir.exists())

    def test_failure_removes_request_directory_and_propagates_original_error(self):
        error = RuntimeError("download failed")
        request_directories = []

        async def fail_download(_address, directory, timeout):
            self.assertEqual(timeout, 7.0)
            request_directories.append(Path(directory))
            (Path(directory) / "partial.xml").write_text("partial", encoding="utf-8")
            raise error

        with (
            patch("mtool.model_sources.Path.home", return_value=self.home),
            patch("mtool.model_sources.download_orca_xml", side_effect=fail_download),
        ):
            with self.assertRaises(RuntimeError) as caught:
                fetch_orca_xml("localhost:50051", timeout=7.0)

        self.assertIs(caught.exception, error)
        self.assertEqual(len(request_directories), 1)
        self.assertEqual(request_directories[0].parent, self.cache_root)
        self.assertFalse(request_directories[0].exists())
        self.assertTrue(self.cache_root.is_dir())
        self.assertEqual(list(self.cache_root.iterdir()), [])
        self.assertFalse((self.home / ".orcagym" / "tmp").exists())


if __name__ == "__main__":
    unittest.main()
