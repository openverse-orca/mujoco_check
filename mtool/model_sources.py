import asyncio
import queue
import shutil
import tempfile
from pathlib import Path

import grpc
from orca_gym.core import orca_gym_local
from orca_gym.core.orca_gym_local import OrcaGymLocal
from orca_gym.core.orca_gym_remote import OrcaGymRemote
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub


ORCA_GRPC_OPTIONS = (("grpc.max_receive_message_length", 64 * 1024 * 1024),)


async def download_orca_xml(address, directory, timeout=30.0):
    channel = grpc.aio.insecure_channel(address, options=ORCA_GRPC_OPTIONS)
    try:
        stub = GrpcServiceStub(channel)
        gym = OrcaGymLocal(stub, xml_assets_dir=str(directory))
        original_file_lock = orca_gym_local.file_lock

        def non_blocking_file_lock(file_path, timeout=30, non_blocking=False):
            return original_file_lock(
                file_path, timeout=timeout, non_blocking=True
            )

        orca_gym_local.file_lock = non_blocking_file_lock
        try:
            xml_path = await asyncio.wait_for(gym.load_model_xml(), timeout=timeout)
            return Path(xml_path)
        finally:
            orca_gym_local.file_lock = original_file_lock
    finally:
        await channel.close()


async def poll_orca_states(
    address, on_state, stop_event, interval=1.0 / 60.0, timeout=2.0
):
    channel = grpc.aio.insecure_channel(address, options=ORCA_GRPC_OPTIONS)
    try:
        gym = OrcaGymRemote(GrpcServiceStub(channel))
        while not stop_event.is_set():
            state = await asyncio.wait_for(
                gym.query_all_qpos_qvel_qacc(), timeout=timeout
            )
            if stop_event.is_set():
                break
            on_state(state)
            await asyncio.sleep(interval)
    finally:
        await channel.close()


async def stream_orca_states(
    address,
    state_queue,
    stop_event,
    on_pushed,
    interval=1.0 / 60.0,
    timeout=5.0,
):
    channel = grpc.aio.insecure_channel(address, options=ORCA_GRPC_OPTIONS)
    try:
        stub = GrpcServiceStub(channel)
        gym = OrcaGymRemote(stub)
        visual_gym = OrcaGymLocal(stub)
        await asyncio.wait_for(gym.pause_simulation(), timeout=timeout)
        while not stop_event.is_set():
            try:
                request_id, qpos, time = state_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(interval)
                continue
            await asyncio.wait_for(
                visual_gym.push_visual_state(qpos, time), timeout=timeout
            )
            on_pushed(request_id)
    finally:
        await channel.close()


def cleanup_orca_fetch(cache_dir):
    shutil.rmtree(cache_dir, ignore_errors=True)


def fetch_orca_xml(address, timeout=30.0):
    cache_root = Path.home() / ".cache" / "mujoco-check" / "orcalab"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(dir=cache_root))
    try:
        xml_path = asyncio.run(
            download_orca_xml(address, cache_dir, timeout=timeout)
        )
    except BaseException:
        cleanup_orca_fetch(cache_dir)
        raise
    return xml_path, cache_dir
