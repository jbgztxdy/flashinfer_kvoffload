# Purpose: KV cache for the tiered sparse MLA benchmarks on H800: one CUDA VA range of near pages
#          (local HBM) followed by far pages (pinned Host NUMA DRAM over PCIe, or a peer GPU's HBM
#          over NVLink), plus an HBM-only alias of the near pages with a compact staging tail for
#          the Memcpy baseline. Also builds top-k page ids from the measured DSA-proxy captures.
import ctypes
import glob
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_glm52_sparse_decode_p2p import (  # noqa: E402
    AccessDescriptor,
    AllocationProperties,
    DLDataType,
    DLDevice,
    DLManagedTensor,
    DLTensor,
    MemoryLocation,
    _dlpack_owners,
    _new_capsule,
    release_dlpack,
)

ROW = 576
ROW_BYTES = ROW * 2
# 16384 rows are 18 MiB, a multiple of the 2 MiB VMM granularity of both locations.
QUANTUM = 16384
RECENT = 4096  # positions this close to the query always stay near (sliding window)


def round_up(value, quantum=QUANTUM):
    return max(quantum, (value + quantum - 1) // quantum * quantum)


class TieredKV:
    """main: [near | far] for the kernels (far pages start at `near`); alias: [near | staging]."""

    def __init__(self, near, far, staging, device, far_kind="host", peer=None):
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.near, self.far, self.staging = round_up(near), round_up(far), round_up(staging)
        self.device, self.far_kind, self.peer = device, far_kind, peer
        self.handles, self.mappings, self.reservations = [], [], []
        self.residency = []
        self._call("cuInit", ctypes.c_uint(0))
        torch.empty(1, device=f"cuda:{device}")
        if far_kind == "peer":
            torch.empty(1, device=f"cuda:{peer}")
            far_location = MemoryLocation(1, peer)
        elif far_kind == "host":
            numa = ctypes.c_int()
            self._call("cuDeviceGetAttribute", ctypes.byref(numa), ctypes.c_int(134), ctypes.c_int(device))
            self.host_numa = max(0, numa.value)
            far_location = MemoryLocation(3, self.host_numa)
        else:
            raise ValueError(far_kind)
        local = MemoryLocation(1, device)
        try:
            near_handle = self._create(local, self.near)
            far_handle = self._create(far_location, self.far)
            staging_handle = self._create(local, self.staging)
            self.main_address = self._reserve(self.near + self.far)
            far_access = [AccessDescriptor(local, 3)]
            if far_kind == "host":
                far_access.append(AccessDescriptor(MemoryLocation(3, self.host_numa), 3))
            elif peer != device:
                far_access.append(AccessDescriptor(MemoryLocation(1, peer), 3))
            self._map(self.main_address, 0, self.near, near_handle, [AccessDescriptor(local, 3)], "near_hbm")
            self._map(self.main_address, self.near, self.far, far_handle, far_access,
                      "far_host_dram" if far_kind == "host" else f"far_peer_hbm_gpu{peer}")
            self.alias_address = self._reserve(self.near + self.staging)
            self._map(self.alias_address, 0, self.near, near_handle, [AccessDescriptor(local, 3)], "near_hbm_alias")
            self._map(self.alias_address, self.near, self.staging, staging_handle, [AccessDescriptor(local, 3)],
                      "staging_hbm")
        except Exception:
            self.close()
            raise

    def _call(self, name, *args):
        status = getattr(self.driver, name)(*args)
        if status:
            error = ctypes.c_char_p()
            self.driver.cuGetErrorString(ctypes.c_int(status), ctypes.byref(error))
            raise RuntimeError(f"{name}: {error.value.decode()} ({status})")

    def _create(self, location, rows):
        prop = AllocationProperties()
        prop.type = 1  # pinned
        prop.location = location
        granularity = ctypes.c_size_t()
        self._call("cuMemGetAllocationGranularity", ctypes.byref(granularity), ctypes.byref(prop), ctypes.c_int(0))
        assert rows * ROW_BYTES % granularity.value == 0, (rows, granularity.value)
        handle = ctypes.c_uint64()
        self._call("cuMemCreate", ctypes.byref(handle), ctypes.c_size_t(rows * ROW_BYTES), ctypes.byref(prop),
                   ctypes.c_uint64(0))
        self.handles.append(handle)
        return handle

    def _reserve(self, rows):
        address = ctypes.c_uint64()
        self._call("cuMemAddressReserve", ctypes.byref(address), ctypes.c_size_t(rows * ROW_BYTES),
                   ctypes.c_size_t(2 << 20), ctypes.c_uint64(0), ctypes.c_uint64(0))
        self.reservations.append((address, rows * ROW_BYTES))
        return address.value

    def _map(self, base, start, rows, handle, access, tier):
        address, size = base + start * ROW_BYTES, rows * ROW_BYTES
        self._call("cuMemMap", ctypes.c_uint64(address), ctypes.c_size_t(size), ctypes.c_size_t(0), handle,
                   ctypes.c_uint64(0))
        self.mappings.append((address, size))
        descriptors = (AccessDescriptor * len(access))(*access)
        self._call("cuMemSetAccess", ctypes.c_uint64(address), ctypes.c_size_t(size), descriptors,
                   ctypes.c_size_t(len(access)))
        actual = AllocationProperties()
        self._call("cuMemGetAllocationPropertiesFromHandle", ctypes.byref(actual), handle)
        self.residency.append(dict(tier=tier, address=hex(address), bytes=size, location_type=actual.location.type,
                                   location_id=actual.location.id))

    def _tensor(self, address, rows):
        shape = (ctypes.c_int64 * 3)(rows, 1, ROW)
        managed = DLManagedTensor()
        managed.tensor = DLTensor(address, DLDevice(2, self.device), 3, DLDataType(4, 16, 1), shape, None, 0)
        managed.deleter = release_dlpack
        pointer = ctypes.addressof(managed)
        _dlpack_owners[pointer] = (managed, shape, self)
        return torch.utils.dlpack.from_dlpack(_new_capsule(pointer, b"dltensor", None))

    def main(self):
        return self._tensor(self.main_address, self.near + self.far)

    def alias(self):
        return self._tensor(self.alias_address, self.near + self.staging)

    def far_cpu(self):
        """CPU view of the far pages (host tier only), for the Memcpy CPU gather."""
        assert self.far_kind == "host"
        address = self.main_address + self.near * ROW_BYTES
        buffer = (ctypes.c_uint16 * (self.far * ROW)).from_address(address)
        return torch.frombuffer(buffer, dtype=torch.bfloat16).view(self.far, 1, ROW)

    def close(self):
        for address, size in self.mappings:
            self._call("cuMemUnmap", ctypes.c_uint64(address), ctypes.c_size_t(size))
        self.mappings = []
        for handle in self.handles:
            self._call("cuMemRelease", handle)
        self.handles = []
        for address, size in self.reservations:
            self._call("cuMemAddressFree", address, ctypes.c_size_t(size))
        self.reservations = []


def load_captures(pattern):
    captures = []
    for path in sorted(glob.glob(pattern)):
        data = torch.load(path)
        captures.append((Path(path).stem, data["selections"], data["group_starts"]))
    if not captures:
        raise FileNotFoundError(pattern)
    return captures


class Requests:
    """Request r replays one (document, layer, group) capture: its query at position start + t
    selects capture token t. Near page of position p: (r % near_requests) * context + p; far page:
    near + r * context + p. Residency is a fixed property of (request, position): positions older
    than RECENT are far with probability theta, chosen so token 0's far share is `fraction`;
    fraction 1 makes every position far except the newest token, as under HiSparse."""

    def __init__(self, captures, requests, context, near_rows, near_requests, fraction, seed, device):
        import random
        rng = random.Random(seed)
        combos = [(c, l, g) for c in range(len(captures))
                  for l in range(captures[c][1].shape[0]) for g in range(captures[c][1].shape[1])]
        rng.shuffle(combos)
        self.context, self.near_rows, self.near_requests = context, near_rows, near_requests
        self.fraction, self.device = fraction, device
        self.sel, self.start, self.far_mask = [], [], []
        for r in range(requests):
            doc, layer, group = combos[r % len(combos)]
            _, selections, starts = captures[doc]
            sel = selections[layer, group].long()  # [tokens, topk] positions
            start = int(starts[group])
            assert int(sel.max()) < context
            self.sel.append(sel)
            self.start.append(start)
            if fraction >= 1.0:
                far = torch.ones(context, dtype=torch.bool)
            else:
                old = sel[0] < start - RECENT
                share = float(old.float().mean())
                theta = min(1.0, fraction / share) if share else 0.0
                u = torch.rand(context, generator=torch.Generator().manual_seed(seed * 7919 + r))
                far = (torch.arange(context) < start - RECENT) & (u < theta)
            self.far_mask.append(far)
        self.tokens = self.sel[0].shape[0]

    def rows(self, t0, mtp):
        """Page ids [requests * mtp, topk] of tokens t0 .. t0 + mtp - 1, and their positions."""
        pages, positions, far_refs = [], [], 0
        for r, (sel, start, far) in enumerate(zip(self.sel, self.start, self.far_mask)):
            s = sel[t0 : t0 + mtp]
            newest = start + t0 + torch.arange(mtp).view(-1, 1)
            on_far = far[s] & (s < newest)
            far_refs += int(on_far.sum())
            near_page = (r % self.near_requests) * self.context + s
            far_page = self.near_rows + r * self.context + s
            pages.append(torch.where(on_far, far_page, near_page))
            positions.append(s)
        return torch.cat(pages).int(), torch.cat(positions).int(), far_refs
