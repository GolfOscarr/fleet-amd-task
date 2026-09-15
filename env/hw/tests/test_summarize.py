"""env/hw/summarize.py against synthetic raw/ directories.

Three machines: one that passes every row, one with a CPX partition, a slow E1,
an XCD rule violation, a system-scope __threadfence(), no PMC output and no
system torch, and one with nothing collected at all. The fixtures are written by
hand in the output shapes of rocminfo, amd-smi, rocprofv3 and the probes.
"""
import sys
from pathlib import Path

import pytest

HW = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW))
import summarize  # noqa: E402


# ---------------------------------------------------------------------------
# fixture text

ROCMINFO = """\
ROCk module version 6.12.12 is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.15
Machine Model:           LARGE
==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    AMD EPYC 9534 64-Core Processor
  Marketing Name:          AMD EPYC 9534 64-Core Processor
  Vendor Name:             CPU
  Device Type:             CPU
  Compute Unit:            26
  Cache Info:
    L1:                      32768(0x8000) KB
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS: FINE GRAINED
      Size:                    469762048(0x1c000000) KB
*******
Agent 2
*******
  Name:                    gfx942
  Uuid:                    GPU-1b2c3d4e5f607182
  Marketing Name:          AMD Instinct MI300X
  Vendor Name:             AMD
  Device Type:             GPU
  Node:                    1
  Cache Info:
    L1:                      32(0x20) KB
    L2:                      4096(0x1000) KB
  Chip ID:                 29857(0x74a1)
  Cacheline Size:          128(0x80)
  Max Clock Freq. (MHz):   2100
  BDFID:                   3072
  Compute Unit:            304
  SIMDs per CU:            4
  Shader Engines:          32
  Wavefront Size:          64(0x40)
  Workgroup Max Size:      1024(0x400)
  Max Waves Per CU:        32
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
      Size:                    201326592(0xc000000) KB
      Allocatable:             TRUE
    Pool 2
      Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
      Size:                    201326592(0xc000000) KB
    Pool 3
      Segment:                 GROUP
      Size:                    64(0x40) KB
      Allocatable:             FALSE
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
      Workgroup Max Size:      1024(0x400)
*** Done ***
"""

KFD = """\
== /sys/class/kfd/kfd/topology/nodes/0/
cpu_cores_count 26
simd_count 0
mem_banks_count 1
caches_count 0
max_waves_per_simd 0
lds_size_in_kb 0
wave_front_size 0
simd_per_cu 0
num_xcc 0
max_engine_clk_fcompute 0
== /sys/class/kfd/kfd/topology/nodes/1/
cpu_cores_count 0
simd_count 1216
mem_banks_count 1
caches_count 316
max_waves_per_simd 8
lds_size_in_kb 64
gds_size_in_kb 0
wave_front_size 64
array_count 32
simd_arrays_per_engine 1
cu_per_simd_array 2
simd_per_cu 4
num_xcc 8
max_engine_clk_fcompute 2100000
"""

AMDSMI_CLOCK = """\
GPU: 0
    CLOCK:
        GFX_0:
            CLK: 141 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 2100 MHz
            CLK_LOCKED: DISABLED
        MEM_0:
            CLK: 900 MHz
            MIN_CLK: 900 MHz
            MAX_CLK: 1300 MHz
"""

def rocprof_block(name, dimensions, expression=None):
    """One counter block of a rocprofv3 --list-avail listing."""
    lines = [f"Counter_Name        :\t{name}",
             f"Description         :\tsynthetic fixture for {name}",
             "Block               :\tTCC"]
    if expression:
        lines.append(f"Expression          :\t{expression}")
    lines.append(f"Dimensions          :\t{dimensions}")
    return "\n".join(lines) + "\n\n"


# rocprofv3 reports instances as a shape, not as one row per instance:
# 16 channels on each of 8 XCDs is the 128 of I5.
ROCPROF_LIST = (
    "# listing command: rocprofv3 --list-avail\n"
    "gpu-agent2:\n\n"
    + rocprof_block("TCC_EA0_RDREQ", "DIMENSION_INSTANCE[0:15] DIMENSION_XCC[0:7]")
    + rocprof_block("TCC_EA0_RDREQ_32B", "DIMENSION_INSTANCE[0:15] DIMENSION_XCC[0:7]")
    + "".join(rocprof_block(name, "DIMENSION_INSTANCE[0:0]",
                            expression=f"reduce({name[:-4]},sum)")
              for name in summarize.PMC_NAMES))

PMC_HEADER = ('"Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id",'
              '"Thread_Id","Grid_Size","Kernel_Id","Kernel_Name","Workgroup_Size",'
              '"LDS_Block_Size","Scratch_Size","VGPR_Count","SGPR_Count",'
              '"Counter_Name","Counter_Value"')

KTRACE = ('"Kind","Agent_Id","Queue_Id","Kernel_Id","Kernel_Name","Correlation_Id",'
          '"Start_Timestamp","End_Timestamp"\n'
          '"KERNEL_DISPATCH",1,1,42,"copy_kernel(float*, float const*, unsigned long)",'
          '1,1000000,1512480\n')


def pmc_csv(pairs):
    """A rocprofv3 counter_collection CSV, each counter split over two dispatches."""
    lines = [PMC_HEADER]
    for dispatch in (1, 2):
        for name, total in pairs:
            lines.append(
                '{d},{d},1,1,4242,4242,1048576,42,"copy_kernel(float*, float const*, '
                'unsigned long)",256,0,0,8,16,"{n}",{v}'.format(
                    d=dispatch, n=name, v=total / 2.0))
    return "\n".join(lines) + "\n"


GRID_SPECS = [(296, 0), (8, 0), (8, 1), (304, 0), (608, 0), (1000, 0), (37, 0)]


def xcc_map_text(violations_296=0, offset=0):
    """One line per run. With a non-zero offset the placement is still a round
    robin over all eight XCDs, but every block breaks the strict mod-8 rule,
    which is what the machine measured."""
    lines = []
    for grid, concurrent in GRID_SPECS:
        per_xcd = summarize.expected_per_xcd(grid, offset)
        violations = grid if offset else (violations_296 if grid == 296 else 0)
        for run in range(3):
            lines.append(
                "xcc_map grid={g} run={r} concurrent={c} distinct=8 ids_in_range=1 "
                "rule_mod8_violations={v} per_xcd={p} stable_vs_run0=1".format(
                    g=grid, r=run, c=concurrent, v=violations,
                    p=",".join(str(x) for x in per_xcd)))
    return "\n".join(lines) + "\n"


def xcc_dump_text(grid, offset=0):
    """An xcc_map --dump CSV: one "block,xcd" line per block."""
    return "".join(f"{b},{(b + offset) % 8}\n" for b in range(grid))


def stream_read_line(**kv):
    base = {"occupancy": "full", "grid": 608, "block": 256, "lds_bytes": 0,
            "blocks_per_cu_query": 2, "size_mib": 1024, "unroll": 8, "ms": 0.275,
            "gbps": 3900.0, "tbps": 3.9, "checksum": "0x5f5e100"}
    base.update(kv)
    return "stream_read " + " ".join(f"{k}={v}" for k, v in base.items())


def e1_text(tbps=(3.9001, 3.9110, 3.8950)):
    return "\n".join(stream_read_line(tbps=t, gbps=round(t * 1000.0, 1))
                     for t in tbps) + "\n"


def unroll_sweep(gbps_by_n, occupancy="one", grid=304):
    return "\n".join(
        stream_read_line(occupancy=occupancy, grid=grid, lds_bytes=33792,
                         blocks_per_cu_query=1, unroll=n, gbps=g,
                         tbps=round(g / 1000.0, 4))
        for n, g in gbps_by_n) + "\n"


def size_sweep(gbps_by_size):
    return "\n".join(
        stream_read_line(size_mib=w, gbps=g, tbps=round(g / 1000.0, 4))
        for w, g in gbps_by_size) + "\n"


FENCE_AGENT = """\
fence kernel=k_release wbl2_sc1=1 inv_sc1=0 wbl2_sc0_sc1=0 inv_sc0_sc1=0 load_sc1=0
fence kernel=k_acquire wbl2_sc1=0 inv_sc1=1 wbl2_sc0_sc1=0 inv_sc0_sc1=0 load_sc1=0
fence kernel=k_threadfence wbl2_sc1=1 inv_sc1=1 wbl2_sc0_sc1=0 inv_sc0_sc1=0 load_sc1=0
fence kernel=k_atomic_load wbl2_sc1=0 inv_sc1=1 wbl2_sc0_sc1=0 inv_sc0_sc1=0 load_sc1=1
"""

FENCE_SYSTEM = FENCE_AGENT.replace(
    "fence kernel=k_threadfence wbl2_sc1=1 inv_sc1=1 wbl2_sc0_sc1=0 inv_sc0_sc1=0 load_sc1=0",
    "fence kernel=k_threadfence wbl2_sc1=0 inv_sc1=0 wbl2_sc0_sc1=1 inv_sc0_sc1=1 load_sc1=0")

CHASE = """\
chase mode=size size_mib=1 loads=1048576 ns_per_load_wall=210.4 ns_per_load_memrealtime=209.8 rate_khz=100000
chase mode=size size_mib=64 loads=1048576 ns_per_load_wall=320.7 ns_per_load_memrealtime=319.9 rate_khz=100000
chase mode=size size_mib=1024 loads=1048576 ns_per_load_wall=540.2 ns_per_load_memrealtime=539.1 rate_khz=100000
chase mode=fence kind=release xcds=1 ns_per_iter_agent=812.5 ns_per_iter_workgroup=98.2 ns_fence_cost=714.3
chase mode=fence kind=acquire xcds=1 ns_per_iter_agent=690.1 ns_per_iter_workgroup=97.9 ns_fence_cost=592.2
chase mode=fence kind=release xcds=8 ns_per_iter_agent=1204.6 ns_per_iter_workgroup=101.3 ns_fence_cost=1103.3
chase mode=fence kind=acquire xcds=8 ns_per_iter_agent=980.4 ns_per_iter_workgroup=100.8 ns_fence_cost=879.6
chase mode=pingpong xcd_a=0 xcd_b=1 rounds=10000 ns_one_way=186.7
"""

BABELSTREAM_FAST = """\
BabelStream
Version: 5.0
Implementation: HIP
Running kernels 50 times
Precision: double
Array size: 2147.5 MB (=2.1 GB)
Using HIP device AMD Instinct MI300X
Function    MBytes/sec  Min (sec)   Max         Average
Copy        4310000.0   0.00100     0.00110     0.00105
Mul         4290000.0   0.00100     0.00110     0.00105
Add         4350000.0   0.00148     0.00150     0.00149
Triad       4340000.0   0.00148     0.00150     0.00149
Dot         3900000.0   0.00110     0.00120     0.00115
"""


def base_files():
    """The raw/ of a machine that meets every expectation in the checklist."""
    return {
        "rocminfo.txt": ROCMINFO,
        "amd-smi-version.txt": "AMDSMI_TOOL_VERSION: 25.3.0\nROCM_VERSION: 7.0.0\n",
        "amd-smi-list.txt": "GPU: 0\n    BDF: 0000:0c:00.0\n    UUID: 1b2c3d4e-0000\n",
        "amd-smi-static-bus.txt": "GPU: 0\n    BUS:\n        BDF: 0000:0c:00.0\n"
                                  "        MAX_PCIE_WIDTH: 16\n",
        "amd-smi-static-driver.txt": "GPU: 0\n    DRIVER:\n        NAME: amdgpu\n"
                                     "        VERSION: 6.12.12\n",
        "hipcc-version.txt": "HIP version: 7.0.51831-a2e4e0c4\n"
                             "AMD clang version 20.0.0git (roc-7.0.0 25292)\n"
                             "Target: x86_64-unknown-linux-gnu\n",
        "rocm-version.txt": "7.0.0-63\n",
        "rocm-include.txt": "amd_comgr\nck\nck_tile\nhip\nhsa\n"
                            "ck_tile: present\nck: present\n",
        "uname.txt": "Linux mi300x-vm 6.8.0-45-generic #45-Ubuntu SMP x86_64 GNU/Linux\n",
        "os-release.txt": 'PRETTY_NAME="Ubuntu 22.04.5 LTS"\nNAME="Ubuntu"\n'
                          'VERSION_ID="22.04"\n',
        "amdgpu-version.txt": "6.12.12\n",
        "docker-version.txt": "Docker version 27.3.1, build ce12230\n",
        "python-version.txt": "Python 3.10.12\n",
        "python-torch.txt": "2.5.1+rocm7.0\n",

        "kfd-topology.txt": KFD,
        # Both residencies land in one file: register-only, then the 58,368 B
        # dynamic LDS request the runtime launches the worker kernel with.
        "occupancy.txt": "occupancy requested_vgprs=182 actual_vgprs=184 "
                         "dynamic_lds=0 blocks_per_cu=2 waves_per_simd=2 "
                         "cus=304 coresident_blocks=608\n"
                         "occupancy requested_vgprs=182 actual_vgprs=184 "
                         "dynamic_lds=58368 blocks_per_cu=1 waves_per_simd=1 "
                         "cus=304 coresident_blocks=304\n",
        "wallclock.txt": "wallclock rate_khz=100000 ticks=10000000 host_ns=100000000 "
                         "ratio=1.0000\n",

        "partition.txt": "GPU: 0\n    COMPUTE_PARTITION: SPX\n"
                         "    MEMORY_PARTITION: NPS1\n",
        "partition-cmd.txt": "amd-smi static --partition\n",
        "amd-smi-set-help.txt": "usage: amd-smi set [-h] [-g GPU]\n"
                                "  -C, --compute-partition {SPX,DPX,TPX,QPX,CPX}\n",

        "amd-smi-metric-clock.txt": AMDSMI_CLOCK,
        "amd-smi-metric-power.txt": "GPU: 0\n    POWER:\n        SOCKET_POWER: 137 W\n"
                                    "        GFX_VOLTAGE: 806 mV\n",
        "amd-smi-metric-mem-usage.txt": "GPU: 0\n    MEM_USAGE:\n"
                                        "        TOTAL_VRAM: 196592 MB\n"
                                        "        USED_VRAM: 283 MB\n"
                                        "        FREE_VRAM: 196309 MB\n",
        "amd-smi-metric-temp.txt": "GPU: 0\n    TEMPERATURE:\n        EDGE: 38 C\n"
                                   "        HOTSPOT: 45 C\n        MEM: 42 C\n",
        "amd-smi-metric-throttle.txt": "GPU: 0\n    THROTTLE:\n"
                                       "        ACCUMULATION_COUNTER: 0\n"
                                       "        PROCHOT_ACCUMULATED: 0\n"
                                       "        PPT_ACCUMULATED: 0\n",
        "amd-smi-static-limit.txt": "GPU: 0\n    LIMIT:\n        MAX_POWER: 750 W\n"
                                    "        MIN_POWER: 0 W\n",
        "amd-smi-static-vram.txt": "GPU: 0\n    VRAM:\n        TYPE: HBM3\n"
                                   "        SIZE: 196592 MB\n        BIT_WIDTH: 8192\n",
        "amd-smi-process.txt": "GPU: 0\n    PROCESS_LIST: No running processes detected\n",
        "rocminfo-max-clock.txt": "  Max Clock Freq. (MHz):   2100\n",

        "e1.txt": e1_text(),
        "e2.txt": unroll_sweep([(1, 1800.0), (2, 2900.0), (4, 3600.0), (8, 3700.0),
                                (16, 3720.0), (32, 3730.0)]),
        "e3.txt": unroll_sweep([(1, 2600.0), (2, 3650.0), (4, 3700.0), (8, 3720.0),
                                (16, 3730.0), (32, 3740.0)], occupancy="grid", grid=608),
        "e4.txt": size_sweep([(4, 12000.0), (16, 11800.0), (32, 11500.0), (64, 5200.0),
                              (128, 5100.0), (256, 5050.0), (512, 3950.0),
                              (1024, 3900.0)]),
        "babelstream.txt": BABELSTREAM_FAST,
        "rocm-bandwidth-test.txt": "Unidirectional copy peak bandwidth GB/s\n"
                                   "Device conn  Device 0     Device 1\n"
                                   "0            N/A          55.123\n"
                                   "1            48.900       N/A\n",

        "xcc-map.txt": xcc_map_text(),
        "xcc-map-296.csv": xcc_dump_text(296),
        "xcc-map-8.csv": xcc_dump_text(8),
        "fence.txt": FENCE_AGENT,
        "chase.txt": CHASE,
        "rocprof-list.txt": ROCPROF_LIST,

        "lscpu.txt": "Architecture:            x86_64\nCPU(s):                  26\n"
                     "Model name:              AMD EPYC 9534 64-Core Processor\n",
        "nproc.txt": "26\n",
        "free.txt": "               total        used        free      shared\n"
                    "Mem:             448          12         420           0\n"
                    "Swap:              0           0           0\n",
        "df-root.txt": "Filesystem      Size  Used Avail Use% Mounted on\n"
                       "/dev/sda1        13T  1.1T   12T   9% /\n",
        "df-home.txt": "Filesystem      Size  Used Avail Use% Mounted on\n"
                       "/dev/sda1        13T  1.1T   12T   9% /home/hotaisle\n",
        "download-log.txt": "Downloading model-00001-of-000004.safetensors: 100%|###| "
                            "8.59G/8.59G [02:31<00:00, 56.7MB/s]\n",
        "docker-gpu.txt": "Agent 2\n*******\n  Name:                    gfx942\n"
                          "  Marketing Name:          AMD Instinct MI300X\n",
        "copy-bytes.txt": "copy_bytes bytes=1073741824 ms=0.512 gbps=4194.3\n",
    }


# A 1 GiB copy each way, the way the machine reports it: every read is a 128 B
# TCC_BUBBLE request, so 2^23 of them are 1 GiB, and every write is a 64 B
# request, so 2^24 of them are 1 GiB. The flat 64 B per read request that
# measure.py used before the collection lands at half, which is the point of I3.
PMC_DIRS = {
    "pmc1": [("TCC_EA0_RDREQ_sum", float(1 << 23)), ("TCC_EA0_RDREQ_32B_sum", 0.0)],
    "pmc2": [("TCC_EA0_WRREQ_sum", float(1 << 24)),
             ("TCC_EA0_WRREQ_64B_sum", float(1 << 24))],
    "pmc3": [("TCC_HIT_sum", 100.0), ("TCC_MISS_sum", float(1 << 24))],
    "pmc4": [("TCC_BUBBLE_sum", float(1 << 23)), ("TCC_EA0_RDREQ_sum", float(1 << 23))],
}


def write_machine(tmp_path, files, with_pmc=True, with_ktrace=True):
    out = tmp_path
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (raw / name).write_text(text)
    if with_pmc:
        for name, pairs in PMC_DIRS.items():
            d = raw / name / "pmc_1"
            d.mkdir(parents=True, exist_ok=True)
            (d / "mi300x-vm_4242_counter_collection.csv").write_text(pmc_csv(pairs))
    if with_ktrace:
        d = raw / "ktrace" / "trace"
        d.mkdir(parents=True, exist_ok=True)
        (d / "mi300x-vm_4242_kernel_trace.csv").write_text(KTRACE)
    return out


def results(rows):
    return {row["id"]: row["result"] for row in rows}


def measured(rows):
    return {row["id"]: row["measured"] for row in rows}


# ---------------------------------------------------------------------------
# the parsers on their own


def test_key_value_parser_reads_a_probe_line():
    rows = summarize.parse_kv_lines(
        "stream_read occupancy=full grid=608 size_mib=1024 gbps=3900.1 checksum=0x5f\n"
        "noise that is not a measurement\n"
        "UNAVAILABLE: exit 127: stream_read --size 1G\n"
        "chase mode=size size_mib=1 ns_per_load_wall=210.4\n")
    assert [r["_name"] for r in rows] == ["stream_read", "chase"]
    assert rows[0]["occupancy"] == "full"
    assert rows[0]["gbps"] == "3900.1"
    assert rows[0]["checksum"] == "0x5f"
    assert rows[1]["mode"] == "size"
    assert summarize.named(rows, "chase") == [rows[1]]


def test_key_value_parser_ignores_prose_and_blank_lines():
    assert summarize.parse_kv_lines("") == []
    assert summarize.parse_kv_lines("hipcc: command not found\n\n") == []


def test_fence_parser_keys_by_kernel_and_returns_integers():
    fence = summarize.parse_fence(FENCE_AGENT)
    assert set(fence) == {"k_release", "k_acquire", "k_threadfence", "k_atomic_load"}
    assert fence["k_release"]["wbl2_sc1"] == 1
    assert fence["k_release"]["wbl2_sc0_sc1"] == 0
    assert fence["k_atomic_load"]["load_sc1"] == 1
    assert "kernel" not in fence["k_release"]


def test_rocminfo_parser_picks_the_gfx_agent():
    agents = summarize.gpu_agents(summarize.parse_rocminfo(ROCMINFO))
    assert len(agents) == 1
    agent = agents[0]
    assert agent["fields"]["Name"] == "gfx942"
    assert agent["fields"]["Marketing Name"] == "AMD Instinct MI300X"
    assert summarize.first_int(agent["fields"]["Compute Unit"]) == 304
    assert summarize.size_to_bytes(agent["cache"]["L2"]) == 4 << 20
    assert summarize.pool_size_bytes(agent, "GROUP") == 65536
    assert summarize.pool_size_bytes(agent, "GLOBAL") == 192 << 30


def test_kfd_parser_finds_the_gpu_node():
    node = summarize.kfd_gpu_node(summarize.parse_kfd(KFD))
    assert node["simd_count"] == 1216
    assert node["simd_per_cu"] == 4
    assert node["num_xcc"] == 8


def test_df_parser_reads_both_gnu_and_bsd_units():
    gnu = ("Filesystem      Size  Used Avail Use% Mounted on\n"
           "/dev/sda1        13T  1.1T   12T   9% /\n")
    bsd = ("Filesystem      Size   Used  Avail Capacity  Mounted on\n"
           "/dev/disk3s5   926Gi   10Gi  300Gi     4%  /Users/x\n")
    assert summarize.df_avail_gib(gnu) == pytest.approx(12 * 1024)
    assert summarize.df_avail_gib(bsd) == pytest.approx(300)
    assert summarize.df_avail_gib("UNAVAILABLE: exit 127: df -h /\n") is None


def test_a_failed_capture_is_not_a_clean_read(tmp_path):
    out = write_machine(tmp_path, {
        "amd-smi-metric-throttle.txt": "amd-smi: command not found\n"
                                       "UNAVAILABLE: exit 127: amd-smi metric --throttle\n",
    }, with_pmc=False, with_ktrace=False)
    s = summarize.Summary(out)
    assert s.read("amd-smi-metric-throttle")[1] is True
    assert s.read_clean("amd-smi-metric-throttle")[1] is False


def test_partition_parser_is_case_insensitive():
    assert summarize.parse_partition("compute_partition: spx\nmemory: NPS1") == ("SPX", "NPS1")
    assert summarize.parse_partition("CPX / NPS4") == ("CPX", "NPS4")
    assert summarize.parse_partition("") == (None, None)


# ---------------------------------------------------------------------------
# machine 1: every expectation met


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    out = write_machine(tmp_path_factory.mktemp("20260915"), base_files())
    rows, markdown = summarize.summarize(out)
    return rows, markdown


def test_every_checklist_row_is_present(good):
    rows, _ = good
    ids = [row["id"] for row in rows]
    assert ids == (
        [f"A{i}" for i in range(1, 9)] + [f"B{i}" for i in range(1, 12)]
        + [f"C{i}" for i in range(1, 5)] + [f"D{i}" for i in range(1, 6)]
        + [f"E{i}" for i in range(1, 7)] + [f"F{i}" for i in range(1, 7)]
        + [f"G{i}" for i in range(1, 5)] + [f"H{i}" for i in range(1, 8)]
        + [f"I{i}" for i in range(1, 6)] + [f"J{i}" for i in range(1, 7)])
    assert len(ids) == 62


def test_identity_and_topology_pass(good):
    rows, _ = good
    r = results(rows)
    for rid in ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9"):
        assert r[rid] == summarize.PASS, rid
    assert r["A5"] == summarize.INFO
    assert r["A8"] == summarize.INFO
    assert measured(rows)["A8"] == "torch 2.5.1+rocm7.0"
    assert r["B8"] == summarize.INFO
    assert measured(rows)["B8"] == "no L3 row listed"


def test_residency_and_clock_pass(good):
    rows, _ = good
    assert results(rows)["B10"] == summarize.PASS
    assert measured(rows)["B10"] == (
        "register-only 2 block(s) per CU (184 VGPRs), "
        "with the 58,368 B request 1 block(s) per CU, 304 co-resident")
    assert results(rows)["B11"] == summarize.PASS


def test_b10_turns_on_the_line_with_the_dynamic_lds_request(tmp_path):
    """The register-only line alone cannot answer B10, but it is still reported."""
    files = base_files()
    files["occupancy.txt"] = ("occupancy requested_vgprs=182 actual_vgprs=184 "
                              "dynamic_lds=0 blocks_per_cu=2 waves_per_simd=2 "
                              "cus=304 coresident_blocks=608\n"
                              "UNAVAILABLE: exit 1: occupancy --dynamic-lds 58368\n")
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    assert results(rows)["B10"] == summarize.UNAVAIL
    assert measured(rows)["B10"] == "register-only 2 block(s) per CU (184 VGPRs)"


def test_b10_fails_when_the_worker_grid_is_not_co_resident(tmp_path):
    """A machine that refuses the 58,368 B request answers with blocks_per_cu=0
    rather than a non-zero exit, so the row has to catch it as a MISMATCH: the
    megakernel deadlocks on a grid that is not co-resident."""
    files = base_files()
    files["occupancy.txt"] = ("occupancy requested_vgprs=182 actual_vgprs=184 "
                              "dynamic_lds=0 blocks_per_cu=2 waves_per_simd=2 "
                              "cus=304 coresident_blocks=608\n"
                              "occupancy requested_vgprs=182 actual_vgprs=184 "
                              "dynamic_lds=58368 blocks_per_cu=0 waves_per_simd=0 "
                              "cus=304 coresident_blocks=0\n")
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    assert results(rows)["B10"] == summarize.MISMATCH


def test_occupancy_lds_helper_reads_the_request():
    assert summarize.occupancy_lds({"dynamic_lds": "58368"}) == 58368
    assert summarize.occupancy_lds({"dynamic_lds": "0"}) == 0
    # A line without the key is the register-only run. lds_bytes belongs to
    # stream_read and must not be mistaken for an LDS request here.
    assert summarize.occupancy_lds({"requested_vgprs": "182"}) == 0
    assert summarize.occupancy_lds({"lds_bytes": "33792"}) == 0


def test_partition_clocks_and_bandwidth_pass(good):
    rows, _ = good
    r = results(rows)
    assert r["C1"] == summarize.PASS
    assert r["C2"] == summarize.PASS
    assert r["C3"] == summarize.INFO
    assert r["D4"] == summarize.PASS
    assert r["D5"] == summarize.PASS
    assert r["E1"] == summarize.PASS
    assert r["E2"] == summarize.PASS
    assert r["E3"] == summarize.PASS
    assert r["E4"] == summarize.INFO
    assert r["E5"] == summarize.PASS
    assert "knee at N=4" in measured(rows)["E2"]
    assert "step above 32 MiB: yes" in measured(rows)["E4"]
    assert "above 256 MiB: yes" in measured(rows)["E4"]


def test_placement_and_fences_pass(good):
    rows, _ = good
    r = results(rows)
    for rid in ("F1", "F2", "F3", "F4", "F5", "F6", "G1", "G2", "G3", "G4"):
        assert r[rid] == summarize.PASS, rid
    assert measured(rows)["G3"].endswith("agent")


def test_latency_rows_are_info_with_numbers(good):
    rows, _ = good
    r, m = results(rows), measured(rows)
    for rid in ("H1", "H2", "H3", "H4", "H5", "H6", "H7"):
        assert r[rid] == summarize.INFO, rid
    assert "210.4 ns wall" in m["H1"]
    assert "540.2 ns wall" in m["H3"]
    assert "714.3 ns" in m["H4"]
    assert "186.7 ns one way" in m["H7"]


def test_profiler_rows_pass_and_the_arithmetic_is_exact(good):
    rows, _ = good
    r, m = results(rows), measured(rows)
    assert r["I1"] == summarize.PASS
    assert m["I1"] == f"all {len(summarize.PMC_NAMES)} present"
    assert r["I2"] == summarize.PASS
    assert r["I3"] == summarize.PASS
    # 2^23 read requests of 128 B and 2^24 write requests of 64 B are 1 GiB each
    # way; the flat 64 B per read request reports half of that.
    assert "reads 1.0000 GiB (+0.0%)" in m["I3"]
    assert "writes 1.0000 GiB (+0.0%)" in m["I3"]
    assert "flat 64 B reads 0.5000 GiB (-50.0%)" in m["I3"]
    assert "flat 64 B writes 1.0000 GiB (+0.0%)" in m["I3"]
    assert r["I4"] == summarize.PASS
    assert r["I5"] == summarize.INFO
    assert m["I5"] == "16 per XCC x 8 XCC = 128"


def test_host_rows(good):
    rows, _ = good
    r, m = results(rows), measured(rows)
    assert r["J1"] == summarize.INFO
    assert "AMD EPYC 9534 64-Core Processor" in m["J1"]
    assert r["J2"] == summarize.INFO
    assert m["J2"] == "448 GB total"
    assert r["J3"] == summarize.PASS
    assert r["J4"] == summarize.INFO
    assert m["J4"] == "last rate 56.7MB/s"
    assert r["J5"] == summarize.PASS
    # Not a command's output; the operator times it and the row carries that.
    assert r["J6"] == summarize.INFO
    assert m["J6"] == "recorded by hand"


def test_a_good_machine_has_no_mismatch_and_nothing_unavailable(good):
    rows, _ = good
    r = results(rows)
    assert [rid for rid, value in r.items() if value == summarize.MISMATCH] == []
    assert [rid for rid, value in r.items() if value == summarize.UNAVAIL] == []


def test_the_markdown_has_a_header_and_one_line_per_row(good):
    rows, markdown = good
    assert markdown.startswith("# Hardware collection ")
    assert "ROCm 7.0.0, hipcc 7.0.51831" in markdown
    assert "mi300x-vm" in markdown
    assert "| # | Check | Measured | Expected | Result |" in markdown
    body = [line for line in markdown.splitlines() if line.startswith("| ")]
    assert len(body) == len(rows) + 1  # the header row
    assert f"{len(rows)} checklist rows" in markdown


# ---------------------------------------------------------------------------
# machine 2: a CPX partition, a slow E1, a broken rule, a system fence, no PMC


@pytest.fixture(scope="module")
def degraded(tmp_path_factory):
    files = base_files()
    files["partition.txt"] = ("GPU: 0\n    COMPUTE_PARTITION: CPX\n"
                              "    MEMORY_PARTITION: NPS1\n")
    files["e1.txt"] = e1_text(tbps=(3.2001, 3.2100, 3.1950))
    files["xcc-map.txt"] = xcc_map_text(violations_296=7)
    files["fence.txt"] = FENCE_SYSTEM
    files["python-torch.txt"] = (
        'Traceback (most recent call last):\n'
        '  File "<string>", line 1, in <module>\n'
        "ModuleNotFoundError: No module named 'torch'\n"
        "UNAVAILABLE: exit 1: python3 -c import torch\n")
    out = write_machine(tmp_path_factory.mktemp("20260916"), files, with_pmc=False)
    rows, markdown = summarize.summarize(out)
    return rows, markdown


def test_cpx_partition_is_a_mismatch(degraded):
    rows, _ = degraded
    assert results(rows)["C1"] == summarize.MISMATCH
    assert measured(rows)["C1"] == "CPX"
    assert results(rows)["C2"] == summarize.PASS


def test_slow_bandwidth_is_a_mismatch(degraded):
    rows, _ = degraded
    assert results(rows)["E1"] == summarize.MISMATCH
    assert "mean 3.202 TB/s" in measured(rows)["E1"]


def test_rule_violations_are_a_mismatch(degraded):
    rows, _ = degraded
    assert results(rows)["F2"] == summarize.MISMATCH
    assert "21 strict-rule violation(s)" in measured(rows)["F2"]
    assert results(rows)["F1"] == summarize.PASS
    assert results(rows)["F3"] == summarize.PASS


def test_the_measured_offset_four_placement(tmp_path):
    """What the machine actually does: a round robin over all eight XCDs, but
    starting at 4. Still a MISMATCH against the strict rule the design assumes,
    and the offset has to be in the text or the row cannot be acted on."""
    files = base_files()
    files["xcc-map.txt"] = xcc_map_text(offset=4)
    files["xcc-map-296.csv"] = xcc_dump_text(296, offset=4)
    files["xcc-map-8.csv"] = xcc_dump_text(8, offset=4)
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    r, m = results(rows), measured(rows)

    for rid in ("F2", "F3", "F4"):
        assert r[rid] == summarize.MISMATCH, rid
    assert "round robin with offset 4 (xcd == (block + 4) mod 8" in m["F2"]
    assert "888 strict-rule violation(s) over 3 run(s)" in m["F2"]
    assert "blocks 0..7 on XCD 4,5,6,7,0,1,2,3 (grid-8 dump)" in m["F3"]
    # 37 blocks at offset 4 is 5,4,4,4,5,5,5,5, not the 5,5,5,5,5,4,4,4 of the
    # checklist: the counts move with the offset and are still consistent.
    assert "37 per XCD [5, 4, 4, 4, 5, 5, 5, 5]" in m["F4"]
    assert "counts match offset 4" in m["F4"]
    # The placement is still one block per XCD over eight, so these still hold.
    for rid in ("F1", "F5", "F6"):
        assert r[rid] == summarize.PASS, rid


def test_scheduler_placement_falls_back_to_the_offset_without_a_dump(tmp_path):
    files = base_files()
    files["xcc-map.txt"] = xcc_map_text(offset=4)
    files["xcc-map-296.csv"] = xcc_dump_text(296, offset=4)
    del files["xcc-map-8.csv"]
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    assert "blocks 0..7 on XCD 4,5,6,7,0,1,2,3 (derived from the offset)" in \
        measured(rows)["F3"]


def test_offset_helpers():
    assert summarize.round_robin_offset([(0, 4), (1, 5), (7, 3), (8, 4)]) == 4
    assert summarize.round_robin_offset([(b, b % 8) for b in range(16)]) == 0
    assert summarize.round_robin_offset([(0, 1), (1, 5)]) is None
    assert summarize.round_robin_offset([]) is None
    assert summarize.expected_per_xcd(37, 0) == [5, 5, 5, 5, 5, 4, 4, 4]
    assert summarize.expected_per_xcd(37, 4) == [5, 4, 4, 4, 5, 5, 5, 5]
    assert summarize.expected_per_xcd(296, 4) == [37] * 8


def test_counters_are_summed_over_the_copy_kernel_only(tmp_path):
    """The copy program also dispatches fill_kernel and warmup_kernel. Summing
    every dispatch reported three times the known 1 GiB of writes."""
    raw = tmp_path / "raw"
    d = raw / "pmc2" / "pmc_1"
    d.mkdir(parents=True)
    rows = [PMC_HEADER]
    for kernel, value in (("fill_kernel(unsigned int*)", 1 << 25),
                          ("copy_kernel(float*, float const*)", 1 << 24),
                          ("warmup_kernel(unsigned int*)", 1 << 20),
                          ("__amd_rocclr_copyBuffer", 1 << 21)):
        rows.append('1,1,1,1,1,1,1,1,"{k}",256,0,0,8,16,"TCC_EA0_WRREQ_sum",{v}'
                    .format(k=kernel, v=float(value)))
    (d / "x_counter_collection.csv").write_text("\n".join(rows) + "\n")
    sums = summarize.parse_pmc_dir(raw / "pmc2")
    assert sums["TCC_EA0_WRREQ_sum"] == float(1 << 24)


def test_i3_is_unavailable_without_the_bubble_counter(tmp_path):
    out = write_machine(tmp_path, base_files(), with_pmc=False)
    raw = out / "raw"
    for name, pairs in PMC_DIRS.items():
        if name == "pmc4":
            continue
        d = raw / name / "pmc_1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "x_counter_collection.csv").write_text(pmc_csv(pairs))
    rows, _ = summarize.summarize(out)
    assert results(rows)["I3"] == summarize.UNAVAIL
    assert "reads not computable (TCC_BUBBLE_sum not collected)" in measured(rows)["I3"]
    # The write side and the flat form are still reported.
    assert "writes 1.0000 GiB (+0.0%)" in measured(rows)["I3"]


def test_throttle_flag_missing_is_info_not_a_verdict(tmp_path):
    """This amd-smi rejects --throttle outright. The temperature was still read,
    so the row is INFO with the reason rather than a claim of no throttling."""
    files = base_files()
    files["amd-smi-metric-throttle.txt"] = (
        "amdsmi_cli_exceptions.AmdSmiInvalidParameterException: Parameter "
        "'--throttle' is invalid. Run 'amd-smi metric -h' for more info. "
        "Error code: -2\n"
        "UNAVAILABLE: exit 1: amd-smi metric --throttle\n")
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    assert results(rows)["D4"] == summarize.INFO
    assert "--throttle is not a flag in this amd-smi" in measured(rows)["D4"]
    assert measured(rows)["D4"].startswith("45 C, ")


def test_e4_says_which_points_are_bandwidth(good):
    rows, _ = good
    assert "launch-bound" in measured(rows)["E4"]
    assert "512 and 1024 MiB points are bandwidth" in measured(rows)["E4"]


def test_e4_drops_the_caveat_once_passes_normalises_the_sweep(tmp_path):
    """--passes keeps the loads per thread constant across the sweep, so the
    small sizes stop being a launch measurement and the caveat is wrong."""
    files = base_files()
    files["e4.txt"] = "\n".join(
        stream_read_line(size_mib=w, gbps=g, tbps=round(g / 1000.0, 4),
                         passes=1024 // w)
        for w, g in [(4, 11800.0), (16, 11700.0), (32, 11500.0), (64, 5200.0),
                     (128, 5100.0), (256, 5050.0), (512, 3950.0),
                     (1024, 3900.0)]) + "\n"
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    assert results(rows)["E4"] == summarize.INFO
    assert "launch-bound" not in measured(rows)["E4"]
    assert "step above 32 MiB: yes" in measured(rows)["E4"]


def test_wall_clock_rate_off_by_more_than_one_percent_is_a_mismatch(tmp_path):
    """Every in-kernel time number converts s_memrealtime ticks with this rate,
    so a rate that disagrees with the host clock is not a note, it is a fault."""
    files = base_files()
    files["wallclock.txt"] = ("wallclock rate_khz=100000 ticks=10000000 "
                              "host_ns=100000000 ratio=1.0730\n")
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    assert results(rows)["B11"] == summarize.MISMATCH
    assert "ratio 1.0730" in measured(rows)["B11"]


def test_e3_without_e2_is_unavailable_not_a_mismatch(tmp_path):
    files = base_files()
    files["e2.txt"] = "UNAVAILABLE: exit 1: stream_read --occupancy one\n"
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    assert results(rows)["E2"] == summarize.UNAVAIL
    # E3 measured something; it just has no neighbour to be judged against.
    assert results(rows)["E3"] == summarize.UNAVAIL
    assert "nothing to compare against" in measured(rows)["E3"]


def test_e3_knee_must_be_about_half_of_e2(tmp_path):
    files = base_files()
    # E2 knees at 4, so E3 has to reach its plateau by N=2.
    files["e3.txt"] = unroll_sweep([(1, 2000.0), (2, 2500.0), (4, 3700.0),
                                    (8, 3720.0), (16, 3730.0), (32, 3740.0)],
                                   occupancy="grid", grid=608)
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    assert results(rows)["E2"] == summarize.PASS
    assert results(rows)["E3"] == summarize.MISMATCH
    assert "E2 knee 4, so at or below N=2" in measured(rows)["E3"]


def test_acquire_that_also_carries_sc0_is_a_mismatch(tmp_path):
    files = base_files()
    files["fence.txt"] = FENCE_AGENT.replace(
        "fence kernel=k_acquire wbl2_sc1=0 inv_sc1=1 wbl2_sc0_sc1=0 inv_sc0_sc1=0 load_sc1=0",
        "fence kernel=k_acquire wbl2_sc1=0 inv_sc1=1 wbl2_sc0_sc1=0 inv_sc0_sc1=2 load_sc1=0")
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    assert results(rows)["G2"] == summarize.MISMATCH
    assert results(rows)["G1"] == summarize.PASS


def test_atomic_load_needs_both_the_load_and_the_invalidate(tmp_path):
    files = base_files()
    files["fence.txt"] = FENCE_AGENT.replace(
        "fence kernel=k_atomic_load wbl2_sc1=0 inv_sc1=1 wbl2_sc0_sc1=0 inv_sc0_sc1=0 load_sc1=1",
        "fence kernel=k_atomic_load wbl2_sc1=0 inv_sc1=0 wbl2_sc0_sc1=0 "
        "inv_sc0_sc1=0 load_sc1=1 load_sc0_sc1=3")
    out = write_machine(tmp_path, files, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    # The load carries sc1 but nothing invalidates after it, so the counter poll
    # would read its own stale line.
    assert results(rows)["G4"] == summarize.MISMATCH
    # fence_grep.py's newer system-scope counter is shown when it is reported.
    assert "load sc0 sc1 x3" in measured(rows)["G4"]


def test_system_scope_threadfence_is_a_mismatch(degraded):
    # The offline compile settled __threadfence() as agent scope, so a machine
    # whose hipcc lowers it to sc0 sc1 has to be filed, not merely noted.
    rows, _ = degraded
    assert results(rows)["G3"] == summarize.MISMATCH
    assert measured(rows)["G3"].endswith("system")
    assert results(rows)["G1"] == summarize.PASS
    assert results(rows)["G2"] == summarize.PASS


def test_missing_pmc_output_is_unavailable(degraded):
    rows, _ = degraded
    r = results(rows)
    assert r["I2"] == summarize.UNAVAIL
    assert r["I3"] == summarize.UNAVAIL
    assert r["I1"] == summarize.PASS
    assert r["I4"] == summarize.PASS


def test_absent_torch_is_info(degraded):
    rows, _ = degraded
    assert results(rows)["A8"] == summarize.INFO
    assert measured(rows)["A8"] == "absent"


# ---------------------------------------------------------------------------
# machine 3: nothing collected


def test_an_empty_raw_directory_gives_every_row_unavailable(tmp_path):
    out = write_machine(tmp_path, {}, with_pmc=False, with_ktrace=False)
    assert summarize.main(["summarize.py", str(out)]) == 0
    rows, _ = summarize.summarize(out)
    assert len(rows) == 62
    # J6 is timed by hand rather than captured, so it is the one row a collection
    # with nothing in it can still answer.
    assert {row["id"] for row in rows if row["result"] != summarize.UNAVAIL} == {"J6"}
    written = (out / "summary.md").read_text()
    assert "62 checklist rows" in written
    assert "0 PASS, 0 MISMATCH, 1 INFO, 61 UNAVAILABLE" in written
    assert "ROCm unknown, hipcc unknown" in written


def test_a_missing_output_directory_still_writes_a_summary(tmp_path):
    out = tmp_path / "20260101"
    assert summarize.main(["summarize.py", str(out)]) == 0
    assert (out / "summary.md").exists()


def test_unavailable_markers_alone_do_not_count_as_a_value(tmp_path):
    out = write_machine(tmp_path, {
        "rocminfo.txt": "UNAVAILABLE: exit 127: rocminfo\n",
        "e1.txt": "UNAVAILABLE: exit 1: stream_read --occupancy full\n",
    }, with_pmc=False, with_ktrace=False)
    rows, _ = summarize.summarize(out)
    r = results(rows)
    assert r["A1"] == summarize.UNAVAIL
    assert r["E1"] == summarize.UNAVAIL


def test_an_error_message_is_never_read_as_a_negative_result(tmp_path):
    """A missing tool must not become "no L3 row", "no throttle flags" or
    "no gfx942 in the container": those readings need a command that ran."""
    missing = "bash: amd-smi: command not found\nUNAVAILABLE: exit 127: amd-smi\n"
    out = write_machine(tmp_path, {
        "rocminfo.txt": "rocminfo: command not found\nUNAVAILABLE: exit 127: rocminfo\n",
        "amdgpu-version.txt": "cat: /sys/module/amdgpu/version: No such file or directory\n"
                              "UNAVAILABLE: exit 1: cat /sys/module/amdgpu/version\n",
        "rocm-include.txt": "ls: /opt/rocm/include: No such file or directory\n"
                            "ck_tile: absent\nck: absent\n"
                            "UNAVAILABLE: exit 1: rocm_include_report\n",
        "amd-smi-static-driver.txt": missing,
        "amd-smi-set-help.txt": missing,
        "amd-smi-metric-throttle.txt": missing,
        "amd-smi-metric-temp.txt": missing,
        "amd-smi-metric-mem-usage.txt": missing,
        "amd-smi-process.txt": missing,
        "docker-gpu.txt": "docker: command not found\nUNAVAILABLE: exit 127: docker run\n",
    }, with_pmc=False, with_ktrace=False)
    r = results(summarize.summarize(out)[0])
    for rid in ("A2", "A5", "A6", "B8", "C4", "D4", "D5", "J5"):
        assert r[rid] == summarize.UNAVAIL, rid
