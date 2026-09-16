#!/usr/bin/env python3
"""Fill docs/gpu-experiments/01-bringup/02-checklist.md in from the raw captures of env/collect_hw.sh.

    python3 env/hw/summarize.py env/hw/<YYYYMMDD>

Reads <out_dir>/raw/<slug>.txt, one file per command, with the slugs listed at
the top of env/collect_hw.sh, and writes <out_dir>/summary.md: one row per
checklist row A1..J6 with the measured value, the expected value and a result
in PASS, MISMATCH, UNAVAILABLE, INFO. The table is printed to stdout too.

Rules for the result column:
  INFO         a row the checklist holds no expectation for, and a value was read
  UNAVAILABLE  the file is missing, empty, or holds no value this parser can use
  PASS         the row has an expectation and the measurement meets it
  MISMATCH     the row has an expectation and the measurement does not

The exit status is always 0: a collection with holes in it is still the record.
Standard library only, so it runs in the system python3 before the venvs exist.
"""
import csv
import re
import statistics
import sys
from pathlib import Path

PASS = "PASS"
MISMATCH = "MISMATCH"
UNAVAIL = "UNAVAILABLE"
INFO = "INFO"

GIB = 1 << 30
PMC_NAMES = [
    "TCC_EA0_RDREQ_sum",
    "TCC_EA0_RDREQ_32B_sum",
    "TCC_EA0_WRREQ_sum",
    "TCC_EA0_WRREQ_64B_sum",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
    # Reads on MI300 arrive as 128 B requests counted by TCC_BUBBLE, so the
    # read side of the byte arithmetic cannot be computed without it.
    "TCC_BUBBLE_sum",
]
# The copy program dispatches fill_kernel and warmup_kernel besides the 1 GiB
# copy, and the runtime adds its own blit kernels for the readback, so counters
# are summed over this kernel only.
PMC_KERNEL = "copy_kernel"
COMPUTE_PARTITIONS = ("SPX", "DPX", "TPX", "QPX", "CPX")
MEMORY_PARTITIONS = ("NPS1", "NPS2", "NPS4", "NPS8")
UNITS = {"B": 1, "KB": 1 << 10, "KIB": 1 << 10, "MB": 1 << 20, "MIB": 1 << 20,
         "GB": 1 << 30, "GIB": 1 << 30, "TB": 1 << 40, "TIB": 1 << 40}


# ---------------------------------------------------------------------------
# small conversions


def first_int(s):
    """The first integer in a string, ignoring thousands separators."""
    if s is None:
        return None
    m = re.search(r"-?\d+", str(s).replace(",", ""))
    return int(m.group()) if m else None


def first_float(s):
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(s).replace(",", ""))
    return float(m.group()) if m else None


def size_to_bytes(s):
    """"64(0x40) KB" -> 65536. A bare number is taken as bytes."""
    n = first_int(s)
    if n is None:
        return None
    m = re.search(r"\b([KMGT]?i?B)\b", str(s), re.I)
    return n * UNITS.get(m.group(1).upper(), 1) if m else n


def human_gib(n_bytes):
    return None if n_bytes is None else n_bytes / float(GIB)


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def to_int(s):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def fmt(x, digits=2):
    if x is None:
        return "-"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:,.{digits}f}"


def pct(x, digits=1):
    return "-" if x is None else f"{x:+.{digits}f}%"


# ---------------------------------------------------------------------------
# parsers


def parse_kv_lines(text):
    """The probes' stdout contract: a name, then key=value pairs, one measurement
    per line. Returns a list of dicts with the name under "_name"."""
    rows = []
    line_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\s+[A-Za-z0-9_]+=\S*)+)\s*$")
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("UNAVAILABLE:"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        row = {"_name": m.group(1)}
        for token in m.group(2).split():
            key, _, value = token.partition("=")
            row[key] = value
        rows.append(row)
    return rows


def named(rows, name):
    return [r for r in rows if r.get("_name") == name]


def occupancy_lds(row):
    """The dynamic LDS request of one occupancy line, in bytes.

    The key is dynamic_lds, between actual_vgprs and blocks_per_cu. A line
    without it is the register-only run. lds_bytes is a stream_read key and is
    deliberately not consulted here."""
    value = to_int(row.get("dynamic_lds"))
    return value if value is not None else 0


def parse_fence(text):
    """fence_grep.py's lines, keyed by kernel, values as integers."""
    out = {}
    for row in named(parse_kv_lines(text), "fence"):
        kernel = row.get("kernel")
        if not kernel:
            continue
        counts = {k: to_int(v) for k, v in row.items() if not k.startswith("_") and k != "kernel"}
        out[kernel] = counts
    return out


def parse_rocminfo(text):
    """Every agent block of rocminfo as {"fields", "cache", "pools"}."""
    agents, block = [], None
    for line in (text or "").splitlines():
        if re.match(r"^Agent\s+\d+\s*$", line):
            block = []
            agents.append(block)
            continue
        if block is not None:
            block.append(line)
    return [_rocminfo_agent(b) for b in agents]


def _rocminfo_agent(lines):
    fields, cache, pools = {}, {}, []
    section = None
    for line in lines:
        s = line.strip()
        if not s or set(s) == {"*"}:
            continue
        indent = len(line) - len(line.lstrip())
        if section == "Pool Info" and re.match(r"^Pool\s+\d+$", s):
            pools.append({})
            continue
        m = re.match(r"^([^:]+?)\s*:\s*(.*)$", s)
        if not m:
            continue
        key, value = m.group(1).strip(), m.group(2).strip()
        if indent <= 2:
            if value == "" and key in ("Cache Info", "Pool Info", "ISA Info"):
                section = key
            else:
                section = None
                fields[key] = value
        elif section == "Cache Info" and re.match(r"^L\d+$", key):
            cache[key] = value
        elif section == "Pool Info" and pools:
            pools[-1][key] = value
    return {"fields": fields, "cache": cache, "pools": pools}


def gpu_agents(agents):
    return [a for a in agents if a["fields"].get("Name", "").startswith("gfx")]


def pool_size_bytes(agent, segment):
    """The largest pool of a segment, in bytes."""
    sizes = []
    for pool in agent["pools"]:
        if (pool.get("Segment", "").upper().startswith(segment.upper())):
            b = size_to_bytes(pool.get("Size"))
            if b is not None:
                sizes.append(b)
    return max(sizes) if sizes else None


def parse_kfd(text):
    """The combined topology capture: "== <node dir>" then that node's properties."""
    nodes, cur = [], None
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("== "):
            cur = {"_node": s[3:]}
            nodes.append(cur)
            continue
        if cur is None or not s or s.startswith("UNAVAILABLE:"):
            continue
        parts = s.split()
        if len(parts) >= 2:
            cur[parts[0]] = to_int(parts[1]) if to_int(parts[1]) is not None else parts[1]
    return nodes


def kfd_gpu_node(nodes):
    """The GPU node is the one with simd_count > 0 (there is no cu_count property)."""
    for node in nodes:
        if isinstance(node.get("simd_count"), int) and node["simd_count"] > 0:
            return node
    return None


def parse_xcc_dump(path):
    """[(block, xcd)] from an xcc_map --dump CSV of "block,xcd" lines."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return []
    pairs = []
    for line in text.splitlines():
        fields = line.strip().split(",")
        if len(fields) != 2:
            continue
        block, xcd = to_int(fields[0]), to_int(fields[1])
        if block is not None and xcd is not None:
            pairs.append((block, xcd))
    return pairs


def round_robin_offset(pairs, xcds=8):
    """k with xcd == (block + k) mod xcds for every block, or None if none fits.

    The design assumes k == 0. The machine measured k == 4, which is still a
    round robin over all eight XCDs and still puts eight consecutive blocks on
    eight distinct XCDs, but it moves which one."""
    if not pairs:
        return None
    offsets = {(xcd - block) % xcds for block, xcd in pairs}
    return offsets.pop() if len(offsets) == 1 else None


def expected_per_xcd(grid, offset, xcds=8):
    """Blocks per XCD for a grid placed round robin with this offset."""
    counts = [0] * xcds
    for block in range(grid):
        counts[(block + offset) % xcds] += 1
    return counts


def parse_partition(text):
    up = (text or "").upper()
    compute = next((m for m in COMPUTE_PARTITIONS if re.search(r"\b" + m + r"\b", up)), None)
    memory = next((m for m in MEMORY_PARTITIONS if re.search(r"\b" + m + r"\b", up)), None)
    return compute, memory


def bytes_read(counters):
    """Bytes fetched, by rocprofv3's own FETCH_SIZE expression on this machine.

    A read on MI300 is a 128-byte request counted by TCC_BUBBLE; what is left
    of TCC_EA0_RDREQ after the 128 B and the 32 B requests are taken out is
    64 B each. Validated on the VM: TCC_BUBBLE 8,388,608 and TCC_EA0_RDREQ
    8,388,760 over a 1 GiB copy give 1.0000 GiB."""
    bubble = counters.get("TCC_BUBBLE_sum")
    rdreq = counters.get("TCC_EA0_RDREQ_sum")
    rd32 = counters.get("TCC_EA0_RDREQ_32B_sum")
    if bubble is None or rdreq is None or rd32 is None:
        return None
    return 128 * bubble + 64 * (rdreq - bubble - rd32) + 32 * rd32


def bytes_written(counters):
    """Bytes written: 64 B per 64-byte request, 32 B for the rest."""
    wrreq = counters.get("TCC_EA0_WRREQ_sum")
    wr64 = counters.get("TCC_EA0_WRREQ_64B_sum")
    if wrreq is None or wr64 is None:
        return None
    return 64 * wr64 + 32 * (wrreq - wr64)


def parse_rocprof_list(text):
    text = text or ""
    present = {name: (name in text) for name in PMC_NAMES}
    instances, detail = parse_rocprof_instances(text)
    if instances is None:
        # Older listings print one row per instance instead of a Dimensions line.
        count = sum(1 for line in text.splitlines()
                    if re.search(r"TCC_EA0_RDREQ\[\d+\]", line))
        instances, detail = (count, str(count)) if count else (0, "")
    return present, instances, detail


def parse_rocprof_instances(text):
    """(instances, detail) from the Dimensions line of the TCC_EA0_RDREQ block.

    rocprofv3 reports the instance count as a shape rather than as one row per
    instance: "Dimensions : DIMENSION_INSTANCE[0:15] DIMENSION_XCC[0:7]" is 16
    channels on each of 8 XCDs."""
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^Counter_Name\s*:\s*(\S+)\s*$", line)
        if not m or m.group(1) != "TCC_EA0_RDREQ":
            continue
        for follow in lines[i + 1:i + 8]:
            if re.match(r"^Counter_Name\s*:", follow):
                break
            d = re.match(r"^Dimensions\s*:\s*(.+)$", follow)
            if not d:
                continue
            dims = [(name, int(hi) - int(lo) + 1) for name, lo, hi in
                    re.findall(r"DIMENSION_(\w+)\[(\d+):(\d+)\]", d.group(1))]
            if not dims:
                continue
            total = 1
            for _, size in dims:
                total *= size
            sizes = dict(dims)
            if "INSTANCE" in sizes and "XCC" in sizes:
                detail = "{} per XCC x {} XCC = {}".format(
                    sizes["INSTANCE"], sizes["XCC"], total)
            else:
                detail = "{} = {}".format(" x ".join(str(s) for _, s in dims), total)
            return total, detail
    return None, ""


def _read_csv(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def _column(row, *wanted):
    """Find a column by name, ignoring case, spaces and underscores."""
    norm = {re.sub(r"[\s_]", "", k or "").lower(): k for k in row}
    for w in wanted:
        k = norm.get(re.sub(r"[\s_]", "", w).lower())
        if k:
            return k
    return None


def parse_pmc_dir(path, kernel=PMC_KERNEL):
    """Sum the counters of one rocprofv3 -d directory over `kernel`'s dispatches.

    The CSV carries one row per counter per dispatch, and the copy program
    dispatches more than the copy: fill_kernel, warmup_kernel and the runtime's
    own blit kernels are in the same file. Summing all of them gave three times
    the known 1 GiB of writes, so rows are kept only when their Kernel_Name
    names the kernel under test. A file with no Kernel_Name column is summed
    whole, which is the old behaviour and the best that can be done."""
    sums = {}
    path = Path(path)
    if not path.is_dir():
        return sums
    files = sorted(path.rglob("*.csv"))
    counter_files = [f for f in files if "counter_collection" in f.name] or files
    for csv_path in counter_files:
        rows = _read_csv(csv_path)
        if not rows:
            continue
        kernel_key = _column(rows[0], "Kernel_Name")
        if kernel_key and kernel:
            rows = [r for r in rows if kernel in (r.get(kernel_key) or "")]
        name_key = _column(rows[0], "Counter_Name") if rows else None
        value_key = _column(rows[0], "Counter_Value") if rows else None
        if name_key and value_key:
            for row in rows:
                value = to_float(row.get(value_key))
                if value is None:
                    continue
                key = (row.get(name_key) or "").strip()
                sums[key] = sums.get(key, 0.0) + value
            continue
        for key in (rows[0] if rows else {}):
            if key and key.strip().upper().startswith(("TCC_", "SQ_")):
                total = sum(to_float(r.get(key)) or 0.0 for r in rows)
                sums[key.strip()] = sums.get(key.strip(), 0.0) + total
    return sums


def parse_ktrace_dir(path, kernel="copy_kernel"):
    """The rows of a rocprofv3 --kernel-trace CSV whose kernel name matches."""
    out = []
    path = Path(path)
    if not path.is_dir():
        return out
    for csv_path in sorted(path.rglob("*.csv")):
        rows = _read_csv(csv_path)
        if not rows:
            continue
        name_key = _column(rows[0], "Kernel_Name", "Name")
        if not name_key:
            continue
        start_key = _column(rows[0], "Start_Timestamp", "Start")
        end_key = _column(rows[0], "End_Timestamp", "End")
        for row in rows:
            if kernel in (row.get(name_key) or ""):
                out.append({"name": row.get(name_key),
                            "start": to_int(row.get(start_key)) if start_key else None,
                            "end": to_int(row.get(end_key)) if end_key else None})
    return out


def amdsmi_value(text, *keys):
    """The first number printed after one of these amd-smi keys, with its unit."""
    for key in keys:
        m = re.search(r"\b" + re.escape(key) + r"\s*:\s*(-?[\d.]+)\s*(\S*)", text or "")
        if m:
            return to_float(m.group(1)), m.group(2).strip()
    return None, None


def parse_amdsmi_clock(text):
    """CLK and MAX_CLK of the first GFX and the first MEM block of amd-smi metric --clock."""
    out, section = {}, None
    for line in (text or "").splitlines():
        s = line.strip()
        m = re.match(r"^([A-Z][A-Z_0-9]*)\s*:?\s*$", s)
        if m:
            name = m.group(1)
            if "MEM" in name:
                section = "mem"
            elif "GFX" in name:
                section = "gfx"
            continue
        if section:
            m2 = re.match(r"^(MAX_CLK|MIN_CLK|CLK)\s*:\s*(-?[\d.]+)", s)
            if m2:
                out.setdefault(section + "_" + m2.group(1).lower(), to_float(m2.group(2)))
    return out


def parse_babelstream(text):
    """{"Copy": MBytes/sec, ..., "Dot": ...} from the BabelStream result table."""
    out = {}
    for line in (text or "").splitlines():
        m = re.match(r"^(Copy|Mul|Add|Triad|Dot)\s+([\d.]+)", line.strip())
        if m:
            out.setdefault(m.group(1), float(m.group(2)))
    return out


def parse_rbt_peak(text):
    """The largest number printed after the unidirectional bandwidth header."""
    peak, seen = None, False
    for line in (text or "").splitlines():
        if "unidirectional" in line.lower():
            seen = True
            continue
        if seen:
            for token in re.findall(r"\b\d+\.\d+\b", line):
                value = float(token)
                peak = value if peak is None or value > peak else peak
    return peak


def df_avail_gib(text):
    """The Avail column of a df -h capture, in GiB. GNU df prints 12T, BSD df 12Ti."""
    scale = {"K": 1.0 / (1 << 20), "M": 1.0 / 1024, "G": 1.0, "T": 1024.0, "P": 1024.0 * 1024}
    for line in reversed([l for l in (text or "").splitlines() if l.strip()]):
        if line.startswith("UNAVAILABLE:"):
            continue
        fields = line.split()
        if len(fields) >= 4 and re.match(r"^[\d.]+[KMGTP]?i?$", fields[3]):
            value = first_float(fields[3])
            unit = re.sub(r"[\d.i]", "", fields[3])[:1].upper()
            return value * scale.get(unit, 1.0 / (1 << 30))
    return None


def knee_of(rows, x_key="unroll", y_key="gbps", frac=0.9):
    """(smallest x whose y reaches frac of the peak, the peak, the points)."""
    points = []
    for row in rows:
        x, y = to_int(row.get(x_key)), to_float(row.get(y_key))
        if x is not None and y is not None:
            points.append((x, y))
    points.sort()
    if not points:
        return None, None, points
    peak = max(y for _, y in points)
    for x, y in points:
        if y >= frac * peak:
            return x, peak, points
    return None, peak, points


def step_after(points, x, threshold=0.15):
    """Is there a change of more than `threshold` between the point at x and the next?"""
    for i in range(len(points) - 1):
        if points[i][0] == x:
            lo, hi = points[i][1], points[i + 1][1]
            if lo:
                return abs(hi - lo) / lo > threshold
    return None


# ---------------------------------------------------------------------------
# the summary


class Summary:
    """Every checklist row, built from one raw directory."""

    def __init__(self, out_dir):
        self.out = Path(out_dir)
        self.raw = self.out / "raw"
        self.rows = []
        self._load()

    # -- raw access ---------------------------------------------------------

    def read(self, slug, ext="txt"):
        """(text, usable). Usable is False when the file is missing, empty, or
        holds nothing but the UNAVAILABLE markers collect_hw.sh writes."""
        try:
            text = (self.raw / f"{slug}.{ext}").read_text(errors="replace")
        except OSError:
            return "", False
        usable = any(line.strip() and not line.startswith("UNAVAILABLE:")
                     for line in text.splitlines())
        return text, usable

    def read_clean(self, slug, ext="txt"):
        """(text, ok) where ok also demands that the command itself succeeded.

        Rows that read a negative result (no L3 row, no throttle flag, no
        process, no gfx942 agent in the container) must not read one out of a
        command's error message, so they ask for a clean capture."""
        text, usable = self.read(slug, ext)
        failed = any(line.startswith("UNAVAILABLE:") for line in text.splitlines())
        return text, (usable and not failed)

    def _load(self):
        self.rocminfo_text, self.rocminfo_ok = self.read("rocminfo")
        self.agents = gpu_agents(parse_rocminfo(self.rocminfo_text))
        self.agent = self.agents[0] if self.agents else None
        self.fields = self.agent["fields"] if self.agent else {}
        self.cache = self.agent["cache"] if self.agent else {}

        self.kfd_text, self.kfd_ok = self.read("kfd-topology")
        self.kfd = kfd_gpu_node(parse_kfd(self.kfd_text)) or {}

        self.occ_text, self.occ_ok = self.read("occupancy")
        self.clock_text, self.clock_ok = self.read("wallclock")

        self.part_text, self.part_ok = self.read("partition")
        self.part_cmd_text, self.part_cmd_ok = self.read("partition-cmd")

        self.e1_text, self.e1_ok = self.read("e1")
        self.e2_text, self.e2_ok = self.read("e2")
        self.e3_text, self.e3_ok = self.read("e3")
        self.e4_text, self.e4_ok = self.read("e4")

        self.xcc_text, self.xcc_ok = self.read("xcc-map")
        self.xcc = named(parse_kv_lines(self.xcc_text), "xcc_map")
        # The per-block dumps carry the placement itself; the lines carry only
        # counts. The offset is read off the 296-block dump and then explains
        # the counts of every other grid.
        self.xcc_dump_296 = parse_xcc_dump(self.raw / "xcc-map-296.csv")
        self.xcc_dump_8 = parse_xcc_dump(self.raw / "xcc-map-8.csv")
        self.xcc_offset = round_robin_offset(self.xcc_dump_296)

        self.fence_text, self.fence_ok = self.read("fence")
        self.fence = parse_fence(self.fence_text)

        self.chase_text, self.chase_ok = self.read("chase")
        self.chase = named(parse_kv_lines(self.chase_text), "chase")

        # pmc4 is read last on purpose: it carries TCC_EA0_RDREQ_sum measured in
        # the same run as TCC_BUBBLE_sum, and the read formula subtracts one
        # from the other, so the pair has to come from one measurement.
        self.pmc = {}
        for k in (1, 2, 3, 4):
            self.pmc.update(parse_pmc_dir(self.raw / f"pmc{k}"))
        self.ktrace = parse_ktrace_dir(self.raw / "ktrace")

    # -- row helpers --------------------------------------------------------

    def add(self, rid, check, measured, expected, result):
        self.rows.append({"id": rid, "check": check,
                          "measured": measured if measured else "-",
                          "expected": expected, "result": result})

    def add_info(self, rid, check, measured, expected):
        self.add(rid, check, measured, expected, INFO if measured else UNAVAIL)

    def add_check(self, rid, check, measured, expected, ok):
        """PASS or MISMATCH when there is a measurement, UNAVAILABLE when there is not."""
        if not measured or ok is None:
            self.add(rid, check, measured, expected, UNAVAIL)
        else:
            self.add(rid, check, measured, expected, PASS if ok else MISMATCH)

    def xcc_runs(self, grid, concurrent=None):
        out = []
        for row in self.xcc:
            if to_int(row.get("grid")) != grid:
                continue
            if concurrent is not None and to_int(row.get("concurrent")) != concurrent:
                continue
            out.append(row)
        return out

    def chase_rows(self, mode, **match):
        out = []
        for row in self.chase:
            if row.get("mode") != mode:
                continue
            if all(row.get(k) == v for k, v in match.items()):
                out.append(row)
        return out

    # -- groups -------------------------------------------------------------

    def group_a(self):
        name = self.fields.get("Name")
        marketing = self.fields.get("Marketing Name", "")
        measured = f"{name}, {marketing}".strip(", ") if name else ""
        ok = bool(name and name.startswith("gfx942") and "MI300X" in marketing.upper())
        self.add_check("A1", "GPU name and target", measured, "gfx942, MI300X", ok)

        bus_text = self.read("amd-smi-static-bus")[0] + self.read("amd-smi-list")[0]
        bdf = re.search(r"\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d\b", bus_text)
        n_gpu = len(self.agents)
        measured, ok = "", None
        if self.rocminfo_ok and n_gpu:
            measured = f"{n_gpu} GPU agent(s), device 0 BDF {bdf.group() if bdf else 'not read'}"
            ok = n_gpu in (1, 2)
        self.add_check("A2", "GPUs and bus ids", measured,
                       "1 or 2; BDF of device 0 recorded", ok)

        rocm_text, rocm_ok = self.read("rocm-version")
        version = None
        if rocm_ok:
            m = re.search(r"\b(\d+)\.(\d+)(?:\.(\d+))?", rocm_text)
            version = m.group(0) if m else None
        if version is None:
            smi_text, smi_ok = self.read("amd-smi-version")
            if smi_ok:
                m = re.search(r"ROCM[_ ]?VERSION\s*:?\s*v?(\d+\.\d+[\d.]*)", smi_text, re.I)
                version = m.group(1) if m else None
        self.rocm_version = version
        major = first_int(version.split(".")[0]) if version else None
        self.add_check("A3", "ROCm version", version or "", "7.0 or newer",
                       major is not None and major >= 7)

        hipcc_text, hipcc_ok = self.read("hipcc-version")
        m = re.search(r"HIP version:\s*(\d+)\.(\d+)\.(\d+)", hipcc_text) if hipcc_ok else None
        if not m and hipcc_ok:
            m = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", hipcc_text)
        self.hipcc_version = ".".join(m.groups()) if m else None
        self.add_check("A4", "hipcc version", self.hipcc_version or "",
                       "same major as the offline 7.0.51831",
                       m is not None and m.group(1) == "7")

        drv_text, drv_ok = self.read_clean("amdgpu-version")
        drv = drv_text.strip().splitlines()[0].strip() if drv_ok else ""
        static_drv, static_drv_ok = self.read_clean("amd-smi-static-driver")
        m = re.search(r"\bVERSION\s*:\s*(\S+)", static_drv) if static_drv_ok else None
        if m:
            drv = f"{drv} (amd-smi {m.group(1)})" if drv else f"amd-smi {m.group(1)}"
        self.add_info("A5", "Driver version", drv, "INFO")

        inc_text, inc_ok = self.read_clean("rocm-include")
        parts = []
        for name_ in ("ck_tile", "ck") if inc_ok else ():
            m = re.search(rf"^{name_}: (\w+)$", inc_text, re.M)
            if m:
                parts.append(f"{name_} {m.group(1)}")
        self.add_info("A6", "CK headers in ROCm", ", ".join(parts),
                      "INFO: absent or a version")

        os_text, os_ok = self.read("os-release")
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', os_text, re.M) if os_ok else None
        uname_text, uname_ok = self.read("uname")
        kernel = uname_text.split()[2] if uname_ok and len(uname_text.split()) > 2 else None
        docker_text, docker_ok = self.read("docker-version")
        docker = re.search(r"Docker version\s+(\S+?),", docker_text) if docker_ok else None
        py_text, py_ok = self.read("python-version")
        python = re.search(r"Python\s+(\S+)", py_text) if py_ok else None
        parts = [p for p in (m.group(1) if m else None, kernel,
                             f"docker {docker.group(1)}" if docker else None,
                             f"python {python.group(1)}" if python else None) if p]
        self.add_info("A7", "OS, kernel, Docker, Python", ", ".join(parts),
                      "INFO; Ubuntu with Docker")

        torch_text, torch_ok = self.read("python-torch")
        if not (self.raw / "python-torch.txt").exists():
            torch_measured = ""
        elif torch_ok and re.search(r"^\d+\.\d+", torch_text.strip().splitlines()[0] if
                                    torch_text.strip() else ""):
            torch_measured = f"torch {torch_text.strip().splitlines()[0].strip()}"
        else:
            torch_measured = "absent"
        self.add_info("A8", "System torch", torch_measured, "INFO; expected absent")

    def group_b(self):
        cu = first_int(self.fields.get("Compute Unit"))
        kfd_cu = None
        if self.kfd.get("simd_count") and self.kfd.get("simd_per_cu"):
            kfd_cu = self.kfd["simd_count"] // self.kfd["simd_per_cu"]
        parts = [f"rocminfo {cu}" if cu is not None else None,
                 f"kfd {self.kfd.get('simd_count')}/{self.kfd.get('simd_per_cu')} = {kfd_cu}"
                 if kfd_cu is not None else None]
        measured = ", ".join(p for p in parts if p)
        ok = cu == 304 and (kfd_cu is None or kfd_cu == 304)
        self.add_check("B1", "Compute units", measured, "304 (kfd 1216 / 4)", ok)

        num_xcc = self.kfd.get("num_xcc")
        distinct = None
        runs_304 = self.xcc_runs(304)
        if runs_304:
            distinct = to_int(runs_304[0].get("distinct"))
        value = num_xcc if isinstance(num_xcc, int) else distinct
        source = "kfd num_xcc" if isinstance(num_xcc, int) else "distinct ids at grid 304"
        self.add_check("B2", "XCD count", f"{value} ({source})" if value is not None else "",
                       "8", value == 8)

        simds = first_int(self.fields.get("SIMDs per CU"))
        wave = first_int(self.fields.get("Wavefront Size"))
        parts = []
        if simds is not None:
            parts.append(f"{simds} SIMDs per CU")
        if wave is not None:
            parts.append(f"wavefront {wave}")
        if self.kfd.get("simd_per_cu") is not None:
            parts.append(f"kfd {self.kfd.get('simd_per_cu')}/{self.kfd.get('wave_front_size')}")
        self.add_check("B3", "SIMDs per CU, wavefront", ", ".join(parts),
                       "4 SIMDs per CU, wavefront 64", simds == 4 and wave == 64)

        waves_cu = first_int(self.fields.get("Max Waves Per CU"))
        waves_simd = self.kfd.get("max_waves_per_simd")
        parts = []
        if waves_cu is not None:
            parts.append(f"{waves_cu} per CU")
        if waves_simd is not None:
            parts.append(f"kfd {waves_simd} per SIMD")
        self.add_check("B4", "Max waves per CU", ", ".join(parts),
                       "32 per CU, 8 per SIMD (gk)",
                       waves_cu == 32 and (waves_simd is None or waves_simd == 8))

        group_bytes = pool_size_bytes(self.agent, "GROUP") if self.agent else None
        lds_kb = self.kfd.get("lds_size_in_kb")
        parts = []
        if group_bytes is not None:
            parts.append(f"GROUP segment {group_bytes // 1024} KB")
        if lds_kb is not None:
            parts.append(f"kfd {lds_kb} KB")
        self.add_check("B5", "LDS per workgroup", ", ".join(parts), "64 KiB",
                       group_bytes == 65536 or (group_bytes is None and lds_kb == 64))

        wg_max = first_int(self.fields.get("Workgroup Max Size"))
        self.add_check("B6", "Workgroup max size",
                       f"{wg_max} work-items" if wg_max is not None else "",
                       "1024 work-items (gk)", wg_max == 1024)

        l2 = self.cache.get("L2")
        l2_bytes = size_to_bytes(l2) if l2 else None
        self.add_check("B7", "L2 size", l2 or "", "4 MB per XCD",
                       l2_bytes == 4 * (1 << 20))

        l3 = self.cache.get("L3")
        measured = l3 if l3 else ("no L3 row listed" if self.agent else "")
        self.add_info("B8", "Last-level cache", measured,
                      "INFO; 256 MB is the secondary figure")

        global_bytes = pool_size_bytes(self.agent, "GLOBAL") if self.agent else None
        vram_text, vram_ok = self.read("amd-smi-static-vram")
        vram_value, vram_unit = amdsmi_value(vram_text, "SIZE", "TOTAL_VRAM")
        vram_gib = None
        if vram_value is not None:
            vram_gib = vram_value * UNITS.get((vram_unit or "MB").upper(), 1 << 20) / float(GIB)
        gib = human_gib(global_bytes)
        if gib is None:
            gib = vram_gib
        parts = []
        if global_bytes is not None:
            parts.append(f"rocminfo GLOBAL {human_gib(global_bytes):.1f} GiB")
        if vram_gib is not None:
            parts.append(f"amd-smi {vram_gib:.1f} GiB")
        self.add_check("B9", "HBM size", ", ".join(parts), "192 GB (about 196,000 MiB)",
                       gib is not None and 190.0 <= gib <= 200.0)

        # Two lines in one file, told apart by their dynamic LDS request: the
        # worker kernel is launched with 58,368 B (runtime_header.h) and is
        # LDS-limited by it, so that line is the one the row turns on. The
        # register-only line is the compiler's ceiling and is reported beside it.
        occ_rows = named(parse_kv_lines(self.occ_text), "occupancy")
        reg_row = next((r for r in occ_rows if occupancy_lds(r) == 0), None)
        lds_row = next((r for r in occ_rows if occupancy_lds(r) == 58368), None)
        parts, ok = [], None
        if reg_row:
            parts.append("register-only {} block(s) per CU ({} VGPRs)".format(
                reg_row.get("blocks_per_cu"), reg_row.get("actual_vgprs")))
        if lds_row:
            parts.append("with the 58,368 B request {} block(s) per CU, "
                         "{} co-resident".format(lds_row.get("blocks_per_cu"),
                                                 lds_row.get("coresident_blocks")))
            blocks = to_int(lds_row.get("blocks_per_cu"))
            coresident = to_int(lds_row.get("coresident_blocks"))
            ok = (blocks is not None and blocks >= 1 and
                  coresident is not None and coresident >= 304)
        self.add_check("B10", "Residency of the worker kernel", ", ".join(parts),
                       "1 block per CU with the 58,368 B dynamic LDS request; "
                       "2 register-limited", ok)

        # Not an INFO row in practice: every in-kernel time number in the
        # project converts s_memrealtime ticks with this rate, so a rate that
        # disagrees with the host clock by more than 1% invalidates group H,
        # the runtime's event timing and the t_b estimate of MAJ-5.
        clock_rows = named(parse_kv_lines(self.clock_text), "wallclock")
        row = clock_rows[0] if clock_rows else None
        measured, ok = "", None
        if row:
            ratio = to_float(row.get("ratio"))
            measured = "rate {} kHz, ratio {}".format(row.get("rate_khz"), row.get("ratio"))
            ok = ratio is not None and abs(ratio - 1.0) <= 0.01
        self.add_check("B11", "Wall-clock rate", measured,
                       "the attribute and the host clock agree within 1%", ok)

    def group_c(self):
        compute, memory = parse_partition(self.part_text if self.part_ok else "")
        self.add_check("C1", "Compute partition", compute or "", "SPX", compute == "SPX")
        self.add_check("C2", "Memory partition", memory or "", "NPS1", memory == "NPS1")
        cmd = self.part_cmd_text.strip().splitlines()[0].strip() if self.part_cmd_ok else ""
        self.add_info("C3", "Query syntax that works", cmd, "INFO")
        help_text, help_ok = self.read_clean("amd-smi-set-help")
        if not help_ok:
            measured = ""
        elif "--compute-partition" in help_text:
            measured = "amd-smi set offers --compute-partition (not executed)"
        else:
            measured = "amd-smi set does not offer --compute-partition"
        self.add_info("C4", "Guest may set the partition", measured,
                      "UNAVAILABLE acceptable if C1 and C2 pass")

    def group_d(self):
        clock_text, clock_ok = self.read("amd-smi-metric-clock")
        clocks = parse_amdsmi_clock(clock_text) if clock_ok else {}
        mem = []
        if "mem_clk" in clocks:
            mem.append(f"current {fmt(clocks['mem_clk'], 0)} MHz")
        if "mem_max_clk" in clocks:
            mem.append(f"max {fmt(clocks['mem_max_clk'], 0)} MHz")
        self.add_info("D1", "Memory clock", ", ".join(mem),
                      "INFO; the rated HBM3 clock behind the 5.3 TB/s peak")

        gfx = []
        if "gfx_clk" in clocks:
            gfx.append(f"current {fmt(clocks['gfx_clk'], 0)} MHz")
        if "gfx_max_clk" in clocks:
            gfx.append(f"max {fmt(clocks['gfx_max_clk'], 0)} MHz")
        max_clock_text, max_clock_ok = self.read("rocminfo-max-clock")
        rocminfo_clock = first_int(max_clock_text) if max_clock_ok else None
        if rocminfo_clock is not None:
            gfx.append(f"rocminfo {rocminfo_clock} MHz")
        if self.kfd.get("max_engine_clk_fcompute") is not None:
            gfx.append(f"kfd {self.kfd['max_engine_clk_fcompute']} kHz")
        self.add_info("D2", "Engine clock", ", ".join(gfx), "INFO; about 2,100 MHz max (gk)")

        power_text, power_ok = self.read("amd-smi-metric-power")
        draw, draw_unit = amdsmi_value(power_text, "SOCKET_POWER", "CURRENT_SOCKET_POWER",
                                       "AVERAGE_SOCKET_POWER") if power_ok else (None, None)
        limit_text, limit_ok = self.read("amd-smi-static-limit")
        cap, cap_unit = amdsmi_value(limit_text, "MAX_POWER", "POWER_CAP",
                                     "SOCKET_POWER") if limit_ok else (None, None)
        parts = []
        if draw is not None:
            parts.append(f"draw {fmt(draw, 0)} {draw_unit or 'W'}")
        if cap is not None:
            parts.append(f"cap {fmt(cap, 0)} {cap_unit or 'W'}")
        self.add_info("D3", "Power cap and draw", ", ".join(parts),
                      "INFO; cap at the 750 W class (gk)")

        temp_text, temp_ok = self.read_clean("amd-smi-metric-temp")
        temp, temp_unit = amdsmi_value(temp_text, "HOTSPOT", "EDGE", "JUNCTION") if temp_ok \
            else (None, None)
        throttle_text, throttle_ok = self.read_clean("amd-smi-metric-throttle")
        flagged = None
        if throttle_ok:
            flagged = bool(re.search(r":\s*(TRUE|ACTIVE|THROTTLED)\b", throttle_text, re.I))
        # Not every amd-smi has the flag; this one rejects it outright, which is
        # a gap in the reading rather than a throttled machine.
        raw_throttle = self.read("amd-smi-metric-throttle")[0]
        unsupported = not throttle_ok and bool(re.search(
            r"invalid|unrecognized|not a valid|unknown option", raw_throttle, re.I))
        parts = []
        if temp is not None:
            parts.append(f"{fmt(temp, 0)} {temp_unit or 'C'}")
        if flagged is not None:
            parts.append("throttle flags set" if flagged else "no throttle flags")
        elif unsupported:
            parts.append("throttle status not readable: "
                         "--throttle is not a flag in this amd-smi")
        if flagged is not None:
            self.add_check("D4", "Temperature and throttle", ", ".join(parts),
                           "no throttle flags", flagged is False)
        elif temp is not None and unsupported:
            self.add("D4", "Temperature and throttle", ", ".join(parts),
                     "no throttle flags", INFO)
        else:
            self.add_check("D4", "Temperature and throttle", ", ".join(parts),
                           "no throttle flags", None)

        mem_text, mem_ok = self.read_clean("amd-smi-metric-mem-usage")
        used, used_unit = amdsmi_value(mem_text, "USED_VRAM", "VRAM_USED") if mem_ok \
            else (None, None)
        used_gib = None
        if used is not None:
            used_gib = used * UNITS.get((used_unit or "MB").upper(), 1 << 20) / float(GIB)
        proc_text, proc_ok = self.read_clean("amd-smi-process")
        n_proc = None
        if proc_ok:
            if re.search(r"no running process", proc_text, re.I):
                n_proc = 0
            else:
                n_proc = len(re.findall(r"^\s*PID\s*:", proc_text, re.M))
        parts = []
        if used_gib is not None:
            parts.append(f"{used_gib:.2f} GiB used")
        if n_proc is not None:
            parts.append(f"{n_proc} process(es)")
        ok = None
        if used_gib is not None and n_proc is not None:
            ok = used_gib < 1.0 and n_proc == 0
        self.add_check("D5", "Memory in use, processes", ", ".join(parts),
                       "near 0, no other process", ok)

    def group_e(self):
        e1 = named(parse_kv_lines(self.e1_text), "stream_read")
        tbps = [to_float(r.get("tbps")) for r in e1 if to_float(r.get("tbps")) is not None]
        gbps = [to_float(r.get("gbps")) for r in e1 if to_float(r.get("gbps")) is not None]
        self.e1_gbps = statistics.fmean(gbps) if gbps else None
        measured, ok = "", None
        if tbps:
            mean = statistics.fmean(tbps)
            spread = (max(tbps) - min(tbps)) / mean if mean else None
            measured = "{} runs, mean {:.3f} TB/s, spread {:.2f}%".format(
                len(tbps), mean, (spread or 0) * 100)
            ok = 3.66 <= mean <= 4.3 and spread is not None and spread < 0.03
        self.add_check("E1", "Read bandwidth, full, 1 GiB, N=8", measured,
                       "3.66 to 4.3 TB/s; spread under 3%", ok)

        e2 = named(parse_kv_lines(self.e2_text), "stream_read")
        knee2, peak2, points2 = knee_of(e2)
        measured, ok = "", None
        if points2:
            measured = "knee at N={}, plateau {:.0f} GB/s ({})".format(
                knee2, peak2, ", ".join(f"N{int(x)}:{y:.0f}" for x, y in points2))
            ok = (knee2 is not None and knee2 <= 4 and peak2 is not None and
                  (self.e1_gbps is None or peak2 >= 0.9 * self.e1_gbps))
        self.e2_knee = knee2
        self.add_check("E2", "Bandwidth at one wave/SIMD vs N", measured,
                       "knee at N about 4, plateau within 10% of E1", ok)

        # "About half the N of E2", floored at 1 since the knee cannot go below
        # one load in flight. Without E2 there is nothing to halve, so the row
        # is UNAVAILABLE rather than a MISMATCH earned by a missing neighbour.
        e3 = named(parse_kv_lines(self.e3_text), "stream_read")
        knee3, peak3, points3 = knee_of(e3)
        measured, ok = "", None
        if points3:
            target = max(1, knee2 // 2) if knee2 is not None else None
            measured = "knee at N={}, plateau {:.0f} GB/s".format(knee3, peak3)
            if target is not None:
                measured += f" (E2 knee {knee2}, so at or below N={target})"
            else:
                measured += " (E2 knee not measured, nothing to compare against)"
            if knee3 is not None and target is not None:
                ok = knee3 <= target
        self.add_check("E3", "Same at two waves/SIMD", measured,
                       "plateau at about half the N of E2", ok)

        e4 = named(parse_kv_lines(self.e4_text), "stream_read")
        _, _, points4 = knee_of(e4, x_key="size_mib")
        measured = ""
        if points4:
            step32 = step_after(points4, 32)
            step256 = step_after(points4, 256)
            measured = "{}; step above 32 MiB: {}; above 256 MiB: {}".format(
                ", ".join(f"{int(x)}MiB:{y:.0f}" for x, y in points4),
                {True: "yes", False: "no", None: "not measured"}[step32],
                {True: "yes", False: "no", None: "not measured"}[step256])
            # Without --passes every point reads its buffer once, so the small
            # sizes finish in about one launch and measure the launch, not the
            # memory system. With it the loads per thread are constant and the
            # whole sweep is comparable, so the caveat is dropped.
            if not any(r.get("passes") for r in e4):
                measured += ("; sizes under 256 MiB are launch-bound in this run "
                             "(no --passes), so only the 512 and 1024 MiB points "
                             "are bandwidth")
        self.add_info("E4", "Bandwidth vs working set", measured,
                      "step above 32 MiB (L2) and above 256 MiB (Infinity Cache)")

        bs_text, bs_ok = self.read("babelstream")
        bs = parse_babelstream(bs_text) if bs_ok else {}
        measured, ok = "", None
        if bs:
            measured = ", ".join(f"{k} {v:,.0f} MB/s" for k, v in bs.items())
            if "Dot" in bs and "Copy" in bs:
                ok = bs["Dot"] >= 3660781 and bs["Copy"] >= 4177285
        self.add_check("E5", "BabelStream", measured,
                       "Dot >= 3,660,781 MB/s, Copy >= 4,177,285 MB/s", ok)

        rbt_text, rbt_ok = self.read("rocm-bandwidth-test")
        peak = parse_rbt_peak(rbt_text) if rbt_ok else None
        measured = f"peak unidirectional {peak:,.1f} GB/s" if peak is not None else (
            "captured" if rbt_ok else "")
        self.add_info("E6", "Host-to-device bandwidth", measured, "INFO")

    def group_f(self):
        runs_304 = self.xcc_runs(304)
        measured, ok = "", None
        if runs_304:
            distinct = to_int(runs_304[0].get("distinct"))
            in_range = to_int(runs_304[0].get("ids_in_range"))
            measured = f"{distinct} distinct ids, in range: {'yes' if in_range else 'no'}"
            ok = distinct == 8 and in_range == 1
        self.add_check("F1", "XCC_ID readable and in range", measured,
                       "ids in 0..7, all eight present", ok)

        offset = self.xcc_offset
        rule = self._rule_text()

        runs_296 = self.xcc_runs(296)
        measured, ok = "", None
        if runs_296:
            violations = sum(to_int(r.get("rule_mod8_violations")) or 0 for r in runs_296)
            per_xcd = self._per_xcd(runs_296[0])
            parts = [f"per XCD {per_xcd}"]
            if rule:
                parts.append(rule)
            parts.append(f"{violations} strict-rule violation(s) over {len(runs_296)} run(s)")
            measured = ", ".join(parts)
            ok = (violations == 0 and per_xcd == expected_per_xcd(296, 0)
                  and offset in (0, None))
        self.add_check("F2", "Rule at the worker grid (296)", measured,
                       "xcd == block mod 8 for every block, 37 per XCD", ok)

        alone = self.xcc_runs(8, concurrent=0)
        concurrent = self.xcc_runs(8, concurrent=1)
        measured, ok = "", None
        if alone or concurrent:
            v_alone = sum(to_int(r.get("rule_mod8_violations")) or 0 for r in alone)
            v_conc = sum(to_int(r.get("rule_mod8_violations")) or 0 for r in concurrent)
            parts = ["alone {} violation(s) over {} run(s), concurrent {} over {}".format(
                v_alone, len(alone), v_conc, len(concurrent))]
            placement = self._scheduler_placement()
            if placement:
                parts.append(placement)
            measured = ", ".join(parts)
            ok = (bool(alone) and bool(concurrent) and v_alone == 0 and v_conc == 0
                  and offset in (0, None))
        self.add_check("F3", "Scheduler grid (8)", measured,
                       "block k on XCD k, alone and concurrent", ok)

        parts, ok = [], None
        violations_total, grids_seen = 0, 0
        per_xcd_37, consistent = None, True
        for grid in (608, 1000, 37):
            runs = self.xcc_runs(grid)
            if not runs:
                continue
            grids_seen += 1
            v = sum(to_int(r.get("rule_mod8_violations")) or 0 for r in runs)
            violations_total += v
            parts.append(f"grid {grid}: {v} violation(s)")
            counts = self._per_xcd(runs[0])
            if offset is not None and counts != expected_per_xcd(grid, offset):
                consistent = False
            if grid == 37:
                per_xcd_37 = counts
                parts.append(f"37 per XCD {counts}")
        if grids_seen:
            if offset is not None:
                parts.append("counts {} offset {}".format(
                    "match" if consistent else "do not match", offset))
            ok = (violations_total == 0 and grids_seen == 3 and
                  per_xcd_37 == expected_per_xcd(37, 0) and offset in (0, None))
        self.add_check("F4", "Rule at other grids", ", ".join(parts),
                       "same rule at 608, 1000, 37; at 37: 5,5,5,5,5,4,4,4", ok)

        measured, ok = "", None
        if self.xcc:
            stable = [to_int(r.get("stable_vs_run0")) for r in self.xcc]
            unstable = sum(1 for s in stable if s != 1)
            measured = f"{len(self.xcc) - unstable} of {len(self.xcc)} runs identical to run 0"
            ok = unstable == 0
        self.add_check("F5", "Stability across launches", measured,
                       "identical maps across three launches", ok)

        measured, ok = "", None
        if runs_304:
            per_xcd = self._per_xcd(runs_304[0])
            measured = f"per XCD {per_xcd}"
            ok = per_xcd == [38] * 8
        self.add_check("F6", "Blocks per XCD at 304", measured, "38 per XCD at 304", ok)

    def _rule_text(self):
        """How the measured placement reads, once the offset is known."""
        if self.xcc_offset is None:
            return ""
        if self.xcc_offset == 0:
            return "round robin, xcd == block mod 8"
        return ("round robin with offset {k} (xcd == (block + {k}) mod 8, "
                "from the 296-block dump)".format(k=self.xcc_offset))

    def _scheduler_placement(self):
        """Which XCD each of the eight scheduler blocks landed on."""
        pairs = [(b, x) for b, x in self.xcc_dump_8 if b < 8]
        source = "grid-8 dump"
        if not pairs and self.xcc_offset is not None:
            pairs = [(b, (b + self.xcc_offset) % 8) for b in range(8)]
            source = "derived from the offset"
        if not pairs:
            return ""
        pairs.sort()
        return "blocks 0..7 on XCD {} ({})".format(
            ",".join(str(x) for _, x in pairs), source)

    @staticmethod
    def _per_xcd(row):
        value = row.get("per_xcd")
        if not value:
            return None
        counts = [to_int(v) for v in value.split(",")]
        return counts if all(c is not None for c in counts) else None

    def group_g(self):
        release = self.fence.get("k_release", {})
        measured, ok = "", None
        if release:
            measured = "buffer_wbl2 sc1 x{}, sc0 sc1 x{}".format(
                release.get("wbl2_sc1"), release.get("wbl2_sc0_sc1"))
            ok = (release.get("wbl2_sc1") or 0) >= 1 and release.get("wbl2_sc0_sc1") == 0
        self.add_check("G1", "Agent-scope release", measured,
                       "buffer_wbl2 sc1, no sc0 sc1 in that kernel", ok)

        # Symmetric with G1: an acquire that also carries sc0 is a system-scope
        # invalidate, which is not what the design relies on.
        acquire = self.fence.get("k_acquire", {})
        measured, ok = "", None
        if acquire:
            measured = "buffer_inv sc1 x{}, sc0 sc1 x{}".format(
                acquire.get("inv_sc1"), acquire.get("inv_sc0_sc1"))
            ok = (acquire.get("inv_sc1") or 0) >= 1 and acquire.get("inv_sc0_sc1") == 0
        self.add_check("G2", "Agent-scope acquire", measured,
                       "buffer_inv sc1, no sc0 sc1 in that kernel", ok)

        # The offline compile on hipcc 7.0.51831 settled this as agent scope, so
        # the row is now graded: the machine's hipcc has to agree.
        tf = self.fence.get("k_threadfence", {})
        measured, ok = "", None
        if tf:
            system = (tf.get("wbl2_sc0_sc1") or 0) + (tf.get("inv_sc0_sc1") or 0)
            verdict = "system" if system > 0 else "agent"
            measured = ("wbl2 sc1 x{}, inv sc1 x{}, wbl2 sc0 sc1 x{}, inv sc0 sc1 x{}: {}"
                        .format(tf.get("wbl2_sc1"), tf.get("inv_sc1"),
                                tf.get("wbl2_sc0_sc1"), tf.get("inv_sc0_sc1"), verdict))
            ok = ((tf.get("wbl2_sc1") or 0) >= 1 and (tf.get("inv_sc1") or 0) >= 1
                  and system == 0)
        self.add_check("G3", "__threadfence()", measured,
                       "agent scope: buffer_wbl2 sc1 and buffer_inv sc1, no sc0 sc1", ok)

        # The checklist wants both halves: sc1 on the load and the invalidate
        # after it. Either one alone is not the counter poll the runtime needs,
        # so this is an AND. load_sc0_sc1, when fence_grep.py reports it, is the
        # system-scope spelling and is shown but does not satisfy the row.
        atomic = self.fence.get("k_atomic_load", {})
        measured, ok = "", None
        if atomic:
            measured = "load sc1 x{}, buffer_inv sc1 x{}".format(
                atomic.get("load_sc1"), atomic.get("inv_sc1"))
            if atomic.get("load_sc0_sc1") is not None:
                measured += ", load sc0 sc1 x{}".format(atomic.get("load_sc0_sc1"))
            ok = (atomic.get("load_sc1") or 0) >= 1 and (atomic.get("inv_sc1") or 0) >= 1
        self.add_check("G4", "Agent-scope atomic load", measured,
                       "sc1 on the load, buffer_inv sc1 after", ok)

    def group_h(self):
        def latency(size_mib):
            rows = [r for r in self.chase_rows("size")
                    if to_int(r.get("size_mib")) == size_mib]
            if not rows:
                return "", None
            row = rows[0]
            wall = to_float(row.get("ns_per_load_wall"))
            real = to_float(row.get("ns_per_load_memrealtime"))
            return ("{} ns wall, {} ns s_memrealtime".format(fmt(wall, 1), fmt(real, 1)), wall)

        measured, _ = latency(1)
        self.add_info("H1", "L2 hit latency", measured, "INFO")
        measured, _ = latency(64)
        self.add_info("H2", "Infinity Cache latency", measured,
                      "INFO; about 218 ns from a secondary source")
        measured, wall = latency(1024)
        if not measured:
            self.add("H3", "HBM latency", "", "INFO; 250 ns to 2 us; flag only above 2 us",
                     UNAVAIL)
        else:
            self.add("H3", "HBM latency", measured,
                     "INFO; 250 ns to 2 us; flag only above 2 us",
                     MISMATCH if wall is not None and wall > 2000 else INFO)

        def fence_cost(kind, xcds):
            rows = [r for r in self.chase_rows("fence", kind=kind)
                    if to_int(r.get("xcds")) == xcds]
            if not rows:
                return ""
            row = rows[0]
            return "{} ns (agent {} ns, workgroup {} ns)".format(
                fmt(to_float(row.get("ns_fence_cost")), 1),
                fmt(to_float(row.get("ns_per_iter_agent")), 1),
                fmt(to_float(row.get("ns_per_iter_workgroup")), 1))

        self.add_info("H4", "Release fence cost", fence_cost("release", 1), "INFO; under 1 us")
        self.add_info("H5", "Acquire fence cost", fence_cost("acquire", 1), "INFO; under 1 us")
        parts = [p for p in (
            f"release {fence_cost('release', 8)}" if fence_cost("release", 8) else None,
            f"acquire {fence_cost('acquire', 8)}" if fence_cost("acquire", 8) else None) if p]
        self.add_info("H6", "Fence cost, eight XCDs", "; ".join(parts), "INFO")

        rows = self.chase_rows("pingpong")
        measured = ""
        if rows:
            measured = "{} ns one way (XCD {} to {}, {} rounds)".format(
                fmt(to_float(rows[0].get("ns_one_way")), 1), rows[0].get("xcd_a"),
                rows[0].get("xcd_b"), rows[0].get("rounds"))
        self.add_info("H7", "Cross-XCD one-way latency", measured,
                      "INFO; 116 to 202.5 ns from a secondary source")

    def group_i(self):
        list_text, list_ok = self.read("rocprof-list")
        present, instances, instance_detail = parse_rocprof_list(list_text if list_ok else "")
        measured, ok = "", None
        if list_ok:
            missing = [n for n in PMC_NAMES if not present[n]]
            measured = ("all {} present".format(len(PMC_NAMES)) if not missing
                        else "missing " + ", ".join(missing))
            ok = not missing
        self.add_check("I1", "Counter names present", measured,
                       "the six counter names plus TCC_BUBBLE_sum", ok)

        rd = self.pmc.get("TCC_EA0_RDREQ_sum")
        rd32 = self.pmc.get("TCC_EA0_RDREQ_32B_sum")
        wr = self.pmc.get("TCC_EA0_WRREQ_sum")
        wr64 = self.pmc.get("TCC_EA0_WRREQ_64B_sum")
        hit = self.pmc.get("TCC_HIT_sum")
        miss = self.pmc.get("TCC_MISS_sum")
        measured, ok = "", None
        if self.pmc:
            measured = ", ".join(f"{k} {v:,.0f}" for k, v in sorted(self.pmc.items()))
            values = [v for v in (rd, wr, hit, miss) if v is not None]
            ok = bool(values) and all(v > 0 for v in values)
        self.add_check("I2", "Counters readable", measured, "non-zero values, no error", ok)

        reads = bytes_read(self.pmc)
        writes = bytes_written(self.pmc)
        parts, ok = [], None
        if reads is not None:
            parts.append("reads {:.4f} GiB ({})".format(
                reads / GIB, pct((reads - GIB) / GIB * 100.0)))
        elif self.pmc:
            parts.append("reads not computable (TCC_BUBBLE_sum not collected)")
        if writes is not None:
            parts.append("writes {:.4f} GiB ({})".format(
                writes / GIB, pct((writes - GIB) / GIB * 100.0)))
        if rd is not None:
            parts.append("flat 64 B reads {:.4f} GiB ({})".format(
                64 * rd / GIB, pct((64 * rd - GIB) / GIB * 100.0)))
        if wr is not None:
            parts.append("flat 64 B writes {:.4f} GiB ({})".format(
                64 * wr / GIB, pct((64 * wr - GIB) / GIB * 100.0)))
        if reads is not None and writes is not None:
            ok = (abs(reads - GIB) / GIB <= 0.05 and abs(writes - GIB) / GIB <= 0.05)
        self.add_check("I3", "Bytes-from-requests arithmetic", ", ".join(parts),
                       "128 B per TCC_BUBBLE read and 64 B per write request, "
                       "both within 5% of 1 GiB", ok)

        measured, ok = "", None
        if self.ktrace:
            durations = [r["end"] - r["start"] for r in self.ktrace
                         if r["start"] is not None and r["end"] is not None]
            measured = "{} copy_kernel dispatch(es)".format(len(self.ktrace))
            if durations:
                measured += ", first {:,} ns".format(durations[0])
            ok = len(self.ktrace) >= 1 and bool(durations) and all(d > 0 for d in durations)
        self.add_check("I4", "Kernel trace", measured,
                       "one dispatch row with a duration", ok)

        self.add_info("I5", "TCC instance count",
                      instance_detail if list_ok and instances else "",
                      "INFO; 128 in SPX (gk)")

    def group_j(self):
        lscpu_text, lscpu_ok = self.read("lscpu")
        model = re.search(r"^Model name:\s*(.+)$", lscpu_text, re.M) if lscpu_ok else None
        nproc_text, nproc_ok = self.read("nproc")
        cores = first_int(nproc_text) if nproc_ok else None
        parts = [p for p in (model.group(1).strip() if model else None,
                             f"{cores} cores" if cores is not None else None) if p]
        self.add_info("J1", "CPU model and cores", ", ".join(parts),
                      "INFO; 8, 13 or 26 cores")

        free_text, free_ok = self.read("free")
        total = None
        if free_ok:
            m = re.search(r"^Mem:\s+(\d+)", free_text, re.M)
            total = first_int(m.group(1)) if m else None
        self.add_info("J2", "RAM", f"{total} GB total" if total is not None else "",
                      "INFO; 224 GB (1x), 448 GB (2x)")

        root_text, root_ok = self.read("df-root")
        home_text, home_ok = self.read("df-home")
        root_gib = df_avail_gib(root_text) if root_ok else None
        home_gib = df_avail_gib(home_text) if home_ok else None
        parts = [p for p in (f"/ {root_gib:,.0f} GiB free" if root_gib is not None else None,
                             f"home {home_gib:,.0f} GiB free" if home_gib is not None else None)
                 if p]
        target = home_gib if home_gib is not None else root_gib
        self.add_check("J3", "Disk free", ", ".join(parts), "over 100 GiB free",
                       target > 100 if target is not None else None)

        dl_text, dl_ok = self.read("download-log")
        measured = ""
        if dl_ok:
            rate = re.findall(r"([\d.]+\s*[KMG]B/s)", dl_text)
            measured = f"last rate {rate[-1]}" if rate else "captured, no rate line"
        self.add_info("J4", "Download rate", measured, "INFO; minutes for 31 GB")

        docker_text, docker_ok = self.read_clean("docker-gpu")
        measured, ok = "", None
        if docker_ok:
            ok = "gfx942" in docker_text
            measured = "gfx942 visible in the container" if ok else \
                "container ran, no gfx942 agent in its rocminfo"
        self.add_check("J5", "Container with GPU passthrough", measured,
                       "the GPU visible inside the container", ok)

        # Not a command's output: the collection cannot see when the VM was
        # provisioned, so the row carries what the operator timed.
        self.add("J6", "VM creation to first ssh", "recorded by hand",
                 "INFO; recorded by hand", INFO)

    def build(self):
        self.group_a()
        self.group_b()
        self.group_c()
        self.group_d()
        self.group_e()
        self.group_f()
        self.group_g()
        self.group_h()
        self.group_i()
        self.group_j()
        return self.rows

    # -- rendering ----------------------------------------------------------

    def header(self):
        uname_text, uname_ok = self.read("uname")
        parts = uname_text.split()
        host = parts[1] if uname_ok and len(parts) > 1 else "unknown host"
        counts = {}
        for row in self.rows:
            counts[row["result"]] = counts.get(row["result"], 0) + 1
        tally = ", ".join(f"{counts.get(k, 0)} {k}" for k in (PASS, MISMATCH, INFO, UNAVAIL))
        return (
            "Collected on {date} on {host}, ROCm {rocm}, hipcc {hipcc}. "
            "{n} checklist rows: {tally}. Raw captures are in raw/, one file per command, "
            "and collect.log holds the run. Every MISMATCH belongs in OPEN-PROBLEMS.md and "
            "the owning 99-open-questions.md with its date and command "
            "(docs/gpu-experiments/01-bringup/02-checklist.md, Sign-off)."
        ).format(date=self.out.name, host=host,
                 rocm=getattr(self, "rocm_version", None) or "unknown",
                 hipcc=getattr(self, "hipcc_version", None) or "unknown",
                 n=len(self.rows), tally=tally)

    def render(self):
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", "; ")

        lines = ["# Hardware collection " + self.out.name, "",
                 self.header(), "",
                 "| # | Check | Measured | Expected | Result |",
                 "|---|---|---|---|---|"]
        for row in self.rows:
            lines.append("| {} | {} | {} | {} | {} |".format(
                row["id"], cell(row["check"]), cell(row["measured"]),
                cell(row["expected"]), row["result"]))
        lines.append("")
        return "\n".join(lines)


def summarize(out_dir):
    """(rows, markdown) for one collection directory. Writes nothing."""
    summary = Summary(out_dir)
    summary.build()
    return summary.rows, summary.render()


def main(argv):
    if len(argv) != 2:
        print("usage: python3 env/hw/summarize.py env/hw/<YYYYMMDD>", file=sys.stderr)
        return 0
    out_dir = Path(argv[1])
    rows, markdown = summarize(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(markdown)
    print(markdown)
    print(f"written: {out_dir / 'summary.md'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
