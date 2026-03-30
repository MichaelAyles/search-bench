"""ANSI progress display for benchmark phases."""

import asyncio
import sys


class _Progress:
    """Thread-safe counters with a 2-second periodic table printer."""

    def __init__(self, total: int, tools: list[str], modes: list[str], label: str):
        self.total = total
        self.label = label
        self._lock = asyncio.Lock()
        self._counts: dict[str, dict] = {
            f"{t}/{m}": {"done": 0, "err": 0, "tttc": 0.0}
            for t in tools for m in modes
        }
        self._done = 0
        self._task: asyncio.Task | None = None

    async def record(self, tool: str, mode: str, tttc: float, error: bool) -> None:
        async with self._lock:
            self._done += 1
            key = f"{tool}/{mode}"
            if key in self._counts:
                self._counts[key]["done"] += 1
                self._counts[key]["tttc"] += tttc
                if error:
                    self._counts[key]["err"] += 1

    def _render(self) -> str:
        lines = [f"  {self.label} — {self._done}/{self.total}"]
        for key, c in sorted(self._counts.items()):
            if c["done"] == 0:
                continue
            avg = c["tttc"] / c["done"]
            lines.append(
                f"    {key:<20} {c['done']:>4} done  {c['err']:>2} err  avg {avg:5.1f}s"
            )
        return "\n".join(lines)

    async def _printer(self) -> None:
        prev_lines = 0
        while True:
            await asyncio.sleep(2)
            rendered = self._render()
            if prev_lines:
                # Move cursor up and clear to end-of-screen, then redraw
                sys.stdout.write(f"\x1b[{prev_lines}A\x1b[J")
            sys.stdout.write(rendered + "\n")
            sys.stdout.flush()
            prev_lines = rendered.count("\n") + 1

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._printer())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
