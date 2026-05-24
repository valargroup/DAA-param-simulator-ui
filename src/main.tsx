import React from "react";
import { createRoot } from "react-dom/client";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  BarChart3,
  Database,
  Info,
  LineChart as LineChartIcon,
  RotateCcw,
  Shield,
  SlidersHorizontal,
} from "lucide-react";
import "./styles.css";

type BlockRow = {
  height: number;
  time: number;
  spacing: number | null;
  target: bigint;
  targetApprox: number;
};

type ParamSet = {
  id: string;
  label: string;
  window: number;
  damping: number;
  maxDownPct: number;
  maxUpPct: number;
  targetSpacing: number;
  enabled: boolean;
};

type ReplayPoint = {
  height: number;
  actualDifficulty: number;
  simulatedDifficulty: number;
  actualDifficultyRatio: number;
  simulatedDifficultyRatio: number;
  actualTargetRatio: number;
  simulatedTargetRatio: number;
  actualSpacing: number;
  expectedHistoricalSpacing: number;
  simulatedSpacing: number;
  errorSignal: number;
  boundedRatio: number;
  clamp: "up" | "down" | "none";
};

type ReplayResult = {
  params: ParamSet;
  warmup: number;
  points: ReplayPoint[];
  variance: {
    timespanCov: number;
    theoreticalCov: number;
    difficultyRollingStd: number;
    difficultyStd: number;
    difficultyMin: number;
    difficultyMax: number;
    difficultyRangePct: number;
    blockSpacingStd: number;
    errorStd: number;
    clampDownShare: number;
    clampUpShare: number;
  };
  halving: {
    blocks: number;
    hours: number;
    thresholdMultiplier: number;
  };
};

const CURRENT_PARAMS: ParamSet[] = [
  {
    id: "zcash",
    label: "Current Zcash",
    window: 17,
    damping: 4,
    maxDownPct: 32,
    maxUpPct: 16,
    targetSpacing: 75,
    enabled: true,
  },
  {
    id: "w108",
    label: "W 108",
    window: 108,
    damping: 4,
    maxDownPct: 32,
    maxUpPct: 16,
    targetSpacing: 75,
    enabled: true,
  },
  {
    id: "w300",
    label: "W 300",
    window: 300,
    damping: 4,
    maxDownPct: 32,
    maxUpPct: 16,
    targetSpacing: 75,
    enabled: true,
  },
];

const POW_LIMIT = (1n << 243n) - 1n;
const MTP_SPAN = 11;
const WINDOW_OPTIONS = [17, 51, 108, 300];
const DAMPING_OPTIONS = [4, 8];
const COLORS = ["#e8e0d4", "#c2410c", "#059669"];

function parseCsv(text: string): BlockRow[] {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  const index = (name: string) => headers.indexOf(name);
  const heightIdx = index("block_height");
  const timeIdx = index("time_unix");
  const spacingIdx = index("block_spacing_seconds");
  const targetIdx = index("difficulty_threshold");

  return lines
    .map((line) => {
      const cols = line.split(",");
      const target = BigInt(cols[targetIdx]);
      return {
        height: Number(cols[heightIdx]),
        time: Number(cols[timeIdx]),
        spacing: cols[spacingIdx] ? Number(cols[spacingIdx]) : null,
        target,
        targetApprox: Number(target) / Number(POW_LIMIT),
      };
    })
    .filter((row) => Number.isFinite(row.height) && row.target > 0n)
    .sort((a, b) => a.height - b.height);
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function meanBigInt(values: bigint[]): bigint {
  return values.reduce((sum, value) => sum + value, 0n) / BigInt(values.length);
}

function stdev(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    (values.length - 1);
  return Math.sqrt(variance);
}

function rollingStd(values: number[], width: number): number {
  if (values.length < width) return stdev(values);
  const samples: number[] = [];
  for (let i = width; i <= values.length; i += width) {
    samples.push(stdev(values.slice(i - width, i)));
  }
  return samples.reduce((sum, value) => sum + value, 0) / samples.length;
}

function exactClampBoundHalving(params: ParamSet): ReplayResult["halving"] {
  const targets = Array<bigint>(params.window).fill(1_000_000_000_000n);
  const maxSteps = params.window * 20;
  let thresholdMultiplier = 1;

  for (let blocks = 1; blocks <= maxSteps; blocks++) {
    const meanTarget = meanBigInt(targets.slice(-params.window));
    const nextTarget = (meanTarget * BigInt(100 + params.maxDownPct)) / 100n;
    targets.push(nextTarget);
    thresholdMultiplier = Number(nextTarget) / 1_000_000_000_000;
    if (thresholdMultiplier >= 2) {
      return {
        blocks,
        hours: (blocks * params.targetSpacing) / 3600,
        thresholdMultiplier,
      };
    }
  }

  return {
    blocks: maxSteps,
    hours: (maxSteps * params.targetSpacing) / 3600,
    thresholdMultiplier,
  };
}

function replayDaa(
  blocks: BlockRow[],
  params: ParamSet,
  meanTargetApprox: number,
): ReplayResult {
  const warmup = params.window + MTP_SPAN;
  const simTargets = blocks.map((block) => block.target);
  const simTimes = blocks.map((block) => block.time);
  const points: ReplayPoint[] = [];
  const errors: number[] = [];
  const timespans: number[] = [];
  const simulatedDifficulties: number[] = [];
  const observedSpacings: number[] = [];
  let clampDown = 0;
  let clampUp = 0;

  for (let i = warmup; i < blocks.length; i++) {
    const newerMedian = median(simTimes.slice(i - MTP_SPAN, i));
    const olderMedian = median(
      simTimes.slice(i - params.window - MTP_SPAN, i - params.window),
    );
    const actualTimespan = newerMedian - olderMedian;
    const targetTimespan = params.window * params.targetSpacing;
    const damped =
      targetTimespan +
      Math.trunc((actualTimespan - targetTimespan) / params.damping);
    const minTimespan = Math.floor(
      (targetTimespan * (100 - params.maxUpPct)) / 100,
    );
    const maxTimespan = Math.floor(
      (targetTimespan * (100 + params.maxDownPct)) / 100,
    );
    const bounded = Math.max(minTimespan, Math.min(maxTimespan, damped));
    const clamp =
      bounded === maxTimespan ? "down" : bounded === minTimespan ? "up" : "none";
    const meanTarget = meanBigInt(simTargets.slice(i - params.window, i));
    let nextTarget = (meanTarget / BigInt(targetTimespan)) * BigInt(bounded);
    if (nextTarget < 1n) nextTarget = 1n;
    if (nextTarget > POW_LIMIT) nextTarget = POW_LIMIT;

    const observedSpacing = blocks[i].spacing ?? blocks[i].time - blocks[i - 1].time;
    const nextTargetApprox = Number(nextTarget) / Number(POW_LIMIT);
    const spacingScale = blocks[i].targetApprox / nextTargetApprox;
    const simulatedSpacing = Math.max(1, observedSpacing * spacingScale);
    simTargets[i] = nextTarget;
    simTimes[i] = simTimes[i - 1] + simulatedSpacing;

    const simulatedDifficulty = 1 / nextTargetApprox;
    const actualDifficulty = 1 / blocks[i].targetApprox;
    const actualDifficultyRatio = meanTargetApprox / blocks[i].targetApprox;
    const simulatedDifficultyRatio = meanTargetApprox / nextTargetApprox;
    simulatedDifficulties.push(simulatedDifficultyRatio);
    observedSpacings.push(observedSpacing);
    errors.push((actualTimespan - targetTimespan) / targetTimespan);
    timespans.push(actualTimespan);
    if (clamp === "down") clampDown++;
    if (clamp === "up") clampUp++;

    points.push({
      height: blocks[i].height,
      actualDifficulty,
      simulatedDifficulty,
      actualDifficultyRatio,
      simulatedDifficultyRatio,
      actualTargetRatio: blocks[i].targetApprox,
      simulatedTargetRatio: nextTargetApprox,
      actualSpacing: observedSpacing,
      expectedHistoricalSpacing: params.targetSpacing * actualDifficultyRatio,
      simulatedSpacing,
      errorSignal: (actualTimespan - targetTimespan) / targetTimespan,
      boundedRatio: bounded / targetTimespan,
      clamp,
    });
  }

  const timespanMean =
    timespans.reduce((sum, value) => sum + value, 0) / Math.max(1, timespans.length);

  const difficultyMin = Math.min(...simulatedDifficulties);
  const difficultyMax = Math.max(...simulatedDifficulties);
  const difficultyMean =
    simulatedDifficulties.reduce((sum, value) => sum + value, 0) /
    Math.max(1, simulatedDifficulties.length);

  return {
    params,
    warmup,
    points,
    variance: {
      timespanCov: stdev(timespans) / Math.max(1, Math.abs(timespanMean)),
      theoreticalCov: 1 / Math.sqrt(params.window),
      difficultyRollingStd: rollingStd(simulatedDifficulties, params.window),
      difficultyStd: stdev(simulatedDifficulties),
      difficultyMin,
      difficultyMax,
      difficultyRangePct:
        ((difficultyMax - difficultyMin) / Math.max(1e-12, difficultyMean)) * 100,
      blockSpacingStd: stdev(observedSpacings),
      errorStd: stdev(errors),
      clampDownShare: clampDown / Math.max(1, points.length),
      clampUpShare: clampUp / Math.max(1, points.length),
    },
    halving: exactClampBoundHalving(params),
  };
}

function fmt(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function Equation({ children }: { children: React.ReactNode }) {
  return <div className="equation">{children}</div>;
}

function ParamEditor({
  params,
  onChange,
  onReset,
}: {
  params: ParamSet[];
  onChange: (next: ParamSet[]) => void;
  onReset: () => void;
}) {
  const update = (id: string, patch: Partial<ParamSet>) => {
    onChange(params.map((set) => (set.id === id ? { ...set, ...patch } : set)));
  };

  return (
    <section className="panel">
      <div className="sectionTitle">
        <SlidersHorizontal size={19} />
        <h2>Parameter Sets</h2>
        <button className="iconButton" onClick={onReset} title="Reset parameters">
          <RotateCcw size={17} />
        </button>
      </div>
      <div className="paramGrid">
        {params.map((set, index) => (
          <div className="paramCard" key={set.id}>
            <div className="paramHeader">
              <input
                value={set.label}
                onChange={(event) => update(set.id, { label: event.target.value })}
                aria-label="Parameter set label"
              />
              <label className="switch">
                <input
                  checked={set.enabled}
                  onChange={(event) => update(set.id, { enabled: event.target.checked })}
                  type="checkbox"
                />
                <span />
              </label>
            </div>
            <div className="swatches windowSwatches">
              {WINDOW_OPTIONS.map((window) => (
                <button
                  className={set.window === window ? "selected" : ""}
                  key={window}
                  onClick={() => update(set.id, { window })}
                >
                  {window}
                </button>
              ))}
              <button
                className={!WINDOW_OPTIONS.includes(set.window) ? "selected" : ""}
                onClick={() => update(set.id, { window: set.window })}
              >
                Custom
              </button>
            </div>
            <label>
              W averaging window custom entry
              <input
                min={11}
                max={1000}
                step={1}
                type="number"
                value={set.window}
                onChange={(event) => update(set.id, { window: Number(event.target.value) })}
              />
            </label>
            <div className="swatches dampingSwatches">
              {DAMPING_OPTIONS.map((damping) => (
                <button
                  className={set.damping === damping ? "selected" : ""}
                  key={damping}
                  onClick={() => update(set.id, { damping })}
                >
                  {damping}
                </button>
              ))}
              <button
                className={!DAMPING_OPTIONS.includes(set.damping) ? "selected" : ""}
                onClick={() => update(set.id, { damping: set.damping })}
              >
                Custom
              </button>
            </div>
            <label>
              D damping factor custom entry
              <input
                min={1}
                max={64}
                step={1}
                type="number"
                value={set.damping}
                onChange={(event) => update(set.id, { damping: Number(event.target.value) })}
              />
            </label>
            <div className="clampPair">
              <label>
                Threshold-up clamp
                <input
                  min={1}
                  max={100}
                  step={1}
                  type="number"
                  value={set.maxDownPct}
                  onChange={(event) =>
                    update(set.id, { maxDownPct: Number(event.target.value) })
                  }
                />
                <span>easier PoW, lower difficulty, faster blocks</span>
              </label>
              <label>
                Threshold-down clamp
                <input
                  min={1}
                  max={90}
                  step={1}
                  type="number"
                  value={set.maxUpPct}
                  onChange={(event) =>
                    update(set.id, { maxUpPct: Number(event.target.value) })
                  }
                />
                <span>harder PoW, higher difficulty, slower blocks</span>
              </label>
            </div>
            <label>
              T target spacing (seconds)
              <input
                min={1}
                max={600}
                step={1}
                type="number"
                value={set.targetSpacing}
                onChange={(event) => update(set.id, { targetSpacing: Number(event.target.value) })}
              />
            </label>
            <div className="colorKey" style={{ background: COLORS[index] }} />
          </div>
        ))}
      </div>
    </section>
  );
}

function App() {
  const [blocks, setBlocks] = React.useState<BlockRow[]>([]);
  const [params, setParams] = React.useState<ParamSet[]>(CURRENT_PARAMS);
  const [loadError, setLoadError] = React.useState<string | null>(null);

  React.useEffect(() => {
    fetch("/zcash_headers.csv")
      .then((response) => response.text())
      .then((text) => setBlocks(parseCsv(text)))
      .catch((error) => setLoadError(String(error)));
  }, []);

  const active = params.filter((set) => set.enabled);
  const meanTargetApprox = React.useMemo(
    () =>
      blocks.length
        ? blocks.reduce((sum, block) => sum + block.targetApprox, 0) / blocks.length
        : 0,
    [blocks],
  );
  const results = React.useMemo(
    () =>
      blocks.length
        ? active
            .map((set) => replayDaa(blocks, set, meanTargetApprox))
            .filter((result) => result.points.length)
        : [],
    [blocks, active, meanTargetApprox],
  );

  const chartData = React.useMemo(() => {
    if (!results.length || !blocks.length) return [];
    const step = Math.max(1, Math.floor(blocks.length / 650));
    const pointsByHeight = new Map(
      results.map((result) => [
        result.params.id,
        new Map(result.points.map((point) => [point.height, point])),
      ]),
    );

    return blocks
      .filter((_, index) => index % step === 0)
      .map((block) => {
        const row: Record<string, number | null> = {
          height: block.height,
          actual: meanTargetApprox / block.targetApprox,
          actualDifficulty: 1 / block.targetApprox,
          actualTarget: block.targetApprox,
          actualSpacing: block.spacing ?? null,
          expectedHistoricalSpacing:
            active[0]?.targetSpacing != null
              ? active[0].targetSpacing * (meanTargetApprox / block.targetApprox)
              : null,
        };
        results.forEach((result) => {
          const sampled = pointsByHeight.get(result.params.id)?.get(block.height);
          row[result.params.id] = sampled?.simulatedDifficultyRatio ?? null;
          row[`${result.params.id}Difficulty`] = sampled?.simulatedDifficulty ?? null;
          row[`${result.params.id}Target`] = sampled?.simulatedTargetRatio ?? null;
          row[`${result.params.id}Spacing`] = sampled?.simulatedSpacing ?? null;
        });
        return row;
      });
  }, [active, blocks, meanTargetApprox, results]);

  const varianceData = results.map((result) => ({
    name: result.params.label,
    empiricalCov: result.variance.timespanCov * 100,
    theoreticalCov: result.variance.theoreticalCov * 100,
    outputStd: result.variance.difficultyRollingStd,
    errorStd: result.variance.errorStd * 100,
  }));

  const offsetData = results.map((result) => ({
    label: result.params.label,
    offset: (6 / result.params.window) * 100,
    window: result.params.window,
  }));

  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="brand">
          <Shield size={18} />
          <div>
            <strong>Zcash DAA Lab</strong>
            <span>Parameter explorer</span>
          </div>
        </div>
        <nav>
          <p>Workspace</p>
          <a href="#parameters" className="active">
            <SlidersHorizontal size={15} />
            Parameters
          </a>
          <a href="#trajectory">
            <LineChartIcon size={15} />
            Threshold replay
          </a>
          <a href="#variance">
            <BarChart3 size={15} />
            Variance
          </a>
          <a href="#diagnostics">
            <Activity size={15} />
            Diagnostics
          </a>
        </nav>
        <div className="sidebarFooter">
          <Database size={14} />
          <div>
            <b>{blocks.length ? blocks.length.toLocaleString() : "..."}</b>
            <span>local rows</span>
          </div>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div>
            <h1>Zebra-style difficulty adjustment replay</h1>
            <span>
              Exact target arithmetic, approximate plots, local mainnet sample
            </span>
          </div>
          <div className="statusGroup">
            <span className="statusPill ok">CSV loaded</span>
            <span className="statusPill">Post-Blossom T=75s</span>
            {loadError && <span className="statusPill danger">{loadError}</span>}
          </div>
        </header>

        <main>
      <section className="explainer">
        <div>
          <h2>Target vs. Difficulty</h2>
          <p>
            Zebra adjusts the expanded difficulty threshold, also called the
            target. A larger target is easier, so displayed difficulty is the
            inverse of threshold. In this UI, relative difficulty is normalized
            so the 10k-block mean equals 1.0.
          </p>
          <Equation>
            relative_difficulty = mean(target_10k) / target
          </Equation>
          <Equation>
            relative_target = target / mean(target_10k)
          </Equation>
        </div>
        <div>
          <h2>Zebra DAA</h2>
          <Equation>
            MeanTarget = average(target[h-W], ..., target[h-1])
          </Equation>
          <Equation>
            ActualTimespan = median(time[h-11..h-1]) - median(time[h-W-11..h-W-1])
          </Equation>
          <Equation>
            Damped = W*T + trunc((ActualTimespan - W*T) / D)
          </Equation>
          <Equation>
            NextTarget = MeanTarget / (W*T) * clamp(Damped)
          </Equation>
        </div>
      </section>

      <div id="parameters" />
      <ParamEditor
        params={params}
        onChange={setParams}
        onReset={() => setParams(CURRENT_PARAMS)}
      />

      <section className="panel" id="trajectory">
        <div className="sectionTitle">
          <Info size={19} />
          <h2>Difficulty Trajectory</h2>
        </div>
        <p className="chartNote">
          The lines plot difficulty relative to the 10k-block mean. Higher
          difficulty means higher expected block times at fixed hashrate. Zebra
          controls difficulty threshold, which is inverse difficulty: threshold
          up makes PoW easier, threshold down makes PoW harder.
        </p>
        <details className="howItWorks">
          <summary>How counterfactual replay works</summary>
          <p>
            The replay preserves each observed block&apos;s realized
            hashrate/luck sequence by scaling solve time by the difficulty
            ratio. This is an algorithm-response view, not a miner-equilibrium
            model.
          </p>
          <Equation>
            expected_next_time = T * relative_difficulty
          </Equation>
          <Equation>
            implied_time(line) = actual_time * line_relative_difficulty / historical_relative_difficulty
          </Equation>
        </details>
        <div className="chartTall">
          <ResponsiveContainer>
            <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="height" minTickGap={40} />
              <YAxis
                domain={["auto", "auto"]}
                tickFormatter={(v) => fmt(Number(v), 8)}
              />
              <Tooltip
                formatter={(value, name, item) => {
                  const row = item.payload as Record<string, number>;
                  const key = String(item.dataKey);
                  const timeLabel =
                    key === "zcash"
                      ? `actual ${fmt(Number(row.actualSpacing), 1)}s`
                      : `implied ${fmt(Number(row[`${key}Spacing`]), 1)}s`;
                  return [
                    `${fmt(Number(value), 4)}x mean difficulty, ${timeLabel}`,
                    name,
                  ];
                }}
              />
              <Legend />
              <ReferenceLine
                y={1}
                stroke="#c4943a"
                strokeDasharray="4 4"
                label={{ value: "10k mean = 1.0", fill: "#c4943a", fontSize: 11 }}
              />
              {results.map((result, index) => (
                <Line
                  dataKey={result.params.id}
                  dot={false}
                  key={result.params.id}
                  name={result.params.label}
                  stroke={COLORS[index]}
                  strokeWidth={2}
                  connectNulls={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="twoCol" id="variance">
        <div className="panel">
          <div className="sectionTitle">
            <h2>Variance Comparison</h2>
          </div>
          <div className="equationGrid">
            <Equation>
              error_signal = (ActualTimespan - W*T) / (W*T)
            </Equation>
            <Equation>
              empirical_CoV = std(ActualTimespan) / mean(ActualTimespan)
            </Equation>
            <Equation>
              theoretical_CoV = 1 / sqrt(W)
            </Equation>
            <Equation>
              output_std = rolling_std(relative_difficulty, W)
            </Equation>
          </div>
          <div className="chart">
            <ResponsiveContainer>
              <BarChart data={varianceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(value) => `${fmt(Number(value), 2)}%`} />
                <Legend />
                <Bar dataKey="empiricalCov" fill="#2563eb" name="Empirical timespan CoV" />
                <Bar dataKey="theoreticalCov" fill="#94a3b8" name="1 / sqrt(W)" />
                <Bar dataKey="errorStd" fill="#c2410c" name="Error signal std" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="panel">
          <div className="sectionTitle">
            <h2>MTP Offset</h2>
          </div>
          <div className="offsetList">
            {offsetData.map((row) => (
              <div key={row.label}>
                <div className="offsetHeader">
                  <b>{row.label}</b>
                  <span>{fmt(row.offset, 1)}% of W</span>
                </div>
                <div className="timeline">
                  <span style={{ width: `${Math.min(100, row.offset)}%` }} />
                </div>
                <small>Approximate fixed 6-block MTP center offset / W={row.window}</small>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel" id="diagnostics">
        <h2>Live Diagnostics</h2>
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th>Set</th>
                <th>W</th>
                <th>D</th>
                <th>Threshold-up clamp</th>
                <th>Threshold-down clamp</th>
                <th>Output rolling std</th>
                <th>Clamp down share</th>
                <th>Clamp up share</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result) => (
                <tr key={result.params.id}>
                  <td>{result.params.label}</td>
                  <td>{result.params.window}</td>
                  <td>{result.params.damping}</td>
                  <td>{result.params.maxDownPct}%</td>
                  <td>{result.params.maxUpPct}%</td>
                  <td>{fmt(result.variance.difficultyRollingStd, 2)}</td>
                  <td>{fmt(result.variance.clampDownShare * 100, 2)}%</td>
                  <td>{fmt(result.variance.clampUpShare * 100, 2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="recommendation">
        <h2>Working Recommendation</h2>
        <p>
          Increase W substantially to reduce input noise and MTP-window
          misalignment, keep D available as a separate response-speed lever, and
          tune the target-up clamp independently because it dominates
          difficulty-down behavior during stress.
        </p>
        <p className="caveat">
          Caveat: this is not an equilibrium miner-behavior simulation. It
          replays the observed sequence of block arrivals under alternate
          targets.
        </p>
      </section>
        </main>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
