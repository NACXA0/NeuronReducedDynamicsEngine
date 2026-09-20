import type { ReactNode } from "react";
import type { Copy } from "./copy";
import { bench, fi, formatNum } from "./data";

type Pt = readonly [number, number];

const pad = { l: 54, r: 18, t: 18, b: 42 };

function line(pts: readonly Pt[], x: (v: number) => number, y: (v: number) => number) {
  return pts
    .map((p, i) => `${i === 0 ? "M" : "L"}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`)
    .join(" ");
}

function ticks(min: number, max: number, step: number) {
  const out: number[] = [];
  for (let v = min; v <= max + 1e-9; v += step) out.push(Number(v.toFixed(4)));
  return out;
}

function Frame({
  title,
  xlab,
  ylab,
  xmin,
  xmax,
  ymin,
  ymax,
  yStep,
  children,
}: {
  title: string;
  xlab: string;
  ylab: string;
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
  yStep: number;
  children: (map: { x: (v: number) => number; y: (v: number) => number; w: number; h: number }) => ReactNode;
}) {
  const w = 640;
  const h = 360;
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;
  const x = (v: number) => pad.l + ((v - xmin) / (xmax - xmin)) * iw;
  const y = (v: number) => pad.t + (1 - (v - ymin) / (ymax - ymin)) * ih;
  const xs = ticks(0.2, 0.8, 0.2);

  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={title}>
        {ticks(ymin, ymax, yStep).map((v) => (
          <g key={`y-${v}`}>
            <line className="grid" x1={pad.l} x2={w - pad.r} y1={y(v)} y2={y(v)} />
            <text className="tick" x={pad.l - 8} y={y(v) + 4} textAnchor="end">
              {formatNum(v, 0)}
            </text>
          </g>
        ))}
        {xs.map((v) => (
          <text key={`x-${v}`} className="tick" x={x(v)} y={h - 16} textAnchor="middle">
            {formatNum(v, 1)}
          </text>
        ))}
        <line className="axis" x1={pad.l} x2={w - pad.r} y1={y(ymin)} y2={y(ymin)} />
        <line className="axis" x1={pad.l} x2={pad.l} y1={pad.t} y2={y(ymin)} />
        <text className="axis-label" x={(pad.l + w - pad.r) / 2} y={h - 2} textAnchor="middle">
          {xlab}
        </text>
        <text className="axis-label" transform={`translate(14 ${(pad.t + y(ymin)) / 2}) rotate(-90)`} textAnchor="middle">
          {ylab}
        </text>
        {children({ x, y, w, h })}
      </svg>
    </figure>
  );
}

export function Curves({ c }: { c: Copy["curve"] }) {
  const ode = fi.map((row) => [row[0], row[1]] as const);
  const fit = fi.map((row) => [row[0], row[2]] as const);
  const lif = fi.map((row) => [row[0], row[3]] as const);

  return (
    <div className="curve-grid">
      <Frame title={c.fullTitle} xlab={c.xlab} ylab={c.ylab} xmin={0} xmax={0.85} ymin={0} ymax={250} yStep={50}>
        {({ x, y }) => (
          <>
            <path className="series series-lif" d={line(lif, x, y)} />
            <path className="series series-ode" d={line(ode, x, y)} />
            <path className="series series-f" d={line(fit, x, y)} />
          </>
        )}
      </Frame>
      <Frame title={c.detailTitle} xlab={c.xlab} ylab={c.ylab} xmin={0} xmax={0.85} ymin={0} ymax={70} yStep={10}>
        {({ x, y }) => (
          <>
            <path className="series series-ode" d={line(ode, x, y)} />
            <path className="series series-f" d={line(fit, x, y)} />
            <text className="callout" x="78" y="46">
              RMSE {formatNum(bench.nrde.rmse, 2)} Hz
            </text>
          </>
        )}
      </Frame>
      <ul className="legend">
        <li>
          <i className="swatch ode" />
          {c.ode}
        </li>
        <li>
          <i className="swatch f" />
          {c.f}
        </li>
        <li>
          <i className="swatch lif" />
          {c.lif}
        </li>
      </ul>
      <p className="caption">{c.caption}</p>
    </div>
  );
}

export function Pareto({ c }: { c: Copy["ledger"] }) {
  const w = 720;
  const h = 420;
  const box = { l: 58, r: 28, t: 28, b: 52 };
  const iw = w - box.l - box.r;
  const ih = h - box.t - box.b;
  const xmin = Math.log10(6);
  const xmax = Math.log10(4000);
  const ymax = 130;
  const x = (ms: number) => box.l + ((Math.log10(ms) - xmin) / (xmax - xmin)) * iw;
  const y = (rmse: number) => box.t + (1 - rmse / ymax) * ih;
  const xTicks = [10, 100, 1000];
  const yTicks = [0, 40, 80, 120];
  const pts = [
    { id: "lif", ms: bench.lif.ms, rmse: bench.lif.rmse, label: "LIF", dx: 12, dy: -10, anchor: "start" as const },
    { id: "nrde", ms: bench.nrde.ms, rmse: bench.nrde.rmse, label: "NRDE", dx: 12, dy: -14, anchor: "start" as const },
    { id: "ode", ms: bench.ode.ms, rmse: bench.ode.rmse, label: "AdExp", dx: -12, dy: -14, anchor: "end" as const },
  ];

  return (
    <figure className="chart pareto">
      <figcaption>{c.paretoTitle}</figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={c.paretoTitle}>
        {yTicks.map((v) => (
          <g key={v}>
            <line className="grid" x1={box.l} x2={w - box.r} y1={y(v)} y2={y(v)} />
            <text className="tick" x={box.l - 8} y={y(v) + 4} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {xTicks.map((v) => (
          <text key={v} className="tick" x={x(v)} y={h - 22} textAnchor="middle">
            {v}
          </text>
        ))}
        <line className="axis" x1={box.l} x2={w - box.r} y1={y(0)} y2={y(0)} />
        <line className="axis" x1={box.l} x2={box.l} y1={box.t} y2={y(0)} />
        <text className="axis-label" x={(box.l + w - box.r) / 2} y={h - 4} textAnchor="middle">
          {c.xlab}
        </text>
        <text className="axis-label" transform={`translate(16 ${(box.t + y(0)) / 2}) rotate(-90)`} textAnchor="middle">
          {c.ylab}
        </text>
        {pts.map((p) => (
          <g key={p.id} className={`dot dot-${p.id}`}>
            <circle cx={x(p.ms)} cy={y(p.rmse)} r={p.id === "nrde" ? 7 : 5.5} />
            <text x={x(p.ms) + p.dx} y={y(p.rmse) + p.dy} textAnchor={p.anchor}>
              {p.label}
            </text>
          </g>
        ))}
      </svg>
      <p className="caption">{c.paretoCaption}</p>
    </figure>
  );
}
