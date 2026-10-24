import { useEffect, useRef, useState } from "react";
import type { Copy } from "./copy";
import { bench, circuit, fill } from "./data";
import { FitSpark } from "./Figures";

const ODE_MS = bench.ode.ms;
const NRDE_MS = bench.nrde.ms;

function digits(totalMs: number): number {
  return totalMs / 1000 >= 1 ? 2 : 3;
}

function formatClock(ms: number, totalMs: number, unit: string): string {
  const capped = Math.min(Math.max(ms, 0), totalMs);
  return `${(capped / 1000).toFixed(digits(totalMs))} ${unit}`;
}

function Lane({
  name,
  shownMs,
  totalMs,
  unit,
  hot,
}: {
  name: string;
  shownMs: number;
  totalMs: number;
  unit: string;
  hot: boolean;
}) {
  const pct = Math.min(1, shownMs / totalMs) * 100;
  return (
    <div className="lane">
      <div className="lane-top">
        <span>{name}</span>
        <strong>{formatClock(shownMs, totalMs, unit)}</strong>
      </div>
      <div className="track" aria-hidden="true">
        <span className={hot ? "is-nrde" : undefined} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function ResultPanels({ c, replayToken }: { c: Copy["race"]; replayToken: number }) {
  const reduce = useState(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  )[0];
  const [elapsed, setElapsed] = useState(reduce ? ODE_MS : 0);
  const [done, setDone] = useState(reduce);
  const rafRef = useRef(0);

  useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    if (reduce) {
      setElapsed(ODE_MS);
      setDone(true);
      return;
    }
    setElapsed(0);
    setDone(false);
    const t0 = performance.now();
    const tick = (now: number) => {
      const t = now - t0;
      setElapsed(Math.min(t, ODE_MS));
      if (t < ODE_MS) rafRef.current = requestAnimationFrame(tick);
      else setDone(true);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [reduce, replayToken]);

  const skip = () => {
    cancelAnimationFrame(rafRef.current);
    setElapsed(ODE_MS);
    setDone(true);
  };

  const odeTime = formatClock(ODE_MS, ODE_MS, c.unit);
  const nrdeTime = formatClock(NRDE_MS, NRDE_MS, c.unit);
  const showSkip = replayToken > 0 && !reduce && !done;

  return (
    <div id="race">
      <div className="race">
        <section className="race-panel" aria-labelledby="race-who">
          <h2 id="race-who">{c.who}</h2>
          <p>{c.whoLede}</p>
          <p className="scale">
            {circuit.cells} {c.cellsLabel} · {circuit.activityMs} {c.activityLabel}
          </p>
          <Lane name={c.ode} shownMs={Math.min(elapsed, ODE_MS)} totalMs={ODE_MS} unit={c.unit} hot={false} />
          <Lane name={c.nrde} shownMs={Math.min(elapsed, NRDE_MS)} totalMs={NRDE_MS} unit={c.unit} hot />
          <p className="sr" aria-live="polite">
            {done ? fill(c.done, { ode: c.ode, odeTime, nrde: c.nrde, nrdeTime }) : ""}
          </p>
          {showSkip ? (
            <button type="button" className="skip-wait" onClick={skip}>
              {c.skip}
            </button>
          ) : null}
        </section>
        <section className="race-panel" aria-labelledby="race-fit">
          <h2 id="race-fit">{c.fit}</h2>
          <FitSpark caption={c.overlap} ode={c.ode} f={c.nrde} />
          <a className="race-more" href="#ledger">
            {c.more}
          </a>
        </section>
      </div>
      <div className="race-notes">
        <p>{c.playback}</p>
        <p>{fill(c.extrapolate, { sample: circuit.odeSampleCells, cells: circuit.cells })}</p>
      </div>
    </div>
  );
}
