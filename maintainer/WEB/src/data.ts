/** Locked to the README comparison and docs/reports/evidence.md. Do not invent. */

export const repo = "https://github.com/NACXA0/NeuronReducedDynamicsEngine";

export const links = {
  repo,
  license: `${repo}/blob/main/LICENSE`,
  evidence: `${repo}/blob/main/docs/reports/evidence.md`,
  rfc001: `${repo}/blob/main/docs/rfcs/RFC-001.md`,
  rfc003: `${repo}/blob/main/docs/rfcs/RFC-003.md`,
  roadmap: `${repo}/blob/main/docs/roadmap.md`,
  readme: `${repo}/blob/main/README.md`,
  shiu: "https://doi.org/10.1038/s41586-024-07763-9",
  adexp: "https://doi.org/10.1152/jn.00686.2005",
  flybrainlab: "https://github.com/FlyBrainLab/FlyBrainLab",
  neurokernel: "https://github.com/neurokernel/neurokernel",
  pysr: "https://github.com/MilesCranmer/PySR",
} as const;

export const bench = {
  ode: { ms: 1805.8, rmse: 0, r2: 1 },
  nrde: { ms: 14.83, rmse: 1.61, r2: 0.994607 },
  lif: { ms: 11.98, rmse: 110.42, r2: -24.25 },
} as const;

/** Held-out midpoints of the aCC f–I grid. Columns: I nA, AdExp Hz, F Hz, unified LIF Hz. */
export const fi: readonly (readonly [number, number, number, number])[] = [
  [0.0267, 0, 0, 0],
  [0.08, 0, 0, 0],
  [0.1333, 0, 0, 0],
  [0.1867, 0, 0, 0],
  [0.24, 5, 0, 57.5],
  [0.2933, 10, 11.25, 85],
  [0.3467, 17.5, 17.5, 110],
  [0.4, 22.5, 23.75, 132.5],
  [0.4533, 27.5, 30, 150],
  [0.5067, 35, 35, 167.5],
  [0.56, 40, 41.25, 182.5],
  [0.6133, 47.5, 46.25, 197.5],
  [0.6667, 52.5, 51.25, 207.5],
  [0.72, 57.5, 57.5, 220],
  [0.7733, 62.5, 62.5, 230],
];

export const quickstart = [
  `git clone ${repo}.git`,
  "cd NeuronReducedDynamicsEngine",
  "uv sync",
  "uv run nrde fetch fly --tier pretrained",
  "uv run nrde run simulate \\",
  "  --fit datasets/fly/fits/aCC_l0.npz \\",
  "  --n 80 --steps 50",
].join("\n");

export function formatNum(n: number, digits = 2): string {
  let s = n.toFixed(digits);
  if (s.includes(".")) {
    s = s.replace(/0+$/, "").replace(/\.$/, "");
  }
  return s;
}

export const speedup = Math.round(bench.ode.ms / bench.nrde.ms);
