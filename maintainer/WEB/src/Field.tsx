import { useEffect, useRef } from "react";

type Kind = 0 | 1 | 2;

type Node = {
  nx: number;
  ny: number;
  type: Kind;
  phase: number;
  period: number;
  flash: number;
};

type Edge = { a: number; b: number };
type Pulse = { e: number; u: number; dir: 1 | -1 };

const COLORS = ["#f0b27a", "#8fd0b0", "#a9c4ff"] as const;

function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function build(wide: boolean): { nodes: Node[]; edges: Edge[] } {
  const rand = mulberry32(11);
  const centers: [number, number][] = wide
    ? [
        [0.68, 0.36],
        [0.86, 0.58],
        [0.58, 0.7],
      ]
    : [
        [0.3, 0.34],
        [0.72, 0.32],
        [0.5, 0.72],
      ];
  const periods = [2.7, 1.45, 3.15];
  const nodes: Node[] = [];
  centers.forEach(([cx, cy], type) => {
    for (let i = 0; i < 16; i++) {
      const ang = rand() * Math.PI * 2;
      const rad = 0.025 + rand() * 0.105;
      nodes.push({
        nx: Math.min(0.97, Math.max(0.03, cx + Math.cos(ang) * rad * (wide ? 1.05 : 0.9))),
        ny: Math.min(0.95, Math.max(0.05, cy + Math.sin(ang) * rad)),
        type: type as Kind,
        phase: rand() * periods[type],
        period: periods[type] * (0.82 + rand() * 0.36),
        flash: 0,
      });
    }
  });

  const edges: Edge[] = [];
  const seen = new Set<string>();
  const add = (a: number, b: number) => {
    if (a === b) return;
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    if (seen.has(key)) return;
    seen.add(key);
    edges.push({ a, b });
  };

  for (let i = 0; i < nodes.length; i++) {
    const nearest: { j: number; d: number }[] = [];
    for (let j = 0; j < nodes.length; j++) {
      if (i === j || nodes[i].type !== nodes[j].type) continue;
      const dx = nodes[i].nx - nodes[j].nx;
      const dy = nodes[i].ny - nodes[j].ny;
      nearest.push({ j, d: dx * dx + dy * dy });
    }
    nearest.sort((p, q) => p.d - q.d);
    nearest.slice(0, 2).forEach((n) => add(i, n.j));
  }

  for (let type = 0; type < 3; type++) {
    const from = nodes.flatMap((n, i) => (n.type === type ? [i] : []));
    const to = nodes.flatMap((n, i) => (n.type === type ? [] : [i]));
    for (let k = 0; k < 4; k++) {
      add(from[Math.floor(rand() * from.length)], to[Math.floor(rand() * to.length)]);
    }
  }

  return { nodes, edges };
}

export function Field() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let nodes: Node[] = [];
    let edges: Edge[] = [];
    let pulses: Pulse[] = [];
    let wide: boolean | null = null;
    let w = 0;
    let h = 0;
    let raf = 0;
    let last = performance.now();

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(w * dpr));
      canvas.height = Math.max(1, Math.floor(h * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const nextWide = w / Math.max(h, 1) > 1.15;
      if (nextWide !== wide) {
        wide = nextWide;
        const built = build(nextWide);
        nodes = built.nodes;
        edges = built.edges;
        pulses = reduce
          ? edges.flatMap((_, index) => (index % 6 === 0 ? [{ e: index, u: 0.42, dir: 1 as const }] : []))
          : [];
        if (reduce) {
          nodes.forEach((n, i) => {
            if (i % 13 === 0) n.flash = 0.85;
          });
        }
      }
    };

    const draw = () => {
      if (w < 2 || h < 2) return;
      ctx.clearRect(0, 0, w, h);
      for (let type = 0; type < 3; type++) {
        let sx = 0;
        let sy = 0;
        let n = 0;
        nodes.forEach((node) => {
          if (node.type !== type) return;
          sx += node.nx;
          sy += node.ny;
          n += 1;
        });
        if (!n) continue;
        const x = (sx / n) * w;
        const y = (sy / n) * h;
        const radius = Math.min(w, h) * 0.34;
        const glow = ctx.createRadialGradient(x, y, 0, x, y, radius);
        glow.addColorStop(0, `${COLORS[type]}55`);
        glow.addColorStop(1, "transparent");
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(244, 240, 230, 0.22)";
      edges.forEach((edge) => {
        const a = nodes[edge.a];
        const b = nodes[edge.b];
        ctx.beginPath();
        ctx.moveTo(a.nx * w, a.ny * h);
        ctx.lineTo(b.nx * w, b.ny * h);
        ctx.stroke();
      });

      pulses.forEach((pulse) => {
        const edge = edges[pulse.e];
        if (!edge) return;
        const a = nodes[edge.a];
        const b = nodes[edge.b];
        const x = (a.nx + (b.nx - a.nx) * pulse.u) * w;
        const y = (a.ny + (b.ny - a.ny) * pulse.u) * h;
        ctx.fillStyle = COLORS[a.type];
        ctx.beginPath();
        ctx.arc(x, y, 2.3, 0, Math.PI * 2);
        ctx.fill();
      });

      nodes.forEach((node) => {
        const x = node.nx * w;
        const y = node.ny * h;
        ctx.fillStyle = COLORS[node.type];
        ctx.beginPath();
        ctx.arc(x, y, 3.1 + node.flash * 1.8, 0, Math.PI * 2);
        ctx.fill();
        if (node.flash > 0.08) {
          ctx.globalAlpha = node.flash * 0.55;
          ctx.strokeStyle = COLORS[node.type];
          ctx.beginPath();
          ctx.arc(x, y, 7 + (1 - node.flash) * 8, 0, Math.PI * 2);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      });
    };

    const advance = (dt: number) => {
      nodes.forEach((node, index) => {
        node.flash = Math.max(0, node.flash - dt * 1.5);
        node.phase += dt;
        if (node.phase < node.period) return;
        node.phase = 0;
        node.flash = 1;
        edges.forEach((edge, edgeIndex) => {
          if (edge.a === index) pulses.push({ e: edgeIndex, u: 0, dir: 1 });
          else if (edge.b === index) pulses.push({ e: edgeIndex, u: 1, dir: -1 });
        });
      });
      pulses.forEach((pulse) => {
        pulse.u += dt * 0.95 * pulse.dir;
      });
      if (pulses.length > 96) pulses.splice(0, pulses.length - 96);
      pulses = pulses.filter((pulse) => pulse.u >= 0 && pulse.u <= 1);
    };

    resize();
    draw();

    const onResize = () => {
      resize();
      if (reduce) draw();
    };
    const observer = new ResizeObserver(onResize);
    observer.observe(canvas);

    if (reduce) {
      return () => observer.disconnect();
    }

    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (!document.hidden) advance(dt);
      draw();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    const onVis = () => {
      last = performance.now();
    };
    document.addEventListener("visibilitychange", onVis);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  return <canvas ref={ref} className="field" aria-hidden="true" />;
}
