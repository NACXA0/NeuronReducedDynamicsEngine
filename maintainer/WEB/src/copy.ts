export type Lang = "zh" | "en";

export type Copy = {
  metaTitle: string;
  metaDescription: string;
  langLabel: string;
  skip: string;
  nav: { contract: string; ledger: string; curve: string; close: string; github: string };
  hero: {
    kicker: string;
    titleA: string;
    titleB: string;
    titleC: string;
    lead: string;
    cta: string;
    fig: string;
    mRmse: string;
    mSpeed: string;
    mR2: string;
  };
  contract: {
    idx: string;
    title: string;
    lede: string;
    chips: { k: string; v: string }[];
  };
  ledger: {
    idx: string;
    title: string;
    lede: string;
    ode: { name: string; stance: string };
    nrde: { name: string; stance: string };
    lif: { name: string; stance: string };
    time: string;
    rmse: string;
    r2: string;
    footnote: string;
    paretoTitle: string;
    paretoCaption: string;
    xlab: string;
    ylab: string;
  };
  curve: {
    idx: string;
    title: string;
    lede: string;
    ode: string;
    f: string;
    lif: string;
    fullTitle: string;
    detailTitle: string;
    caption: string;
    xlab: string;
    ylab: string;
  };
  bounds: {
    idx: string;
    title: string;
    lede: string;
    items: { h: string; p: string }[];
  };
  start: {
    title: string;
    lede: string;
    copy: string;
    copied: string;
    cta: string;
    evidence: string;
    note: string;
  };
  footer: { line: string; license: string };
};

export const copy: Record<Lang, Copy> = {
  zh: {
    metaTitle: "NRDE — 神经元降阶动力学引擎",
    metaDescription:
      "连接组进，按类型绑定的降阶函数 F 出。运行时做稀疏传播再查表，不再逐步积分 AdExp / HH。果蝇 MaleCNS 是当前实例。",
    langLabel: "语言",
    skip: "跳到正文",
    nav: {
      contract: "合同",
      ledger: "对照",
      curve: "曲线",
      close: "开始",
      github: "GitHub",
    },
    hero: {
      kicker: "神经元降阶动力学引擎",
      titleA: "连接组进",
      titleB: "按类型的",
      titleC: "出",
      lead: "运行时只做稀疏传播，再查一张按类型绑定的降阶函数。不再逐步积分 AdExp 或 Hodgkin–Huxley。果蝇 MaleCNS 是当前实例，不是引擎的全部范围。",
      cta: "看对照",
      fig: "图 1 · 三种类型分色的稀疏传播示意，不是实测轨迹",
      mRmse: "相对在线 AdExp 的发放率 RMSE",
      mSpeed: "249 细胞 × 200 ms，约快于逐步 Euler",
      mR2: "留出 f–I 的 R²",
    },
    contract: {
      idx: "01",
      title: "引擎合同只有三样东西",
      lede: "图、按类型绑定的表、这一步的外源电流。主线里身份维只有类型：一类一张 F。",
      chips: [
        { k: "W", v: "稀疏图。在线只沿着它传播，不逐步积细胞方程。" },
        { k: "F", v: "类型表。离线把该类的源动力学拟合进去，在线只查。" },
        { k: "Iₑₓₜ", v: "外源电流。感觉、刺激、或你自己接上的输入。" },
      ],
    },
    ledger: {
      idx: "02",
      title: "保真留在膝点，时间掉下去",
      lede: "同一电流网格的留出中点，对照的是本库 AdExp Euler。统一 LIF 同样快，但偏出一整条曲线。",
      ode: { name: "在线 AdExp", stance: "逐步积分。保真，全脑太贵。" },
      nrde: { name: "NRDE", stance: "类型各绑一张 F。稀疏传播之后查表。" },
      lif: { name: "统一 LIF", stance: "全图一条曲线。快，但不描述这组 AdExp。" },
      time: "249 × 200 ms",
      rmse: "RMSE",
      r2: "R²",
      footnote:
        "主机 AMD Ryzen 7 5800X，NumPy。墙钟换机器会变。相对的是本库 AdExp Euler，不是活体，也不是 FlyBrainLab。",
      paretoTitle: "图 2 · 误差与墙钟",
      paretoCaption: "横轴是墙钟（对数），纵轴是发放率 RMSE。又快又接近的位置在左下。",
      xlab: "墙钟 (ms)",
      ylab: "RMSE (Hz)",
    },
    curve: {
      idx: "03",
      title: "F 贴着教师，LIF 离开",
      lede: "aCC 的 AdExp 稳态发放率。F 是降阶表，LIF 是默认抑制细胞那条一维曲线，放在同一电流上。",
      ode: "在线 AdExp",
      f: "NRDE  F",
      lif: "统一 LIF",
      fullTitle: "图 3 · 全量程",
      detailTitle: "图 4 · 0–70 Hz，只留 AdExp 与 F",
      caption:
        "留出中点，不是训练结点。LIF 的 R² 为负，是因为它在这组电流上发放远快于 AdExp，不是算式写错。",
      xlab: "电流 I (nA)",
      ylab: "发放率 (Hz)",
    },
    bounds: {
      idx: "04",
      title: "这一版说什么，不说什么",
      lede: "v0.1 验收逼近误差、墙钟和接口 Demo。终局是抽出这一范式，不是回答果蝇生理学。",
      items: [
        { h: "类型可以绑定不同的 F", p: "默认兴奋、默认抑制、AdExp 替身，不必共用一条曲线。" },
        { h: "可嵌入的步进", p: "图加上类型表，再加外源。具身接口是基础设施，FlyGym 是其中一个后端。" },
        { h: "不说已经重建真实动力学", p: "发布集是拟合表和一条附属回路，不是 N 种活体电生理。" },
        { h: "不主张有意义的行为", p: "闭环只验收接口：能走、不发散。不构成行为主张。" },
      ],
    },
    start: {
      title: "五分钟，只做传播",
      lede: "不要装进系统 Python。仓库里用 uv，命令一律 uv run。表已经拟合好。",
      copy: "复制",
      copied: "已复制",
      cta: "在 GitHub 打开",
      evidence: "证据链",
      note: "克隆后 fetch --tier pretrained 只校验 33 个发布 npz 的 sha256。文件都在时，标准错误应是「已校验，未下载」。",
    },
    footer: {
      line: "降阶沿这些工作集成，不是另起一套动力学。口径以 RFC-001 为准。",
      license: "Apache-2.0",
    },
  },
  en: {
    metaTitle: "NRDE — Neuron Reduced-Dynamics Engine",
    metaDescription:
      "Connectome in, type-bound reduced F out. Sparse propagation, then a lookup — not a stepwise AdExp or Hodgkin–Huxley integration. The male Drosophila CNS is the current instance.",
    langLabel: "Language",
    skip: "Skip to content",
    nav: {
      contract: "Contract",
      ledger: "Compare",
      curve: "Curves",
      close: "Start",
      github: "GitHub",
    },
    hero: {
      kicker: "Neuron Reduced-Dynamics Engine",
      titleA: "Connectome in",
      titleB: "Type-bound",
      titleC: "out",
      lead: "At runtime the engine propagates on a sparse graph, then looks up a reduced function bound to cell type. It does not step AdExp or Hodgkin–Huxley. The male Drosophila CNS is the current instance, not the scope of the engine.",
      cta: "See the comparison",
      fig: "Fig. 1 · Sparse propagation, colored by three types. An illustration, not a recorded trace.",
      mRmse: "Firing-rate RMSE against online AdExp",
      mSpeed: "249 cells × 200 ms, versus stepwise Euler",
      mR2: "Held-out f–I R²",
    },
    contract: {
      idx: "01",
      title: "The contract is three objects",
      lede: "A graph, a table bound by type, and the external current on this step. The only identity axis on the main line is type: one F per class.",
      chips: [
        { k: "W", v: "The sparse graph. Online work is propagation, not a per-cell ODE." },
        { k: "F", v: "The type table. Source dynamics are fit offline. Online is a lookup." },
        { k: "Iₑₓₜ", v: "External current. Sensation, a stimulus, or an input you attach." },
      ],
    },
    ledger: {
      idx: "02",
      title: "Fidelity stays at the knee",
      lede: "Held-out midpoints of one current grid, against this library’s AdExp Euler. A single LIF is just as fast, and it misses the curve.",
      ode: { name: "Online AdExp", stance: "Step the ODE. Faithful, and too expensive for a whole brain." },
      nrde: { name: "NRDE", stance: "Each type binds its own F. Propagate, then look up." },
      lif: { name: "One LIF", stance: "A single curve for the whole graph. Fast, and not this AdExp." },
      time: "249 × 200 ms",
      rmse: "RMSE",
      r2: "R²",
      footnote:
        "Host: AMD Ryzen 7 5800X, NumPy. Wall clock moves with the machine. The reference is this library’s AdExp Euler, not a living cell and not FlyBrainLab.",
      paretoTitle: "Fig. 2 · Error against wall clock",
      paretoCaption: "Wall clock is logarithmic. Firing-rate RMSE is linear. Fast and close sits at the lower left.",
      xlab: "Wall clock (ms)",
      ylab: "RMSE (Hz)",
    },
    curve: {
      idx: "03",
      title: "F sits on the teacher",
      lede: "Steady firing of an aCC AdExp. F is the reduced table. The LIF is the default inhibitory cell, driven by the same current.",
      ode: "Online AdExp",
      f: "NRDE  F",
      lif: "One LIF",
      fullTitle: "Fig. 3 · Full range",
      detailTitle: "Fig. 4 · 0–70 Hz, AdExp and F only",
      caption:
        "Held-out midpoints, not the training knots. The LIF R² is negative because that cell fires far faster than the AdExp on this current — not because the arithmetic failed.",
      xlab: "Current I (nA)",
      ylab: "Rate (Hz)",
    },
    bounds: {
      idx: "04",
      title: "What this release claims",
      lede: "v0.1 is judged on approximation error, wall clock, and an interface demo. The aim is to extract the pattern, not to answer fly physiology.",
      items: [
        { h: "Types may bind different F", p: "Default excitatory, default inhibitory, and an AdExp stand-in need not share one curve." },
        { h: "A step you can embed", p: "Graph, type table, external current. The embodied interface is infrastructure; FlyGym is one backend." },
        { h: "Not a rebuilt electrophysiology", p: "The release is fitted tables and one accessory circuit, not N living cell models." },
        { h: "No claim of meaningful behavior", p: "The closed loop is an interface check: it runs and does not diverge. That is not a behavioral result." },
      ],
    },
    start: {
      title: "Five minutes, propagation only",
      lede: "Do not install into system Python. Use uv inside the clone. The tables are already fit.",
      copy: "Copy",
      copied: "Copied",
      cta: "Open on GitHub",
      evidence: "Evidence",
      note: "After a clone, fetch --tier pretrained only checks sha256 for 33 release npz files. When they are present, stderr should say they were verified and not downloaded.",
    },
    footer: {
      line: "The reduction is an integration of this work, not a new dynamical law. Claims follow RFC-001.",
      license: "Apache-2.0",
    },
  },
};
