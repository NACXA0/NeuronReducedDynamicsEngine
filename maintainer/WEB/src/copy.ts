export type Lang = "zh" | "en";

export type Copy = {
  metaTitle: string;
  metaDescription: string;
  langLabel: string;
  skip: string;
  nav: { race: string; instance: string; ledger: string; close: string; faq: string; github: string };
  hero: { kicker: string; title: string; cta: string };
  race: {
    who: string;
    whoLede: string;
    fit: string;
    overlap: string;
    ode: string;
    nrde: string;
    cellsLabel: string;
    activityLabel: string;
    unit: string;
    playback: string;
    extrapolate: string;
    skip: string;
    more: string;
    done: string;
  };
  instance: { title: string; lede: string; horizon: string };
  uses: { title: string; lede: string; items: { h: string; p: string }[] };
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
    flyAlt: string;
    flyCaption: string;
  };
  start: {
    title: string;
    lede: string;
    copy: string;
    copied: string;
    cta: string;
    evidence: string;
    note: string;
    requires: string;
  };
  next: { title: string; items: { h: string; p: string }[] };
  faq: { title: string; items: { q: string; a: string }[] };
  footer: { line: string; license: string; licenseNote: string };
};

export const copy: Record<Lang, Copy> = {
  zh: {
    metaTitle: "NRDE — 果蝇脑子里的一个小回路",
    metaDescription:
      "NRDE 是一个神经回路仿真引擎。果蝇脑子里的一个小回路，算得又快又准。同一段活动，一边要一秒多，一边一下子就算完，两边的答案几乎重合。",
    langLabel: "语言",
    skip: "跳到正文",
    nav: {
      race: "赛跑",
      instance: "例子",
      ledger: "对照",
      close: "开始",
      faq: "问答",
      github: "GitHub",
    },
    hero: {
      kicker: "NRDE 是一个神经回路仿真引擎。",
      title: "果蝇脑子里的一个小回路，算得又快又准。",
      cta: "看这场赛跑",
    },
    race: {
      who: "谁先算完",
      whoLede: "同一段，一边要一秒多，一边一下子就算完。",
      fit: "算得对不对",
      overlap: "几乎重合",
      ode: "原来的算法",
      nrde: "NRDE",
      cellsLabel: "个细胞",
      activityLabel: "毫秒的活动",
      unit: "秒",
      playback: "这是一次对照测量的回放，不是你的电脑正在计算。",
      extrapolate:
        "左边的用时，是把 {sample} 个细胞的测量按比例放到这条 {cells} 个细胞的回路上，不是当场把这条回路一个一个算完。",
      skip: "跳过等待",
      more: "看对照",
      done: "{ode} {odeTime}。{nrde} {nrdeTime}。",
    },
    instance: {
      title: "为什么是果蝇",
      lede: "眼下用果蝇，是因为这种动物被研究得透，连线数据在手。引擎的一步是一张网、每一类一种反应，不绑在果蝇上。",
      horizon: "接下来想从这套算法里，长出新的仿生智能。",
    },
    uses: {
      title: "你可以拿它做什么",
      lede: "眼下能说的，是这一小段。",
      items: [
        { h: "这段回路算得便宜", p: "同一小段活动，原来的算法要一秒多，这边一下子就算完。" },
        { h: "不同种类不必共用同一种反应", p: "同一张网上，每一类可以有自己的反应。" },
        {
          h: "这一步以后可以嵌进别的循环",
          p: "一张网，加上每一类一种反应，就可以接上。果蝇的身体只是其中一个接口。",
        },
      ],
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
        "主机 AMD Ryzen 7 5800X，NumPy。墙钟换机器会变。在线 AdExp 的墙钟，是把 {sample} 个细胞的测量按比例放到 {cells} 个细胞上，不是当场把整条回路逐步算完。相对的是本库 AdExp Euler，不是活体，也不是 FlyBrainLab。",
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
      flyAlt: "果蝇线框身体",
      flyCaption: "接到身体上的接口。能走、不发散。不主张有意义的行为。",
    },
    start: {
      title: "五分钟，只做传播",
      lede: "不要装进系统 Python。仓库里用 uv，命令一律 uv run。表已经拟合好。",
      copy: "复制",
      copied: "已复制",
      cta: "在 GitHub 打开",
      evidence: "证据链",
      note: "克隆后 fetch --tier pretrained 只校验 33 个发布 npz 的 sha256。文件都在时，标准错误应是「已校验，未下载」。",
      requires:
        "需要 Python 3.10 或以上，以及 uv。五分钟示例在 CPU 上就能跑。页上的墙钟来自一台 AMD Ryzen 7 5800X 的 NumPy。GPU 和果蝇身体接口是可选项；身体接口需要 Python 3.12 到 3.14。",
    },
    next: {
      title: "接下来",
      items: [
        { h: "脱离果蝇文档", p: "让这一步在果蝇文档之外也能单独跑通。" },
        { h: "比较不同的反应", p: "在同一张网上比较不同的反应。" },
        { h: "也许成为一层", p: "再往后，同一套计算有可能被当成新结构的一层。这件事要另开规格，现在没有做。" },
      ],
    },
    faq: {
      title: "问答",
      items: [
        {
          q: "可以拿去商用吗？",
          a: "可以。Apache-2.0：可以商用、修改和再分发，须保留许可证声明；按现状提供，没有担保。",
        },
        {
          q: "和 NEST、Brian、NEURON 有什么差别？",
          a: "那些工具在线逐步解神经元方程。NRDE 先按类型收成一张反应，在线只沿连接传播再查表。本页对照的是本库逐步 AdExp 和统一 LIF，不是那三家的跑分。",
        },
        {
          q: "只能模拟果蝇吗？",
          a: "眼下的例子是果蝇的一个小回路。这一步是一张网加上每一类一种反应，不绑在果蝇上。别的动物的演示还没有。",
        },
        {
          q: "怎么盈利？",
          a: "现在先让「又快又准」能被别人独立复现。收入方式还没定，当面谈。",
        },
      ],
    },
    footer: {
      line: "降阶沿这些工作集成，不是另起一套动力学。口径以 RFC-001 为准。",
      license: "Apache-2.0",
      licenseNote: "可以商用、修改和再分发。须保留许可证声明。按现状提供，没有担保。",
    },
  },
  en: {
    metaTitle: "NRDE — A small circuit from the fly brain",
    metaDescription:
      "NRDE is a neural-circuit simulation engine. A small circuit from the fly brain, fast and still on the answer. One stretch takes more than a second the original way, and a moment with NRDE.",
    langLabel: "Language",
    skip: "Skip to content",
    nav: {
      race: "Race",
      instance: "Example",
      ledger: "Compare",
      close: "Start",
      faq: "Questions",
      github: "GitHub",
    },
    hero: {
      kicker: "NRDE is a neural-circuit simulation engine.",
      title: "A small circuit from the fly brain, fast and still on the answer.",
      cta: "Watch the race",
    },
    race: {
      who: "Who finishes first",
      whoLede: "The same stretch. One side takes more than a second. The other is done at once.",
      fit: "Does it still match",
      overlap: "Almost on top of each other",
      ode: "The original method",
      nrde: "NRDE",
      cellsLabel: "cells",
      activityLabel: "ms of activity",
      unit: "s",
      playback: "A playback of one comparison. Not a calculation on your computer.",
      extrapolate:
        "The slower time takes a measurement of {sample} cells and scales it to this circuit of {cells}. It is not a live run of every cell.",
      skip: "Skip the wait",
      more: "See the comparison",
      done: "{ode} {odeTime}. {nrde} {nrdeTime}.",
    },
    instance: {
      title: "Why a fly",
      lede: "The fly is the example because this animal is thoroughly studied and the wiring data is in hand. The step is a graph and one reaction per kind. It is not tied to the fly.",
      horizon: "The hope, ahead of us, is to grow a new bio-inspired intelligence from this step.",
    },
    uses: {
      title: "What you can do with it",
      lede: "What can be said now is about this small circuit.",
      items: [
        {
          h: "This stretch is cheap to run",
          p: "The same short stretch takes more than a second the original way, and a moment here.",
        },
        {
          h: "Kinds need not share one reaction",
          p: "On the same graph, each kind can keep its own reaction.",
        },
        {
          h: "The step can sit inside another loop",
          p: "A graph, plus one reaction per kind, is enough to connect. The fly body is one such interface.",
        },
      ],
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
        "Host: AMD Ryzen 7 5800X, NumPy. Wall clock moves with the machine. The online AdExp time scales a measurement of {sample} cells up to {cells}. It is not a live integration of the whole circuit. The reference is this library’s AdExp Euler, not a living cell and not FlyBrainLab.",
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
      flyAlt: "Wireframe fly body",
      flyCaption: "The body interface. It runs and does not diverge. That is not a claim of meaningful behavior.",
    },
    start: {
      title: "Five minutes, propagation only",
      lede: "Do not install into system Python. Use uv inside the clone. The tables are already fit.",
      copy: "Copy",
      copied: "Copied",
      cta: "Open on GitHub",
      evidence: "Evidence",
      note: "After a clone, fetch --tier pretrained only checks sha256 for 33 release npz files. When they are present, stderr should say they were verified and not downloaded.",
      requires:
        "Python 3.10 or newer, and uv. The five-minute example runs on CPU. The wall-clock numbers on this page are NumPy on an AMD Ryzen 7 5800X. GPU and the fly-body interface are optional. The body interface needs Python 3.12 up to 3.14.",
    },
    next: {
      title: "What comes next",
      items: [
        { h: "Without the fly docs", p: "Make this step runnable on its own, without the fly documentation." },
        { h: "Compare reactions", p: "Compare different reactions on the same graph." },
        {
          h: "Maybe a layer",
          p: "Later, the same computation might become a layer in a new structure. That needs its own specification. It is not done.",
        },
      ],
    },
    faq: {
      title: "Questions",
      items: [
        {
          q: "Can I use it commercially?",
          a: "Yes. Apache-2.0 allows commercial use, modification, and redistribution if you keep the license notice. It is provided as is, without warranty.",
        },
        {
          q: "How is this different from NEST, Brian, and NEURON?",
          a: "Those tools step neuron equations online. NRDE first reduces each kind to one reaction, then propagates along the connections and looks the reaction up. The comparison on this page is this library’s stepwise AdExp against one LIF, not a score against those three.",
        },
        {
          q: "Does it only simulate a fly?",
          a: "The example in hand is one small circuit from the fly. The step is a graph plus one reaction per kind. It is not tied to the fly. There is no demo of another animal yet.",
        },
        {
          q: "How does it make money?",
          a: "The present task is to let the speed and the match be reproduced by someone else. How it would earn money is not settled. That is a conversation.",
        },
      ],
    },
    footer: {
      line: "The reduction is an integration of this work, not a new dynamical law. Claims follow RFC-001.",
      license: "Apache-2.0",
      licenseNote: "Commercial use is allowed under Apache-2.0. Keep the license notice. It is provided as is, without warranty.",
    },
  },
};
