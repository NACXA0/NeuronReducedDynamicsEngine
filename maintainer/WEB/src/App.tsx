import { useEffect, useState } from "react";
import { copy, type Lang } from "./copy";
import { bench, formatNum, links, quickstart, speedup } from "./data";
import { Field } from "./Field";
import { Curves, Pareto } from "./Figures";
import "./styles.css";

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem("nrde-lang");
    if (saved === "en" || saved === "zh") return saved;
  } catch {
    /* private mode */
  }
  return "zh";
}

function Mark() {
  return (
    <svg className="mark" viewBox="0 0 36 36" aria-hidden="true">
      <path d="M8 22 L21 11 M8 22 L24 28" />
      <circle cx="8" cy="22" r="3.4" className="c0" />
      <circle cx="21" cy="11" r="3.4" className="c1" />
      <circle cx="24" cy="28" r="3.4" className="c2" />
    </svg>
  );
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="gh">
      <path
        fill="currentColor"
        d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.7 7.7 0 0 1 8 3.87a7.7 7.7 0 0 1 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
      />
    </svg>
  );
}

export function App() {
  const [lang, setLang] = useState<Lang>(initialLang);
  const [solid, setSolid] = useState(false);
  const [copied, setCopied] = useState(false);
  const c = copy[lang];

  useEffect(() => {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    document.title = c.metaTitle;
    document.getElementById("meta-desc")?.setAttribute("content", c.metaDescription);
    try {
      localStorage.setItem("nrde-lang", lang);
    } catch {
      /* ignore */
    }
  }, [lang, c.metaTitle, c.metaDescription]);

  useEffect(() => {
    const hero = document.getElementById("top");
    if (!hero) return;
    const observer = new IntersectionObserver(([entry]) => setSolid(!entry.isIntersecting), {
      rootMargin: "-72px 0px 0px 0px",
      threshold: 0,
    });
    observer.observe(hero);
    return () => observer.disconnect();
  }, []);

  const choose = (next: Lang) => {
    setLang(next);
    setCopied(false);
  };

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(quickstart);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const methods = [
    { key: "ode" as const, ...c.ledger.ode, ...bench.ode },
    { key: "nrde" as const, ...c.ledger.nrde, ...bench.nrde },
    { key: "lif" as const, ...c.ledger.lif, ...bench.lif },
  ];

  return (
    <>
      <a className="skip" href="#content">
        {c.skip}
      </a>
      <header className={solid ? "nav is-solid" : "nav"}>
        <div className="wrap nav-bar">
          <a className="brand" href="#top">
            <Mark />
            <span>NRDE</span>
          </a>
          <nav className="nav-links" aria-label="NRDE">
            <a href="#contract">{c.nav.contract}</a>
            <a href="#ledger">{c.nav.ledger}</a>
            <a href="#curve">{c.nav.curve}</a>
            <a href="#close">{c.nav.close}</a>
          </nav>
          <div className="nav-end">
            <div className="lang" role="group" aria-label={c.langLabel}>
              <button type="button" aria-pressed={lang === "zh"} onClick={() => choose("zh")}>
                中文
              </button>
              <button type="button" aria-pressed={lang === "en"} onClick={() => choose("en")}>
                EN
              </button>
            </div>
            <a className="btn btn-ghost btn-nav" href={links.repo} target="_blank" rel="noreferrer">
              <GitHubIcon />
              <span>{c.nav.github}</span>
            </a>
          </div>
        </div>
      </header>

      <main id="content" tabIndex={-1}>
        <section className="hero" id="top">
          <Field />
          <div className="wrap hero-inner">
            <div className="hero-copy">
              <p className="kicker">NRDE · {c.hero.kicker}</p>
              <h1>
                <span>{c.hero.titleA}</span>
                <span>
                  {c.hero.titleB} <i>F</i> {c.hero.titleC}
                </span>
              </h1>
              <p className="lead">{c.hero.lead}</p>
              <div className="hero-actions">
                <a className="btn btn-light" href="#ledger">
                  {c.hero.cta}
                </a>
                <a className="btn btn-ghost" href={links.repo} target="_blank" rel="noreferrer">
                  <GitHubIcon />
                  GitHub
                </a>
              </div>
            </div>
            <div>
              <dl className="metrics">
                <div>
                  <dt>{c.hero.mRmse}</dt>
                  <dd>
                    {formatNum(bench.nrde.rmse, 2)}
                    <small>Hz</small>
                  </dd>
                </div>
                <div>
                  <dt>{c.hero.mSpeed}</dt>
                  <dd>
                    {speedup}
                    <small>×</small>
                  </dd>
                </div>
                <div>
                  <dt>{c.hero.mR2}</dt>
                  <dd>
                    {formatNum(bench.nrde.r2, 4)}
                    <small>R²</small>
                  </dd>
                </div>
              </dl>
              <p className="fig">{c.hero.fig}</p>
            </div>
          </div>
        </section>

        <section className="sheet" id="contract">
          <div className="wrap">
            <header className="sec-head">
              <p className="idx">{c.contract.idx}</p>
              <div>
                <h2>{c.contract.title}</h2>
                <p className="lede">{c.contract.lede}</p>
              </div>
            </header>
            <div className="eq-block" aria-label="I = W(r Δt) + I ext, r = F(I)">
              <p className="eq">
                <i>I</i>
                <span className="op">=</span>
                <i>W</i>
                <span className="arg">(r Δt)</span>
                <span className="op">+</span>
                <i>
                  I<sub>ext</sub>
                </i>
              </p>
              <p className="eq">
                <i>r</i>
                <span className="op">=</span>
                <i>F</i>
                <span className="arg">(I)</span>
              </p>
            </div>
            <ul className="chips">
              {c.contract.chips.map((chip) => (
                <li key={chip.k}>
                  <strong>{chip.k}</strong>
                  <span>{chip.v}</span>
                </li>
              ))}
            </ul>

            <header className="sec-head follow" id="ledger">
              <p className="idx">{c.ledger.idx}</p>
              <div>
                <h2>{c.ledger.title}</h2>
                <p className="lede">{c.ledger.lede}</p>
              </div>
            </header>
            <div className="cards">
              {methods.map((method) => (
                <article key={method.key} className={method.key === "nrde" ? "card card-hot" : "card"}>
                  <header>
                    <h3>{method.name}</h3>
                    {method.key === "nrde" ? <span className="badge">{speedup}×</span> : null}
                  </header>
                  <p>{method.stance}</p>
                  <dl>
                    <div>
                      <dt>{c.ledger.time}</dt>
                      <dd>{formatNum(method.ms, 2)} ms</dd>
                    </div>
                    <div>
                      <dt>{c.ledger.rmse}</dt>
                      <dd>{formatNum(method.rmse, 2)} Hz</dd>
                    </div>
                    <div>
                      <dt>{c.ledger.r2}</dt>
                      <dd>{formatNum(method.r2, 6)}</dd>
                    </div>
                  </dl>
                </article>
              ))}
            </div>
            <Pareto c={c.ledger} />
            <p className="footnote">{c.ledger.footnote}</p>
          </div>
        </section>

        <section className="curve" id="curve">
          <div className="wrap">
            <header className="sec-head">
              <p className="idx">{c.curve.idx}</p>
              <div>
                <h2>{c.curve.title}</h2>
                <p className="lede">{c.curve.lede}</p>
              </div>
            </header>
            <Curves c={c.curve} />
          </div>
        </section>

        <section className="close" id="close">
          <div className="wrap">
            <header className="sec-head">
              <p className="idx">{c.bounds.idx}</p>
              <div>
                <h2>{c.bounds.title}</h2>
                <p className="lede">{c.bounds.lede}</p>
              </div>
            </header>
            <ul className="bounds">
              {c.bounds.items.map((item) => (
                <li key={item.h}>
                  <h3>{item.h}</h3>
                  <p>{item.p}</p>
                </li>
              ))}
            </ul>

            <div className="start">
              <div>
                <h2>{c.start.title}</h2>
                <p className="lede">{c.start.lede}</p>
                <div className="hero-actions">
                  <a className="btn btn-fill" href={links.repo} target="_blank" rel="noreferrer">
                    <GitHubIcon />
                    {c.start.cta}
                  </a>
                  <a className="btn btn-line" href={links.evidence} target="_blank" rel="noreferrer">
                    {c.start.evidence}
                  </a>
                </div>
              </div>
              <div className="term">
                <div className="term-bar">
                  <span>bash</span>
                  <button type="button" onClick={onCopy}>
                    {copied ? c.start.copied : c.start.copy}
                  </button>
                </div>
                <pre>
                  <code>{quickstart}</code>
                </pre>
              </div>
            </div>
            <p className="footnote">{c.start.note}</p>
          </div>
        </section>
      </main>

      <footer className="foot">
        <div className="wrap">
          <p>{c.footer.line}</p>
          <ul className="cites">
            <li>
              <a href={links.shiu} target="_blank" rel="noreferrer">
                Shiu et al., 2024
              </a>
            </li>
            <li>Jolivet, Lewis &amp; Gerstner — SRM</li>
            <li>
              <a href={links.adexp} target="_blank" rel="noreferrer">
                Brette &amp; Gerstner, 2005 — AdExp
              </a>
            </li>
            <li>Kobayashi, 2009 — MAT</li>
            <li>Di Volo et al., 2019</li>
            <li>Becker, 2024</li>
            <li>
              <a href={links.flybrainlab} target="_blank" rel="noreferrer">
                FlyBrainLab
              </a>
            </li>
            <li>
              <a href={links.neurokernel} target="_blank" rel="noreferrer">
                Neurokernel
              </a>
            </li>
            <li>
              <a href={links.pysr} target="_blank" rel="noreferrer">
                PySR (Cranmer)
              </a>
            </li>
          </ul>
          <p className="foot-meta">
            <a href={links.license} target="_blank" rel="noreferrer">
              {c.footer.license}
            </a>
            <a href={links.rfc001} target="_blank" rel="noreferrer">
              RFC-001
            </a>
            <a href={links.rfc003} target="_blank" rel="noreferrer">
              RFC-003
            </a>
            <a href={links.roadmap} target="_blank" rel="noreferrer">
              Roadmap
            </a>
            <a href={links.readme} target="_blank" rel="noreferrer">
              README
            </a>
          </p>
        </div>
      </footer>
    </>
  );
}
