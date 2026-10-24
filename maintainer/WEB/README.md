# 静态宣传页

给仓库以外的人看的单页。首屏是一次对照测量的回放：谁先算完，以及两条曲线是否几乎重合。美观优先，口径不另起一套：类型可绑定不同的 \(F\)；不说已经重建真实动力学，也不说虚拟果蝇有有意义的行为。数字锁在 `src/data.ts`，与 [证据链](../../docs/reports/evidence.md) 和 README 对照表同一套。改证据链时改这一处。

仓库：<https://github.com/NACXA0/NeuronReducedDynamicsEngine>

## 本地预览

需要 Node.js。不要把 `node_modules/` 或 `dist/` 提交进 Git。

```bash
cd maintainer/WEB
npm ci
npm run dev
```

构建：

```bash
npm run build
```

产物在 `dist/`。技术栈是 React 19 和 Vite，没有后端。

## 用 Caddy 挂出去

示例是同目录的 [Caddyfile](Caddyfile)。构建之后：

```bash
caddy run --config Caddyfile
```

默认监听 `:8080`，站点根是 `dist`。带哈希的 `/assets/*` 长期缓存，`index.html` 不缓存死。

换成自己的域名时，删掉 `:8080` 站点，打开文件末尾注释里的 `example.com` 块，把域名改掉。Caddy 会为该域名申请证书。

页面上的 GitHub、证据链、RFC 和许可证都指向 <https://github.com/NACXA0/NeuronReducedDynamicsEngine>。
