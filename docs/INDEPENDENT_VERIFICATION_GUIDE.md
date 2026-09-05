# Rb–Yb Rydberg 通道与门设计：独立复核所需 setting

> **给复核同学的说明**：这份文件只给出本工作的物理输入、模型假设、门参数和论文声称的结果，**不提供原有程序、函数、数据文件或运行命令**。请根据这些 setting 独立建立 Hamiltonian、编写 simulation 并完成收敛检查。在独立实现固定之前，不要用论文结果反向拟合 basis、符号或脉冲参数。
>
> 论文拟公开的数据与代码归档地址为 <https://github.com/wanda0929/RbYb_pub>；独立实现不应复制其中的门模拟代码。

---

## 1. 希望独立复核的四个问题

1. 指定的 $^{87}\mathrm{Rb}$–$^{171}\mathrm{Yb}$ Förster pair channel 是否确实近共振，并且在 tweezer-array 距离上主要表现为 $S+S\leftrightarrow P+P$ 两态交换？
2. 在本文列出的模型和误差假设下，给定的 composite pulse 是否实现 local-$Z$ 等价的 CZ 门，论文中的 loss-aware overlap 是否正确？
3. 指定的 vdW pair channel 是否在所列离散距离与截断基组内给出平滑、排斥、接近 $R^{-6}$ 的静态 interaction branch？
4. 现有证据是否只支持把 vdW 通道称为 selection-rule-allowed static candidate，而不支持任何 vdW gate 或 process-fidelity 结论？

这四项的证据等级不同：

- **Förster channel**：有限基组 pair-Hamiltonian 的通道主张，并附截断收敛检查；
- **Förster CZ**：在显式包含 $^{87}\mathrm{Rb}$ 核自旋的 all-$(m_J,m_I)$ pair spectrum 投影后的 driven model 中给出的门；
- **vdW channel**：finite-basis pair-Hamiltonian 的 sampled static-interaction 主张，并附 one-at-a-time basis-expansion 检查；
- **vdW gate**：本文没有指定脉冲、driven multichannel propagation 或 process-fidelity 结果。

因此，请不要把本工作概括成“Förster 和 vdW 两个通道都完成了 full gate simulation”。本论文只有 Förster channel 上的 composite CZ 是主要门设计；vdW 主要结果是可供未来 blockade gate 使用的静态 interaction resource。

---

## 2. 公共约定

### 2.1 原子、数据库与外场

| quantity | setting |
|---|---|
| species | $^{87}\mathrm{Rb}$ 和 $^{171}\mathrm{Yb}$ |
| atomic-structure package used for the quoted numbers | PairInteraction 2.5.0 |
| Rb database | v1.2 |
| Yb database | `Yb171_mqdt` v1.4 |
| dc electric field | 0 |
| quantization axis | magnetic field direction，定义为 $\hat z$ |
| $\theta$ | internuclear axis 与 $\hat z$ 的夹角 |
| $^{87}\mathrm{Rb}$ nuclear spin | $I=3/2$，$m_I=-3/2,-1/2,+1/2,+3/2$ |

独立复核可以使用另一套程序，但必须确认使用的是同一物理态和同一能量零点。Yb MQDT state 应由 **effective principal quantum number $\nu$、energy 和 label** 共同识别；整数 selector 会随数据库版本变化。

用于交叉检查 isolated-state linear Zeeman term 的数据库有效因子为：Rb $56S_{1/2}$ 与 $66S_{1/2}$ 的 $g_J=2.0023$，Rb $56P_{1/2}$ 的 $g_J=0.6659$；Yb $S(\nu=48.369927)$、$P(\nu=48.014048)$、$S(\nu=62.682293)$ 的 $g_F$ 分别为 $2.4414$、$1.3112$、$0.4007$。Rb nuclear-Zeeman term 另外使用 $g_I=-9.951414\times10^{-4}$。

### 2.2 单位

- 本文所有 Hamiltonian 参数写为 cyclic frequency $H/h$，单位 MHz。
- 时间单位为 $\mu\mathrm{s}$ 或 ns。
- 若数值矩阵存储的是 $H/h$，时间演化必须使用

$$
U(t)=\exp[-i2\pi(H/h)t].
$$

- 本文写出的 $\Omega/2\pi$ 和 $\Delta/2\pi$ 已经是 MHz，不应再除一次 $2\pi$。
- Förster 的 projected $2\times2$ generalized splitting 是 $30.8130\,\mathrm{MHz}$，有限基组两个 bright modes 的 splitting 是 $31.1099\,\mathrm{MHz}$；两者都不是 $|V|/h$，也不能在非零 defect 下直接写成 $2|V|/h$。
- vdW 的 $U/h>0$ 表示相对于同一磁场下 infinite-separation asymptote 的向上能移，即 repulsive interaction。

### 2.3 双比特约定

ket 顺序为

$$
|q_{\mathrm{Rb}}q_{\mathrm{Yb}}\rangle,
$$

computational basis 排列为

$$
\{|00\rangle,|01\rangle,|10\rangle,|11\rangle\}.
$$

目标门为

$$
U_{\mathrm{CZ}}=\operatorname{diag}(1,1,1,-1).
$$

允许在门后使用两个 virtual single-qubit $Z$ rotations 去除确定性的单原子相位，但不允许通过任意 two-qubit phase correction 消除 conditional-phase error。

---

## 3. Förster channel setting

### 3.1 两个目标 pair states

定义

$$
\begin{aligned}
|SS\rangle={}&|\mathrm{Rb}\,56S_{1/2}\rangle
|\mathrm{Yb};\nu=48.369927,L=0,F=1/2\rangle,\\
|PP\rangle={}&|\mathrm{Rb}\,56P_{1/2}\rangle
|\mathrm{Yb};\nu=48.014048,L=1,F=1/2\rangle.
\end{aligned}
$$

用于锁定同一 MQDT root 的 state information：

| state | quantum numbers / identity |
|---|---|
| Rb $56S$ | $n=56,l=0,j=1/2$ |
| Rb $56P$ | $n=56,l=1,j=1/2$ |
| Yb $S$ | $\nu=48.369927,L=0,F=1/2$；v1.4 selector $(n,l,s,f)=(53,0,1,1/2)$；energy $50396.3143067\,\mathrm{cm}^{-1}$ |
| Yb $P$ | $\nu=48.014048,L=1,F=1/2$；v1.4 selector $(n,l,s,f)=(52,1,0,1/2)$；energy $50395.6164394\,\mathrm{cm}^{-1}$ |

这些 energy 只用于确认选中了同一个数据库 state，不应直接代替独立能级查询和 defect 计算。

### 3.2 Spectroscopic characterization geometry

用于确认通道本身的 reference setting：

| quantity | setting |
|---|---:|
| magnetic field | $B=0$ |
| angle | $\theta=0$ |
| target electronic product | $(m_{J,\mathrm{Rb}},m_{F,\mathrm{Yb}})=(+1/2,+1/2)$ |
| main distance | $R=3.4\,\mu\mathrm{m}$ |
| useful distance region | approximately $3.4$–$3.5\,\mu\mathrm{m}$ |
| interaction | electric dipole–dipole |

这里的 fixed-electronic-$m$ calculation 是用于识别 two-level-dominated exchange channel 的 spectroscopic reference。Hyperfine-resolved calculation 的 initial component 另有 $m_I=+3/2$；还应在 $B=0$ 用下面的完整 hyperfine Hamiltonian 重算 all-$(m_J,m_I)$ spectrum 和 exchange。它们与第 4 节 $B=3.10\,\mathrm{G}$ driven-gate setting 分开报告。

### 3.3 $^{87}\mathrm{Rb}$ Rydberg hyperfine Hamiltonian

PairInteraction 2.5.0 提供 Rb electronic Hamiltonian（包括 electronic Zeeman term），但不携带 $^{87}\mathrm{Rb}$ nuclear spin。本文在 uncoupled basis

$$
|nLJ,m_J,m_I\rangle
$$

中显式加入

$$
\frac{H_{\mathrm{Rb}}}{h}
=\frac{H_{\mathrm{PI}}^{\mathrm{el}}}{h}
+\frac{A}{h}X
+\frac{B}{h}
\frac{3X^2+\tfrac32X-I(I+1)J(J+1)}
{2I(2I-1)J(2J-1)}
+g_I\frac{\mu_B}{h}B_zI_z,
\qquad X=\mathbf I\cdot\mathbf J.
$$

其中 $J=1/2$ 时 quadrupole term 为零；$g_I=-0.0009951414$ 是以 $\mu_B$ 归一化的 atomic-Hamiltonian convention，$\mu_B/h=1.39962449361\,\mathrm{MHz/G}$。请同时保留 electronic Zeeman 和 nuclear Zeeman，不能把超精细结构仅作为 asymptotic scalar shift 加在 $SS/PP$ 上。

本文使用的 hyperfine scaling 为

$$
A(n)/h=A_0/(n^*)^3,\qquad B(n)/h=B_0/(n^*)^3.
$$

| Rb series | $A_0$ (GHz) | $B_0$ (GHz) | provenance |
|---|---:|---:|---|
| $nS_{1/2}$ | 18.550 | 0 | $^{87}\mathrm{Rb}$ direct Rydberg measurement |
| $nP_{1/2}$ | 4.890 | 0 | measured $^{85}\mathrm{Rb}$ normalization scaled by isotope nuclear-$g$ ratio |
| $nP_{3/2}$ | 1.033 | 0.143 | fit to compiled low-$n$ $^{87}\mathrm{Rb}$ data |
| $nD_{3/2}$ | 0.828 | 0.050 | same |
| $nD_{5/2}$ | -0.357 | 0 | same |

由此 $A_{56S_{1/2}}/h=0.1255297\,\mathrm{MHz}$，zero-field $F=2$--$F=1$ interval 为 $0.2510595\,\mathrm{MHz}$；$A_{56P_{1/2}}/h=0.0322127\,\mathrm{MHz}$，interval 为 $0.0644254\,\mathrm{MHz}$。后者是 isotope-scaled estimate，没有额外加入 hyperfine anomaly uncertainty。

保留的 $F_J$ spectator states 没有可用的 Rydberg hyperfine normalization，因此 central calculation 令其 $A=B=0$。本文另做 deliberately oversized stress test，对全部 $F_J$ states 指定 $(A_0,B_0)=(0.828,0.143)\,\mathrm{GHz}$；这只是 sensitivity test，不是 $F$-state constants 的物理上界。

Electronic-centroid defect 为 $-0.763732\,\mathrm{MHz}$。对于 stretched components，

$$
\frac{\Delta_{\mathrm{hfs}}}{h}
=\frac{\Delta_{\mathrm{el}}}{h}
+\frac34\frac{A_{56S}-A_{56P_{1/2}}}{h}
=-0.693744\,\mathrm{MHz}.
$$

### 3.4 应独立构造的 pair model

不要只计算下面的 $2\times2$ projected model。应建立包含附近 spectator pair states 的 pair Hamiltonian，然后检查目标 subspace 是否自然隔离。在有序基 $\{|PP\rangle,|SS\rangle\}$ 中，以 $E_{PP}$ 为能量零点，二态投影为

$$
\frac{H_2}{h}=
\begin{pmatrix}
0 & V/h\\
V^*/h & \delta/h
\end{pmatrix}.
$$

其中 $\delta=E_{SS}-E_{PP}$。非零 defect 时必须使用

$$
\Delta\nu_2=\sqrt{(\delta/h)^2+4|V/h|^2},\qquad
P_{\max}^{(2)}=\frac{4|V|^2}{\delta^2+4|V|^2},\qquad
t_{\max}^{(2)}=\frac{1}{2\Delta\nu_2}.
$$

只有在 $\delta=0$ 时，才可化简为 $\Delta E=2|V|$ 和 $t_{\max}=h/(4|V|)$。

复核要求：

- atomic basis 必须同时包含生成 $|SS\rangle$、$|PP\rangle$ 及其附近 dipole-coupled pair states 所需的 levels；
- interaction 至少包含本工作使用的 dipole–dipole coupling；
- 给出 basis-size、energy-window、$n/l$ cutoff 的 convergence，而不是只给单一截断结果；
- fixed-electronic-$m$ characterization 与 hyperfine-resolved all-$(m_J,m_I)$ calculation 分开报告；
- eigenvector 必须按对 bare $SS/PP$ states 的 overlap 识别，不能只按 eigenenergy 排序。

### 3.5 论文声称的结果——这些是待验证输出，不是输入

| quantity | claimed result |
|---|---:|
| Rb $56S\rightarrow56P$ interval | $20.9222993\,\mathrm{GHz}$ |
| Yb $S\rightarrow P$ released interval | $20.9215356\,\mathrm{GHz}$ |
| electronic-centroid defect $[E_{SS}-E_{PP}]/h$ | $-0.763732\,\mathrm{MHz}$ |
| stretched-component hyperfine defect | $-0.693744\,\mathrm{MHz}$ |
| direct projected coupling $|V|/h$ | $15.401772\,\mathrm{MHz}$ |
| projected $2\times2$ generalized splitting $\Delta\nu_2$ | $30.813011\,\mathrm{MHz}$ |
| projected $2\times2$ $P_{\max}^{(2)}$ / $t_{\max}^{(2)}$ | $99.938565\%$ / $16.226911\,\mathrm{ns}$ |
| full finite-basis bright-mode splitting | $31.109926\,\mathrm{MHz}$ |
| full first $SS\rightarrow PP$ exchange maximum | $15.989779\,\mathrm{ns}$ |
| full population at that time: $(PP,SS,\mathrm{spectator})$ | $(97.286387,2.377034,0.336579)\%$ |
| bright upper mode: $(PP,SS,\mathrm{other})$ | $(57.754970,42.098743,0.146288)\%$ |
| bright lower mode: $(PP,SS,\mathrm{other})$ | $(42.167548,57.673937,0.158515)\%$ |
| all-mode unitarity upper bound on $SS\rightarrow PP$ transfer | $97.309617\%$ |
| final P0-4 axial HFS first-exchange transfer at $B=0$ | $93.922694\%$ |
| final P0-4 axial HFS first-exchange transfer at $B=3.10\,\mathrm{G}$ | $98.812568\%$ at $16.409014\,\mathrm{ns}$ |
| final P0-4 axial HFS sampled maximum (0–10 G grid) | $99.204761\%$ at $4\,\mathrm{G}$ |

主计算使用 fixed-$m$、$\Delta n=3$、$\ell\leq3$、$\pm80\,\mathrm{GHz}$ pair-energy window 和 2411 个 pair states。把 window 扩至 $\pm100\,\mathrm{GHz}$、把 $\Delta n$ 扩至 4、或把 $\ell_{\max}$ 扩至 4 时，full splitting 变化不超过 $0.00074\,\mathrm{MHz}$，transfer 变化不超过 $0.0045$ 个百分点，spectator population 变化不超过 $0.0023$ 个百分点。

Fig. 1(d) 的 HFS 曲线单独使用 final P0-4 reference basis：$\Delta n=3$、atomic $\pm80$ GHz、pair $\pm40$ GHz、$\Delta\ell=2$、interaction order 4，保留 axial conserved HFS block 的全部 eigenmodes。两条磁场曲线均从 SS 出发，无光驱动、无衰减，寻找第一个 bright-state period 内最大的 PP population（不是最早的微小 spectator ripple，也不是 PP-initial 0–80 ns global maximum）。3.10 G 时 $(PP,SS,\mathrm{spectator})=(98.812568,0.383730,0.803702)\%$，splitting $30.647414$ MHz；lower/upper $(PP,SS)$ weights 为 $(46.626553,52.977550)\%$ / $(52.908289,46.639480)\%$，与 P1-4 一致。zero-field centroid/stretched defects 分别是 $-0.763732/-0.693744$ MHz，而 3.10 G 的 dressed $SS-PP$ defect 是 $+4.663904$ MHz；不可互换。

HFS peak locator 从 801 到 83437 samples 加局部连续优化，3.10 G transfer 变化小于 $2\times10^{-15}$；这只是时间定位检查，不是新的 HFS basis-convergence claim。上面的 one-at-a-time basis checks 仍仅适用于 zero-field fixed-$m$。角度子图也仅是 fixed-$m$ diagnostic，不代表 full-HFS angular robustness。当前 Fig. 1 全部直接读取 `forster_characterization.json`，不依赖 gate JSON；旧 HFS scan 原样保留于 `historical_all_m_field_scan`。默认 reproducer checkpoint 更新缺失 HFS 点；`--rebuild-fixed-m` 才重建原始 fixed-$m$ 数据，绘图用 `python3 scripts/plot_channel_forster.py --characterization-only`。

请独立判断这些结果是否足以支持“two-level-dominated Förster channel”这一表述，并检查在 $R=3.3$–$3.5\,\mu\mathrm{m}$ 之外是否有 spectator mixing 或 exchange-contrast degradation。完成独立实现后，可与 `data/forster_characterization.json` 逐项交叉比对；该记录由 `scripts/reproduce_forster_characterization.py` 生成，并包含数据库文件的 SHA-256、扫描数据和收敛检查。

---

## 4. Förster composite CZ setting

### 4.1 Qubit encoding 与 optical access

Rb control qubit：

$$
|1\rangle_{\mathrm{Rb}}=|5S_{1/2},F=2,m_F=+2\rangle,
$$

$|0\rangle_{\mathrm{Rb}}$ 位于 $F=1$ manifold，并假设对所用 excitation ladder 为 dark。$|1\rangle_{\mathrm{Rb}}$ 通过

$$
5S_{1/2}\xrightarrow{780\,\mathrm{nm}}
5P_{3/2}\xrightarrow{480\,\mathrm{nm}}56S_{1/2},m_J=+1/2
$$

的 far-detuned two-photon ladder 激发。门模型把该过程视为 effective two-level drive，没有显式加入 $5P_{3/2}$、intermediate-state scattering 或 AC Stark shift。

Stretched ground state 有唯一的 uncoupled-basis decomposition，

$$
|5S_{1/2},F=2,m_F=+2\rangle
=|m_J=+1/2,m_I=+3/2\rangle.
$$

Electric-dipole ladder 不作用于 nuclear spin，因此它映射到 $|56S_{1/2},m_J=+1/2,m_I=+3/2\rangle$；不能把这里的 $m_I$ 平均掉。

Yb target qubit：

$$
\begin{aligned}
|1\rangle_{\mathrm{Yb}}&=|{}^3P_0,F=1/2,m_F=-1/2\rangle,\\
|0\rangle_{\mathrm{Yb}}&=|{}^3P_0,F=1/2,m_F=+1/2\rangle.
\end{aligned}
$$

理想 $\sigma^+$、$302.043\,\mathrm{nm}$ drive 将 $|1\rangle_{\mathrm{Yb}}$ 耦合到

$$
|S(\nu=48.369927),F=1/2,m_F=+1/2\rangle.
$$

$|0\rangle_{\mathrm{Yb}}$ 在该 manifold 中 dark，因为不存在 $F'=1/2,m'_F=+3/2$。目标 line 的计算 dipole 为 $2.39\times10^{-3}\,ea_0$。$|PP\rangle$ 不由 laser 直接驱动，只通过 dipole–dipole interaction 与 $|SS\rangle$ 相连。

### 4.2 Nominal geometry

| quantity | setting |
|---|---:|
| distance | $R_0=3.4\,\mu\mathrm{m}$ |
| angle | $\theta=0$ |
| magnetic field | $B=3.10\,\mathrm{G}$ |
| electric field | 0 |

### 4.3 Pulse sequence

顺序为：

1. Rb control resonant square $\pi$ pulse；
2. Yb target 五段 palindromic command $A$–$B$–$C$–$B$–$A$；
3. 与第一脉冲 phase coherent 的 reverse Rb $\pi$ pulse。

数值模型中两个 Rb pulse 使用相同 optical phase $\phi=0$ 和相同正 Rabi frequency；“reverse”只表示第二个 $\pi$ pulse 将 population de-excite，不表示 Rabi frequency 反号。

Rb pulse setting：

| quantity | value |
|---|---:|
| $\Omega_{\mathrm{Rb}}/2\pi$ | $5.000000\,\mathrm{MHz}$ |
| nominal $\pi$ duration | $100.000000\,\mathrm{ns}$ |

Yb ideal command：五段具有相同 duration。

| segment | $\Omega/2\pi$ (MHz) | $\Delta/2\pi$ (MHz) | duration (ns) |
|---|---:|---:|---:|
| A | 4.480374546 | -0.364774234 | 25.634115266 |
| B | 9.229350330 | -1.023517674 | 25.634115266 |
| C | 11.778486107 | +2.431262781 | 25.634115266 |
| B | 9.229350330 | -1.023517674 | 25.634115266 |
| A | 4.480374546 | -0.364774234 | 25.634115266 |

| timing quantity | value |
|---|---:|
| ideal Yb command | $128.170576\,\mathrm{ns}$ |
| filtered Yb target window | $160.028949\,\mathrm{ns}$ |
| total gate duration | $360.028949\,\mathrm{ns}$ |

这个门是 **driven composite revival**：conditional phase 来自 unblocked $|01\rangle$ 与 Förster-dressed $|11\rangle$ 的不同闭合轨迹。它不是 static blockade gate，也不是 adiabatic passage。

### 4.4 AOM response setting

Yb amplitude 和 detuning 分别通过 independent first-order response：

$$
\tau=\frac{t_{10-90}}{\ln 9},\qquad t_{10-90}=10\,\mathrm{ns},
$$

$$
\dot\Omega=\frac{\Omega_{\mathrm{cmd}}-\Omega}{\tau},\qquad
\dot\Delta=\frac{\Delta_{\mathrm{cmd}}-\Delta}{\tau}.
$$

具体边界条件：

- Yb optical amplitude 从 0 开始；
- 第一段 detuning 在 optical amplitude 打开前已经预设，不模拟从 0 到 $\Delta_A$ 的初始 slew；
- 五段 ideal command 后继续保留 $7\tau$ 的 amplitude ring-down；
- ring-down 期间 detuning command 保持最后 A 段的值；
- 这是 assumed effective bandwidth model，不是 measured AOM/RF transfer function。

### 4.5 Pair basis 与数值截断 setting

门 simulation 必须保留 conserved sector 内所有 electronic magnetic sublevels 和全部四个 $^{87}\mathrm{Rb}$ nuclear projections，不能把第 3 节 fixed-electronic-$m$ characterization 直接当作 driven gate Hamiltonian。在 nominal axial geometry，

$$
M_{\mathrm{tot}}=m_I+m_{J,\mathrm{Rb}}+m_{F,\mathrm{Yb}}=5/2
$$

严格守恒。实现可以用 exact block reduction：electronic $M=1,2,3,4$ 分别配对 $m_I=+3/2,+1/2,-1/2,-3/2$，hyperfine off-diagonal terms 在这些 sectors 间耦合。

Historical pulse-search/SCAN basis（仅保留作明确标记的 historical diagnostics；不再用于当前 Fig. 3，也不是最终脉冲的优化模型）：

| item | setting |
|---|---:|
| atomic expansion around target states | $\Delta n=3$ |
| Rb angular expansion | $\Delta l=2,\ \Delta j=2$ |
| Yb angular expansion | $\Delta l=2,\ \Delta f=3$ |
| atomic energy selection | `delta_energy = 80 GHz` |
| pair window | $\pm20\,\mathrm{GHz}$ about field-dressed $PP$ asymptote |
| magnetic sublevels | conserved $M_{\mathrm{tot}}=5/2$ block 内 all-$(m_J,m_I)$ |
| interaction | dipole–dipole |
| retained optically bright modes | $|\langle\lambda_j|SS\rangle|^2>10^{-6}$ |
| final propagation step | $0.125\,\mathrm{ns}$，并精确包含每个 command boundary |
| scan pair basis / connected HFS block | 2075 / 1075 states |

论文最终 headline pulse 由旧 search-basis pulse 作为 seed，在下面的 numerical-reference model 内重新局部优化并验证：

| item | setting |
|---|---:|
| atomic expansion around target states | $\Delta n=3$ |
| maximum orbital angular momentum | $\ell_{\max}=3$ |
| atomic energy selection | $\pm80\,\mathrm{GHz}$ |
| pair window | $\pm40\,\mathrm{GHz}$ about field-dressed $PP$ asymptote |
| interaction terms | cumulative through $R^{-4}$: dipole–dipole、dipole–quadrupole、quadrupole–dipole |
| pair basis / connected axial HFS block | 3684 / 3684 states |
| bright-mode cutoff | $10^{-6}$ |
| propagation step | maximum $0.125\,\mathrm{ns}$ |
| pulse reoptimization | 在 nominal 与 axial $\pm50\,\mathrm{nm}$、四个独立 Rabi vertices 加 nominal 的 final-model set 上做 fixed-duration local refinement |
| reference local $Z$ | $(\alpha,\beta)=(-3.1407723494,2.9411417009)\,\mathrm{rad}$ |

重优化在 objective evaluation 之前只构造并缓存 nominal 与 axial
$\pm50\,\mathrm{nm}$ 三个 P0-4 model，bright-mode cutoff 为 $10^{-6}$，objective
为八个 endpoint/amplitude vertices 加 nominal 上的
$\max(1-F)+0.02\,\operatorname{mean}(1-F)$。Bounded adaptive Nelder--Mead 只改变
六个 amplitude/detuning 参数，segment duration 固定为 $25.634115266\,\mathrm{ns}$。
该 deterministic run 用尽 400 iterations / 625 evaluations，optimizer
没有报告 convergence；候选点因为 exact $0.125\,\mathrm{ns}$ nominal 超过
0.999 且 endpoint objective 和 sampled bounded minimum 均不劣于 seed 而被接受。
它是 accepted local candidate，不是 converged/global optimum。

One-at-a-time convergence audit 分别使用：

- pair window $\pm20,\pm40,\pm60\,\mathrm{GHz}$；
- $\Delta n=3\rightarrow4$；
- atomic window $\pm80\rightarrow\pm160\,\mathrm{GHz}$；
- $\ell_{\max}=3\rightarrow4$；
- interaction order 从 pure dipole–dipole 到 through-$R^{-4}$，再到 PairInteraction 的 partial order-5 setting；
- bright-mode cutoff $10^{-4},10^{-5},10^{-6},10^{-7},10^{-8}$；
- maximum propagation step $1,0.5,0.25,0.125\,\mathrm{ns}$。

另做 gate-level bright-mode projection audit：在 nominal point 与最终 sampled
limiting point（$-50\,\mathrm{nm}$ axial、Yb 0.99、Rb 0.99）保持 P0-4 pulse、
decay prescription、time grid 和 reference local-$Z$ 全部不变，比较
$10^{-6}$、$10^{-8}$ 与同一 3684-state connected block 的全部 eigenmodes。

| point | cutoff | active modes | retained $SS$ weight | fixed-reference $Z$ overlap |
|---|---:|---:|---:|---:|
| nominal | $10^{-6}$ | 52 | 0.9999806033 | 0.9993184594783 |
| nominal | $10^{-8}$ | 197 | 0.9999993726 | 0.9993184594336 |
| nominal | all modes | 3684 | 1.0000000000 | 0.9993184594331 |
| limiting axial vertex | $10^{-6}$ | 56 | 0.9999814566 | 0.9986938231608 |
| limiting axial vertex | $10^{-8}$ | 209 | 0.9999993864 | 0.9986938224397 |
| limiting axial vertex | all modes | 3684 | 1.0000000000 | 0.9986938224424 |

因此 all-mode 与 selected $10^{-6}$ projection 的绝对 fidelity difference 在
nominal point 为 $4.52\times10^{-11}$，在 limiting point 为
$7.18\times10^{-10}$。Sparse-star propagation 与原 $10^{-6}$ dense-mode
implementation 的最大 Kraus-amplitude difference 为 $2.04\times10^{-13}$。
这里的 “all modes” 只表示保留同一 P0-4 connected block 的全部 eigenmodes；
它不是更大 pair basis，也不是 transverse full-angular calculation。

最终 P0-4 axial reference block 的详细 eigenspectrum 另存于
`forster_p1_4_reference_spectrum.json`。以 stretched $SS$ asymptote 为零，两个
target modes 位于 $-19.0116$ 与 $+11.6358\,\mathrm{MHz}$，其
$(SS,PP,\mathrm{other})$ weights 分别为
$(0.529775,0.466266,0.003959)$ 和
$(0.466395,0.529083,0.004522)$。最近 eigenstate 距 target doublet
$48.646\,\mathrm{MHz}$，但 target-subspace weight 可忽略；其最大 bare
component（0.8485）是
$\mathrm{Rb}\,53F_{5/2},m_J=3/2,m_I=-3/2+
\mathrm{Yb}\,D(\nu=48.3),F=5/2,m_F=5/2$。target-overlap 最大的 spectator
有 0.002284 target weight，主要属于
$\mathrm{Rb}\,56P_{3/2}+\mathrm{Yb}\,P(\nu=48.0)$。这些是 stated finite
P0-4 block 内的 direct diagonalization，不应简称为 untruncated
“full Hamiltonian”。

PairInteraction 的 partial order-5 setting 加入 quadrupole–quadrupole，但没有加入同为 $R^{-5}$ 的 dipole–octupole / octupole–dipole。因此它只能是 higher-order sensitivity check，不能叫作 complete $R^{-5}$ convergence。所有 basis rows 都保持脉冲不变；`fixed reference Z` 列固定上面的 $(\alpha,\beta)$，`phase recalibrated` 列只重算两个 virtual local phases。

在 $\theta=0$ 时，可以利用 exact total-$M$ symmetry，但该操作应当只是 exact block reduction，不能删除同一 block 中的 magnetic spectator states。复核者还应自行扩大 basis/window、降低 mode cutoff 和减小 time step，以确认结果收敛。

对于 sampled transverse offsets（最大 $\theta=0.843^\circ$），本文仍限制在上述 $M_{\mathrm{tot}}$ block：保留 angle-dependent $\Delta M=0$ tensor term，但省略 $\Delta M=\pm1,\pm2$ terms。因此 transverse points 不是 mathematically exact all-$M_{\mathrm{tot}}$ calculations。一个 electronic-only control 在原 dipole–dipole search model 的全部 19 geometries 比较 full-angle 与该 projection，76 个 loss-aware overlap values 的最大绝对差为 $4.97\times10^{-6}$；两个模型各自的 limiting vertex 都是 $\theta=0$ axial point。

### 4.6 Decay model

使用 PairInteraction 在 $0\,\mathrm{K}$ 下的 radiative lifetimes：

| state | lifetime ($\mu\mathrm{s}$) |
|---|---:|
| Rb $56P$ | 413.838578 |
| Rb $56S$ | 191.774081 |
| Yb $P$，$\nu=48.014048$ | 340.682588 |
| Yb $S$，$\nu=48.369927$ | 79.520898 |

使用 non-Hermitian no-jump evolution。对于 pair eigenmode $|\lambda_j\rangle$，令其在 $PP$、$SS$ 和其他 pair states 上的权重分别为 $w_{PP,j}$、$w_{SS,j}$ 和 $w_{\mathrm{other},j}$。mode decay rate 取

$$
\gamma_j=
w_{PP,j}(\gamma_{\mathrm{Rb},56P}+\gamma_{\mathrm{Yb},P})
+w_{SS,j}(\gamma_{\mathrm{Rb},56S}+\gamma_{\mathrm{Yb},S})
+w_{\mathrm{other},j}\max(\gamma_{PP},\gamma_{SS}).
$$

传播只保留 no-jump branch；损失的 norm 不通过 resolved jump operators 回流，但模型也不为 decay event 指定实际 final state。因此：

- 不是 trace-preserving Lindblad channel；
- 不包含 finite-temperature blackbody transfer；
- 不描述 Rb decay 回到 computational manifold 后的具体 Pauli error。

### 4.7 Computational return 与 loss-aware overlap

四个 inputs 的预期物理作用：

| input | intended evolution |
|---|---|
| $|00\rangle$ | 两原子均 dark |
| $|01\rangle$ | 只有 Yb 被驱动，完成 detuned closed trajectory |
| $|10\rangle$ | Rb 被激发到 $56S$，等待 target window 后返回 |
| $|11\rangle$ | Yb drive 访问 $SS$ 并耦合到 Förster-dressed manifold，最后 composite revival |

在模型中只保留每个 computational input 返回原 computational state 的 no-jump amplitude：

$$
K=\operatorname{diag}(k_{00},k_{01},k_{10},k_{11}).
$$

local correction 定义为

$$
C_Z=\operatorname{diag}
\left(1,e^{i\beta},e^{i\alpha},e^{i(\alpha+\beta)}\right).
$$

numerical-reference nominal point 使用的相位为

$$
\alpha=-3.140772349,\qquad
\beta=2.941141701\ \mathrm{rad}.
$$

Final-reference-selected pulse 在 historical dipole--dipole SCAN model 的 calibration 是 $(-3.098232263,2.983673967)\,\mathrm{rad}$；只应用于明确标为 SCAN-model 的 diagnostics。

$F_{\mathrm{avg}}$ 定义为 survival-weighted Haar average：

$$
F_{\mathrm{avg}}=
\frac{\operatorname{Tr}(K^\dagger K)
+|\operatorname{Tr}(U_{\mathrm{CZ}}^\dagger C_ZK)|^2}{20}.
$$

这不是四个 basis-state return probabilities 的普通平均，也不是 trace-preserving Lindblad channel 的 average fidelity 或 postselected conditional fidelity。

另外分别报告 mean computational survival

$$
\bar p_{\mathrm{surv}}=\frac{1}{4}\operatorname{Tr}(K^\dagger K)
$$

和 success-weighted conditional diagnostic $F_{\mathrm{cond}}=F_{\mathrm{avg}}/\bar p_{\mathrm{surv}}$。后者是 no-jump postselection diagnostic，不是 reconstructed CPTP channel fidelity。把所有 decay rates 设为零后，用重新校准的 local $Z$ 得到 coherent return-and-phase overlap，用于把 no-jump attenuation 与 coherent return/phase error 分开。

### 4.8 Robustness setting

Relative position 是 static bounded offset：

$$
\mathbf R=R_0\hat z+\delta\mathbf r,
\qquad |\delta\mathbf r|\le50\,\mathrm{nm}.
$$

它不是 Gaussian thermal distribution，也不包含 gate 期间的 atomic motion。

用于论文最终 sampled validation 的 geometry：

- nominal point；
- $|\delta\mathbf r|=25\,\mathrm{nm}$ shell 上 9 个 direction cosines：$-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1$；
- $|\delta\mathbf r|=50\,\mathrm{nm}$ shell 上相同 9 个 direction cosines；
- 共 19 个 geometries。

Rabi-scale error：

$$
s_{\mathrm{Yb}},s_{\mathrm{Rb}}\in\{0.99,1.01\},
$$

两者独立，因此每个 geometry 有 4 个 amplitude vertices。所有 76 个 joint scenarios 使用 nominal point 的同一对 $(\alpha,\beta)$，不能逐点重新拟合 local $Z$。

这里的 minimum 只是这 76 个点上的 sampled minimum，不是连续三维 $50\,\mathrm{nm}$ ball 上经过数学证明的 worst case。

另有一组与训练网格独立的 space-filling check：使用二维、unscrambled、deterministic Sobol sequence 的前 32 点，把第一维映射为

$$
r=50\,\mathrm{nm}\,u^{1/3},
$$

第二维映射为 direction cosine $2v-1$，再与同样的 4 个 amplitude vertices 组合。它只是球体内部的准随机覆盖检查，不是 thermal sampling，也不构成连续误差域的 worst-case certificate。

P1-1 numerical-reference extension 使用一个五维 deterministic unscrambled
Sobol sequence，三维位置按球体体积均匀映射，两个 Rabi scale 连续映射到
$[0.99,1.01]$。昂贵 Hamiltonian 不在每个 Sobol point 重建，而是在
$r=(0,12.5,25,37.5,50)\,\mathrm{nm}$、九个 direction cosines 和两个
$5$-point amplitude axes 上构造 cubic tensor response surface。共有 37 个
不同的 direct P0-4 retained-block geometries；surrogate 只负责 coverage 与找点。六个
独立 off-grid retained-block holdouts 的最大绝对误差为
$1.58\times10^{-5}$，rms 误差为 $9.76\times10^{-6}$。

| nested Sobol prefix | sampled minimum |
|---:|---:|
| 128 | 0.9988244812 |
| 256 | 0.9988244812 |
| 512 | 0.9988244812 |
| 1024 | 0.9988244812 |

四个 prefix 的 minimum 都是共有的第一个 deterministic Sobol point：zero
displacement、Yb 0.99、Rb 0.99；相等不代表已收敛到 continuous-domain
boundary。取十二个最低 Sobol points 做 constrained SLSQP local adversarial
search 后，leading candidates 回到 $-50\,\mathrm{nm}$ axial boundary。三个
candidate 用完整 P0-4 Hamiltonian 直接复算，最小值为 0.9986938232，对应
Yb 0.99、Rb 0.99。这个结果应称为 **sampled plus local
adversarial evidence**；不是 mathematically certified global minimum。

热运动必须作为独立 probability model，不能把上述 uniform-ball Sobol
设计解释成 thermal ensemble。文献输入与适用范围如下：

| species / platform | gate-time or preparation temperature | measured trap frequencies | use |
|---|---:|---:|---|
| $^{171}$Yb Rydberg gate [Peper2024] | $2.9\,\mu\mathrm K$ | radial $60\,\mathrm{kHz}$，axial $10\,\mathrm{kHz}$ | direct Yb baseline |
| $^{87}$Rb single tweezer [Kaufman2012] | $13(1)\,\mu\mathrm K$ release--recapture | $(154,150,30)\,\mathrm{kHz}$ | complete but cross-platform Rb benchmark |
| $^{87}$Rb Rydberg array [Zhang2025Cryogenic] | $5\,\mu\mathrm K$ before excitation | radial $123\,\mathrm{kHz}$ at $0.84\,\mathrm{mK}$ full depth | lacks axial and lowered-depth frequencies |

用前两行仅作 cross-apparatus scale benchmark，独立采样两种原子的 harmonic
Wigner phase space，seed 为 20260904，sample count 为 $2^{18}$，并把两个弱
trap axes 与 pair axis 对齐。在 target midpoint，relative one-sigma widths
为 $(49.8,50.5,265.7)\,\mathrm{nm}$，offset median 为 $193\,\mathrm{nm}$，
落入 50-nm ball 的概率为 0.0411。Weak-axis Doppler standard deviation 对 Yb
302-nm drive 为 $0.039\,\mathrm{MHz}$；对 Rb 780+480-nm ladder，counter- 与
co-propagating 情形分别为 $0.028$ 与 $0.119\,\mathrm{MHz}$。这些数字不属于
同一个 apparatus，所以不能据此报告 thermal gate fidelity；最终计算仍需
两种原子在相同实验条件下的 temperature、release-depth three-axis trap
frequencies、axis rotations、effective wavevectors 和 trap-off timing。

另外有三组固定 pulse 的 sensitivity audit：

1. **Spectroscopy-motivated deterministic envelope**：Peper *et al.* 对 Yb $S$ series 报告 $2.3\,\mathrm{MHz}$ rms fit residual；靠近本文 target 的 tabulated $P$ level 有 $+3.16\,\mathrm{MHz}$ residual。因为文章和 tabulated supplement 没有给 parameter covariance，而且这些 residual 来自与当前 PairInteraction v1.4 不完全相同的 fit version，所以它们不作为数据库修正或 confidence interval。只取
   $$
   \delta E_S/h\in\{-2.3,+2.3\}\,\mathrm{MHz},\qquad
   \delta E_P/h\in\{-3.2,+3.2\}\,\mathrm{MHz}
   $$
   的四个 signed vertices。Pair defect offset 是 $\delta E_S/h-\delta E_P/h$。Carrier-tracked case 令 Yb laser 跟随新的 $S$ line；fixed-laser case 还加入 detuning $-\delta E_S/h$。它不改变 matrix elements、wave functions、其他 levels 或 Rb structure。
2. **Pure target-pair defect scan**：保持 optically addressed $SS$ asymptote 和 pulse command 不变，只给目标 bare-$PP$ projector 加能移，使 $E_{SS}-E_{PP}$ 增加 $-5.5,-3.2,-2,-1,0,+1,+2,+3.2,+5.5\,\mathrm{MHz}$。这只是 scalar diagnostic。
3. **Axial dc electric field**：在 Rb、Yb isolated-atom 和 pair Hamiltonian 中沿 quantization axis 加入
   $$
   E_z=0,\ \pm0.001,\ \pm0.003,\ \pm0.01,\ \pm0.05,\ \pm0.10\,\mathrm{V/cm}.
   $$
   Carrier-tracked case 保持 laser 在该 field 的 isolated transition 上；fixed-zero-field-laser case 则使用 Rb $56S$ 与 Yb $S$ 的 isolated-state Stark shift 作为 control/target detuning。Ground/metastable-state Stark shift 没有加入。$3\,\mathrm{mV/cm}$ 取自已报道的 day-to-day field-compensation variation scale，并在 $\pm60\,\mathrm{GHz}$ pair window 下额外检查。

每个点都计算 reference local-$Z$ correction 固定不变的 overlap，以及只重新校准两个 local phases 后的 diagnostic；从不重新优化 pulse。Electric-field audit 只覆盖 axial direction；不能据此声称 arbitrary-direction stray-field robustness。

### 4.9 论文声称的 gate outputs——待独立验证

| quantity | claimed result |
|---|---:|
| numerical-reference nominal $F_{\mathrm{avg}}$ | 0.9993184595 |
| numerical-reference mean computational survival | 0.9993186403 |
| numerical-reference success-weighted conditional diagnostic | 0.9999998191 |
| numerical-reference sampled position-only minimum | 0.9992023215 |
| numerical-reference sampled position+amplitude minimum | 0.9986938232 |
| numerical-reference worst sampled point | $-50\,\mathrm{nm}$ axial，Yb scale 0.99，Rb scale 0.99 |
| P1-1 1024-point Sobol response-surface minimum | 0.9988244812 |
| P1-1 direct local-adversarial recheck minimum | 0.9986938232 |
| P1-2 maximum all-mode projection change | $7.18\times10^{-10}$ |
| SCAN-model nominal $F_{\mathrm{avg}}$ | 0.9989603273 |
| SCAN-model zero-decay coherent return-and-phase overlap | 0.9996161861 |
| SCAN-model zero-decay mean computational return | 0.9999921425 |
| SCAN-model 32-point Sobol position+amplitude minimum | 0.9981315010 |
| final-model sampled phase-recalibrated field maximum | 0.9993185936828 at $B=3.085\,\mathrm{G}$ |
| final-model axial nominal-amplitude minimum | 0.9992023214963 at $-50\,\mathrm{nm}$ |
| final-model independent Yb-amplitude minimum | 0.9990738202625 at $+1\%$ |
| final-model independent Rb-amplitude minimum | 0.9990719554109 at $-1\%$ |
| final-model unblocked Yb maximum Rydberg population | 0.9982655361089 |
| final-model unblocked final Yb Rydberg residual | $3.6835583412\times10^{-6}$ |
| final-model $|11\rangle$ final computational population | 0.9997756134877 |
| final-model maximum transient $PP$ population | 0.1521197629699 |
| final-model maximum transient $SS$ population | 0.0173042289797 |
| final-model maximum unshown spectator population | $7.6224900439\times10^{-4}$ |
| final-model final unshown spectator population | $3.8602689003\times10^{-7}$ |

当前 Fig. 3 的全部 driven curves 和最后七个 population quantities 使用 final P0-4 reference model、$10^{-6}$ cutoff、$0.125\,\mathrm{ns}$ maximum step。Ideal command 不变。Response time 的 $2,5,10,15,20\,\mathrm{ns}$ 五点分别为 0.9993407127558、0.9993321962781、0.9993184594783、0.9992909515334、0.9992373815341，每点只重新校准 nominal local-$Z$；不重新优化 command。Axial trace 直接读取 P1-1 的 $0,\pm12.5,\pm25,\pm37.5,\pm50\,\mathrm{nm}$ exact retained-block nodes，而非 response-surface interpolation；它和两个 independent amplitude traces 固定 $3.10\,\mathrm{G}$、$10\,\mathrm{ns}$ nominal correction。这些 one-dimensional minima 不是 joint minimum。

`scripts/reproduce_forster_gate.py` 默认重建一个 nominal reference model，并在读取 P0、P1-1 和 `forster_p0_4_field_scan.json` 后检查固定输入配置；输出中记录这些输入文件当时的 SHA-256（读取时不与另一个预存 hash 清单比对），生成当前 `bounded_minimax_forster_gate_results.json` 和四联图；`--plot-only` 只重画。旧 SCAN record 保存在该 JSON 的 `historical_scan_basis_diagnostics.record` 中，不参与当前图。`--historical-scan` / `--optimize` 使用单独的 historical 输出文件，不覆盖当前图。

Numerical convergence：

| one-at-a-time setting | connected size | splitting (MHz) | min target weight | static transfer | max driven spectator | fixed-reference $Z$ | phase-recalibrated |
|---|---:|---:|---:|---:|---:|---:|---:|
| pair window $\pm20\,\mathrm{GHz}$ | 2075 | 30.724861 | 0.9954846 | 0.9832421 | 0.0007390 | 0.9991045 | 0.9992470 |
| numerical reference | 3684 | 30.647414 | 0.9954777 | 0.9881177 | 0.0007622 | 0.9993185 | 0.9993185 |
| pair window $\pm60\,\mathrm{GHz}$ | 5212 | 30.654708 | 0.9954692 | 0.9879627 | 0.0007654 | 0.9993177 | 0.9993178 |
| $\Delta n=4$ | 4658 | 30.642617 | 0.9954653 | 0.9879206 | 0.0007651 | 0.9993182 | 0.9993188 |
| atomic window $\pm160\,\mathrm{GHz}$ | 6359 | 30.647436 | 0.9954776 | 0.9881151 | 0.0007623 | 0.9993185 | 0.9993185 |
| $\ell_{\max}=4$ | 4398 | 30.647412 | 0.9954777 | 0.9881178 | 0.0007622 | 0.9993185 | 0.9993185 |
| dipole–dipole only | 1872 | 30.783215 | 0.9974761 | 0.9884960 | 0.0001950 | 0.9989833 | 0.9992127 |
| partial order 5 | 3684 | 30.669294 | 0.9954822 | 0.9881088 | 0.0007594 | 0.9993179 | 0.9993180 |

Numerical-reference field-dressed defect 是 $4.663904\,\mathrm{MHz}$。Pair window $\pm40\rightarrow\pm60\,\mathrm{GHz}$ 令 fixed-$Z$ overlap 改变 $7.7\times10^{-7}$；$\Delta n$、atomic window 与 $\ell_{\max}$ 的 one-at-a-time span 是 $2.9\times10^{-7}$。Bright cutoff $10^{-4}\rightarrow10^{-8}$ 改变 $1.3\times10^{-9}$；time step $1\rightarrow0.125\,\mathrm{ns}$ 改变 $6.9\times10^{-8}$。这些数字支持百分数保留两位，不支持把 optimizer 的全部小数解释为 physical precision。

Spectroscopy-motivated deterministic envelope：

| $\delta E_S/h$ (MHz) | $\delta E_P/h$ (MHz) | defect offset (MHz) | tracked, fixed $Z$ | tracked, recal. $Z$ | fixed laser, fixed $Z$ | fixed laser, recal. $Z$ |
|---:|---:|---:|---:|---:|---:|---:|
| -2.3 | -3.2 | +0.9 | 0.9987514120 | 0.9991264148 | 0.9327951979 | 0.9850125816 |
| -2.3 | +3.2 | -5.5 | 0.9814455270 | 0.9924114906 | 0.9114981460 | 0.9574685035 |
| +2.3 | -3.2 | +5.5 | 0.9744684516 | 0.9907734355 | 0.9120576513 | 0.9607300063 |
| +2.3 | +3.2 | -0.9 | 0.9988011156 | 0.9991523426 | 0.9343700726 | 0.9859490752 |

这些是 deterministic sensitivity vertices，不是 error bars。

Pure target-pair-defect sensitivity：

| defect offset (MHz) | fixed reference $Z$ | phase-recalibrated |
|---:|---:|---:|
| -5.5 | 0.9814455270 | 0.9924114906 |
| -3.2 | 0.9930059383 | 0.9970805679 |
| -2.0 | 0.9968010817 | 0.9984665469 |
| -1.0 | 0.9986801026 | 0.9991121291 |
| 0 | 0.9993184595 | 0.9993184595 |
| +1.0 | 0.9986174049 | 0.9990820448 |
| +2.0 | 0.9964381352 | 0.9983643580 |
| +3.2 | 0.9916204630 | 0.9967609695 |
| +5.5 | 0.9744684516 | 0.9907734355 |

Axial dc-electric-field sensitivity in the numerical-reference basis：

| $E_z$ (V/cm) | tracked, fixed $Z$ | tracked, recal. $Z$ | fixed laser, fixed $Z$ | fixed laser, recal. $Z$ |
|---:|---:|---:|---:|---:|
| -0.10 | 0.8480736640 | 0.9401783520 | 0.7678789459 | 0.9239900858 |
| -0.05 | 0.9919245787 | 0.9968704355 | 0.9840449452 | 0.9948210891 |
| -0.01 | 0.9993066725 | 0.9993128664 | 0.9992948126 | 0.9993102563 |
| -0.003 | 0.9993179791 | 0.9993180180 | 0.9993179448 | 0.9993180546 |
| -0.001 | 0.9993183757 | 0.9993183772 | 0.9993183810 | 0.9993183837 |
| 0 | 0.9993184595 | 0.9993184595 | 0.9993184595 | 0.9993184595 |
| +0.001 | 0.9993185868 | 0.9993185968 | 0.9993185906 | 0.9993186028 |
| +0.003 | 0.9993184927 | 0.9993186390 | 0.9993184172 | 0.9993186633 |
| +0.01 | 0.9993036701 | 0.9993134391 | 0.9992902630 | 0.9993103748 |
| +0.05 | 0.9905429966 | 0.9963746320 | 0.9824812293 | 0.9943043790 |
| +0.10 | 0.8184221085 | 0.9258970196 | 0.7439803729 | 0.9132356688 |

在 $E_z=+0.003\,\mathrm{V/cm}$ 把 pair window 扩至 $\pm60\,\mathrm{GHz}$ 后，tracked/fixed-$Z$ 与 fixed-laser/fixed-$Z$ 分别为 0.9993174661 与 0.9993173356。

Hyperfine on/off controls at the original pulse-search basis：

| calculation | phase-recalibrated $F_{\mathrm{avg}}$ | static $PP\rightarrow SS$ maximum |
|---|---:|---:|
| full Rb hyperfine | 0.9989603273 | 0.9826807 |
| electronic-only control | 0.9989208385 | 0.9821801 |
| oversized $F_J$-HFS stress test | 0.9989603273 | 0.9826807 |

在 original dipole–dipole primary basis 中，以 stretched $SS$ asymptote 为能量零点，两个 target modes 为

| $E/h$ (MHz) | $SS$ weight | $PP$ weight | other weight |
|---:|---:|---:|---:|
| -18.5924 | 0.554459 | 0.443018 | 0.002523 |
| +12.2861 | 0.441812 | 0.555854 | 0.002334 |

最近的 retained spectator 与 target doublet 相距 $53.9537\,\mathrm{MHz}$，其 $SS/PP$ weight 可忽略，主要成分（0.9951）是 $\mathrm{Rb}\,52D_{5/2},m_J=5/2,m_I=-3/2+\mathrm{Yb}\,D(\nu=50.3),F=3/2,m_F=3/2$。Target overlap 最大的 spectator 位于 $+567.87\,\mathrm{MHz}$，主要由两个 $\mathrm{Rb}\,56P_{3/2}+\mathrm{Yb}\,P(\nu=48.0)$ magnetic products 组成；下一项位于 $-4.905\,\mathrm{GHz}$，主要是 $\mathrm{Rb}\,56S_{1/2}+\mathrm{Yb}\,D(\nu=48.3)$。复核时应检查这些 identity，而不只比较能量排序。

当前 final-reference $B$ scan 在 $0$--$5\,\mathrm{G}$ 每 $0.5\,\mathrm{G}$（另加 $3.10\,\mathrm{G}$）取样，并在 peak 附近加密至 $0.025\,\mathrm{G}$ 及更细。它同时保存 per-field nominal recalibration 和 fixed-$3.10\,\mathrm{G}$ local-$Z$ 两条曲线；后者在 $3.00$--$3.15\,\mathrm{G}$ 细扫范围内为 0.9993000070858--0.9993184594783。Fig. 3(c) 仅展示主扫描，细扫数据保留在数值记录中。两者都使用 local isolated-atom carrier tracking，不是 fixed-laser magnetic-noise predictions。

固定 pulse 的 sampled nominal peak $B=3.085\,\mathrm{G}$ 仅增加 $1.3420\times10^{-7}$，却将 full-19-geometry joint minimum 降至 0.9986901281088。$B=3.1025\,\mathrm{G}$ 的 full-grid nominal / position-only / joint 值分别是 0.9993184076620 / 0.9992029363804 / 0.9986944135247；joint gain 是 $5.9036\times10^{-7}$。针对性 $\pm60\,\mathrm{GHz}$ nominal/axial-vertex check 固定每个 field 的 40-GHz correction，给出：

| $B$ (G) | nominal, 60 GHz | axial vertex minimum, 60 GHz |
|---:|---:|---:|
| 3.10 | 0.9993176931560 | 0.9986891210659 |
| 3.1025 | 0.9993176227652 | 0.9986879855262 |

这使 field-change 的 joint gain 反转为 $-1.1355\times10^{-6}$；即便每个 field 在 60 GHz 重新校准 phases，也分别得到 0.9986942464839 和 0.9986931187781。故保留 $3.10\,\mathrm{G}$。Pair-window sensitivity 不是 statistical error bar；这也不是 joint pulse/field optimization 或 global optimum。完整 checkpoint、运行命令和 resource 用量在 `forster_p0_4_field_scan.json`，由 `scan_forster_p0_4_field.py` 生成；`--assess-only` 和 `--plot-only` 不再构建模型。

---

## 5. vdW static channel setting

### 5.1 Target pair state

$$
|rr\rangle=
|\mathrm{Rb}\,66S_{1/2},m_{\mathrm{Rb}}\rangle
|\mathrm{Yb};S(\nu=62.682292758),F=1/2,m_{\mathrm{Yb}}\rangle.
$$

Yb state identity：

| quantity | value |
|---|---:|
| effective principal quantum number | $\nu=62.682292758$ |
| angular labels | $L=0,F=1/2$ |
| v1.4 selector | $(n,l,s,f)=(67,0,0,1/2)$ |
| energy | $50415.28796\,\mathrm{cm}^{-1}$ |
| $^3P_0\rightarrow S$ wavelength | $301.8699\,\mathrm{nm}$ |

### 5.2 Zero-field $C_6$ setting

| quantity | setting |
|---|---:|
| field | $B=0$ |
| angle | $\theta=0$ |
| magnetic sectors | $m_{\mathrm{Rb}},m_{\mathrm{Yb}}=\pm1/2$ |
| interaction order | dipole–dipole second-order vdW coefficient |

因为 $B=0$ 时 $M=0$ 的两个 bare products 简并，复核时必须在

$$
\{|-1/2,+1/2\rangle,|+1/2,-1/2\rangle\}
$$

中构造完整的 $2\times2$ effective $C_6$ matrix，而不能只分别给两个 diagonal shifts。

### 5.3 Finite-field pair setting

| quantity | setting |
|---|---:|
| magnetic field | $B=25\,\mathrm{G}$ |
| angle | $\theta=0$ |
| addressed product | $(m_{\mathrm{Rb}},m_{\mathrm{Yb}})=(+1/2,+1/2)$ |
| total projection | $M=+1$ |
| distance range | $R=3.0$–$5.0\,\mu\mathrm{m}$ |
| candidate point | $R=3.3\,\mu\mathrm{m}$ |

Pair-basis setting：

| item | setting |
|---|---:|
| Rb principal range | $n=63$–69 |
| Rb orbital range | $l=0$–3 |
| Yb effective-$n$ range | $\nu=59.382292758$–65.982292758 |
| Yb orbital range | $l=0$–3 |
| pair-energy window | $\pm80\,\mathrm{GHz}$ about the same-field dressed asymptote |
| magnetic basis | all individual atomic $m$；只固定 total $M$ |
| interaction | dipole–dipole |

论文中的 pair shift 定义为

$$
U(R,B)=E_{\mathrm{pair}}(R,B)-E_{\mathrm{asym}}(B),
$$

其中 $E_{\mathrm{asym}}(B)$ 是同一 magnetic products 在相同 $B$、infinite separation 下的 field-dressed energy。不能用 zero-field bare energy 作 reference，否则会把 pair Zeeman shift 错算进 $U$。

所研究的 branch 必须从 large $R$ 的 bare $|rr\rangle$ 连续跟踪到 small $R$。branch identity 应同时由 bare-product overlap 和相邻距离的 eigenvector continuity 判断，避免在 near crossing 处换支。

### 5.4 论文声称的 static outputs——待独立验证

Zero-field results：

| quantity | claimed result |
|---|---:|
| stretched $M=\pm1$ | $C_6/h=+73.5794485\,\mathrm{GHz}\,\mu\mathrm{m}^6$ |
| $M=0$ matrix diagonal | $73.607479\,\mathrm{GHz}\,\mu\mathrm{m}^6$ |
| $M=0$ matrix off-diagonal | $0.056060\,\mathrm{GHz}\,\mu\mathrm{m}^6$ |
| $M=0$ eigenvalues | 73.551418 and $73.663539\,\mathrm{GHz}\,\mu\mathrm{m}^6$ |
| nearest dipole-coupled partner detuning | $-315.3248\,\mathrm{MHz}$；论文表格报告其大小 $315\,\mathrm{MHz}$ |

Finite-field addressed branch：

| $R$ ($\mu\mathrm{m}$) | claimed $U/h$ (MHz) | claimed bare-product weight $w$ |
|---:|---:|---:|
| 3.0 | 100.8254 | 0.94364 |
| 3.3 | 57.0721 | 0.96972 |
| 3.5 | 40.1161 | 0.97901 |
| 4.0 | 18.0072 | 0.99089 |
| 5.0 | 4.7199 | 0.99764 |

在 $R=3.3\,\mu\mathrm{m}$，四个 magnetic products 的 claimed values：

| $(m_{\mathrm{Rb}},m_{\mathrm{Yb}})$ | $U/h$ (MHz) | $w$ |
|---|---:|---:|
| $(-1/2,-1/2)$ | 56.79259 | 0.969757 |
| $(-1/2,+1/2)$ | 57.12724 | 0.970173 |
| $(+1/2,-1/2)$ | 56.85998 | 0.970267 |
| $(+1/2,+1/2)$ | 57.07205 | 0.969718 |

还需检查的 claims：

- branch 在 $3.0$–$5.0\,\mu\mathrm{m}$ 内平滑且没有明显 avoided crossing；
- 相对 zero-field $C_6/(hR^6)$ guide 的最大偏差小于 $0.11\,\mathrm{MHz}$；
- 四个 magnetic sectors 的 $U$ spread 约为 $0.6\%$。

$w$ 是 pair-eigenstate composition，不是 gate fidelity；$1-w\approx3\%$ 不能直接称为 $3\%$ gate error。

### 5.5 One-at-a-time basis convergence

收敛检查固定 $B=25\,\mathrm{G}$、$R=3.3\,\mu\mathrm{m}$、$\theta=0$、
$M=+1$、target product、dipole--dipole interaction 和 branch-selection rule，
每行只改变一个数值截断轴。Rb 与 Yb radial range 必须分别改变，不能用同时扩大
两者的结果冒充 one-at-a-time test。有限场求解可利用 Hamiltonian 的 exact
connected components，但包含 target 的 component 必须携带单位 bare-target norm；
下面的 $N$ 仍报告 reduction 前的完整 pair-basis size。

| one-at-a-time setting | $N$ | $C_6/h$ ($\mathrm{GHz}\,\mu\mathrm{m}^6$) | $U/h$ (MHz) | $w$ |
|---|---:|---:|---:|---:|
| reference: Rb $n=66\pm3$, Yb $\nu=62.6823\pm3.3$, $\ell_{\max}=3$, $\pm80\,\mathrm{GHz}$ | 16279 | 73.579448 | 57.072054 | 0.9697184 |
| Rb $n=66\pm4$ | 19317 | 73.579484 | 57.077514 | 0.9697095 |
| Yb $\nu=62.6823\pm4.3$ | 19522 | 73.578727 | 57.071734 | 0.9697186 |
| $\ell_{\max}=4$ | 33180 | 73.579448 | 57.069345 | 0.9697315 |
| pair window $\pm120\,\mathrm{GHz}$ | 21681 | 73.589941 | 57.079907 | 0.9697181 |

相对 reference，四个 expanded rows 的最大绝对变化为
$0.0104924\,\mathrm{GHz}\,\mu\mathrm{m}^6$、$0.0078535\,\mathrm{MHz}$ 和
$1.3122\times10^{-5}$，对应 $C_6/h$、$U/h$ 和 $w$ 的最大相对变化
$1.4260\times10^{-4}$、$1.3761\times10^{-4}$ 和 $1.3532\times10^{-5}$。
另有 Rb radial、Yb radial、$\ell_{\max}$ 和 pair-window 的 contracted rows；
full-precision records 在 `vdw_p1_5_basis_convergence.json`，由
`evaluate_vdw_p1_5.py` 生成。

这项结果只支持所列 static quantities 的 one-axis truncation stability。它没有
测试多个轴同时扩展产生的 cross-terms、dipole--dipole 以外的 multipoles、
driven vdW dynamics 或 process fidelity。

### 5.6 Optical access setting

Rb：

$$
|5S_{1/2},F=2,m_F=+2\rangle
\xrightarrow{780+480\,\mathrm{nm}}
|66S_{1/2},m_J=+1/2\rangle.
$$

Yb：

$$
|{}^3P_0,F=1/2,m_F=+1/2\rangle
\xrightarrow[\pi\ \mathrm{polarization}]{301.87\,\mathrm{nm}}
|S(\nu=62.6823),F=1/2,m_F=+1/2\rangle.
$$

论文使用的 optical scale：

- Yb dipole：$6.27\times10^{-4}\,ea_0$；
- $12\,\mu\mathrm{m}$ waist 下达到 $\Omega/2\pi=2\,\mathrm{MHz}$：约 $18.7\,\mathrm{mW}$。

这里应称为 **selection-rule-allowed candidate excitation route**：它只说明 selection rule 与 wavelength/intensity scale 合理，不表示 exact-line state selectivity 已被实验确认。

---

## 6. 不属于当前模型的效应

### 6.1 Förster gate 未包含

- finite-temperature blackbody redistribution；
- gate 期间的 dynamical atomic motion 和 thermal ensemble；
- trace-preserving Lindblad jumps 与 state-resolved branching；
- Rb $5P_{3/2}$ intermediate scattering 和 AC Stark shifts；
- measured AOM/RF impulse response 与 amplitude–detuning cross-coupling；
- laser phase noise、额外 frequency noise 和 timing jitter；
- single fixed phase calibration 下的 magnetic-field noise；
- arbitrary-direction/transverse stray electric field；
- magnetic-state preparation error；
- polarization impurity 的完整 driven simulation；
- spectator atoms 与 many-body crosstalk。

另外，retained Rb $F_J$ spectator states 的真实 Rydberg HFS normalization 仍未知；central calculation 令其为零，并只用第 3.3 节的 oversized assignment 做 sensitivity test。Yb fit 也没有可用 covariance matrix，所以第 4.8 节只给 deterministic energy envelope，没有传播 wave-function/matrix-element covariance。这些 tests 都不应被解释为 rigorous uncertainty bound。

### 6.2 vdW channel / future gate 未包含

- full laser-driven multichannel gate dynamics；
- optical coupling 到各 dressed pair components 的 complex amplitudes；
- exact-state resolved decay branching 和 blackbody transfer；
- Rb/Yb laser technical errors、intermediate scattering 和 AC Stark totals；
- position/motional ensemble；
- exact-line selectivity、polarization impurity 和 magnetic-state preparation error。

因此，$99.9318\%$ nominal 与 $99.8694\%$ sampled minimum 是 Förster **specified numerical-reference model 内**的 loss-aware overlaps，不是实验 fidelity prediction；vdW 的 $57.1\,\mathrm{MHz}$ 是经 one-at-a-time basis checks 的 finite-basis static pair shift，不是 gate fidelity。

---

## 7. 希望复核同学提交的结果

### 7.1 Förster channel

- 独立得到的 atomic energies、defect 和 coupling；
- 两个主要 eigenstates 的 $SS$、$PP$、spectator weights；
- $SS\rightarrow PP$ transfer curve；
- fixed-electronic-$m$ 与 hyperfine-resolved all-$(m_J,m_I)$ 的差别；
- basis/window convergence；
- 对“two-level-dominated Förster channel”是否成立的判断。

### 7.2 Förster CZ

- 独立构造的 driven Hamiltonian 和四个 computational inputs 的 return amplitudes；
- $K$、local phases、conditional phase、survival、coherent error 和 $F_{\mathrm{avg}}$；
- nominal、position-only 和 position+Rabi-error 结果；
- nested 128/256/512/1024 Sobol trace、local adversarial candidates、
  direct retained-block rechecks 和 response-surface holdout error；
- thermal literature inputs 与 cross-apparatus phase-space scale benchmark，
  并说明为何它不能当作 same-apparatus thermal gate fidelity；
- actual-point target eigenspectrum、nearest/top spectators，以及 unshown spectator population；
- hyperfine on/off 和 unknown-$F_J$ stress-test 结果；
- time-step、bright-mode cutoff、pair-window、$\Delta n$、$\ell_{\max}$、atomic-window 和 multipole-order convergence；
- nominal 与 limiting point 的 all-3684-mode projection audit；
- 四个 spectroscopy-motivated vertices、pure defect scan 和 fixed-laser axial electric-field scan；
- 对给定 pulse 是否 local-$Z$ equivalent to CZ 的判断；
- 对 no-jump/orthogonal-leakage assumption 是否会高估或低估实际性能的讨论。

### 7.3 vdW channel

- zero-field $C_6$ 和 $M=0$ effective matrix；
- finite-field $U(R)$、bare-product weight 和 branch-continuity check；
- 四个 magnetic sectors 的比较；
- basis/window/eigenpair convergence；
- 对该 channel 是否适合作为 blockade interaction resource 的判断。

最终报告请分别给出 **PASS / FAIL / INCONCLUSIVE**，并严格区分：

1. pair channel 是否存在；
2. 给定模型内的 gate calculation 是否正确；
3. 数值结果是否收敛；
4. 结论能否外推到实验。
