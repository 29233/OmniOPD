# 类别校准的多 Teacher on-policy 蒸馏框架

状态：可讨论、可实现的方法设计；尚未修改训练实现或启动训练。本文的多Teacher路由、类别校准和分割质量路由是新设计，不能作为附件S-OPD已有结论引用。

**OPSD口径修订：用户明确本项目以Student自身的无特权策略作为削弱分支。主方案以 [多特权视图OPSD修订](MULTI_PRIVILEGE_OPSD_DESIGN.md) 为准。本文此前的独立Teacher、每Teacher单独去特权评分及T0回退属于另一组OPD/归因消融，不是当前自蒸馏主版。**

## 1. 目标与训练假设

Student部署输入x=(V,A,r)：视频、音频、指代表达。训练额外可用特权标注z=(c*,M*)：目标类别及目标实例的逐帧mask。先使用已有标注，不假定存在逐词CoT正确性标签、声音源逐事件时间戳或人工关系标签。

输出y=(CoT,final category,[SEG])。沿用当前自然语言目标断言，训练日志中解析最后一个明确肯定的目标类别；如规范成固定`Target: <canonical category>. It is [SEG].`，必须先做格式适配，并单独消融格式改变。未知、否定、冲突的结论不计为正确。类别正确不保证实例或CoT正确。

目标：普通输入下同时增强音视频指代推理、同类实例定位、mask分割。训练期由各Teacher的特权优势提供监督，部署时只保留Student。

本方法称MOPD只是本项目多Teacher OPD的工作名称，本文不主张复现某篇同名论文。

## 2. Teacher设计

| Teacher | 完整视图x_k+及训练期特权 | 介入视图x_k- | 拟蒸馏能力 | 接入前条件 |
| --- | --- | --- | --- | --- |
| T0 基础推理Teacher | 原V、A、r，无标注提示 | 无需介入；回退监督 | 通用语言、音视频基础推理、输出协议 | 验证普通输入能力，不以规模代替可靠性 |
| TR 语义Teacher | 原V、A、r，附`Target category: c*`，不给GT CoT | 同模板`Target category: unknown`；媒体、占位结构保持一致 | 目标类别与声音、行为、指代关系的推理，错误类别纠正 | 用类别特权进行训练/适配；检查矛盾率，不把类别照抄视为推理能力证明 |
| TG 空间分割Teacher | 原V、A、r，额外原首帧RGB叠加GT目标轮廓/框或单独二值mask；不裁剪 | 相同画布/预算的普通首帧RGB或与前景无关的匹配占位图；同一通用任务说明 | 原视频坐标中的目标实例绑定、SEG空间表示及mask | 必须先做提示适配及分割训练；验证同类实例与共同类别正确子集分割能力 |

TG图片只能使用GT mask派生，原图与SAM始终保持原坐标。首帧mask Teacher与全帧mask Teacher分别报告，主版用首帧以评估跨帧推广。空首帧不选择更晚的非空帧，标记为无有效空间特权。

若TG不带语言模型，仅是SAM等分割专家，它只能进入mask路由，不提供词表token目标。采用独立LLM+分割头的TG可同时进入文本及mask路由。

Teacher可以采用相同tokenizer家族的不同大小或不同能力adapter。首版优先同tokenizer并验证completion IDs完全对齐。不同模型Teacher与同模型多视图Teacher需分开消融。当前未经适配的SFT+特权图片不能直接假定是空间专家：已观察到类别相同但mask退化。

### 2.1 正负视图的具体配置与音频专家

配对原则：对每个Teacher，保留相同原视频帧、时序、指代和Student前缀；只改变预先定义的证据。这里的“削弱视图”不是统一删除全部视频和音频。

**语义TR**：正视图`(V,A_s,r,"Target category: c*")`，负视图`(V,A_s,r,"Target category: unavailable")`。二者用同一任务指令，不变更视频/音频。测量类别特权的贡献，不是视觉/音频理解的贡献。类别提示长度与含义无法做到完全一致，必须记录prompt和completion边界，不声称无prompt差异。该Teacher适合语义纠错，不能证明末尾类别照抄以外的推理收益。

**空间TG主版**：正视图`(V,A_s,r,I_marked)`，负视图`(V,A_s,r,I_original)`。I_marked为首帧原RGB叠加GT外轮廓（或另一个独立实验中的GT框）；I_original为同一首帧RGB、无标记。两边固定同一说明：`The auxiliary image is the first video frame. When a target annotation is present, use it as a cue and locate the corresponding instance in the original video.` 不在负视图继续宣称存在GT标记。该介入移除实例标记，同时保留纹理、位置、整体画布及编码预算。原二值mask正/负视图改变量更大，放在单独消融，不与RGB轮廓混称一个介入。

**音频TA主版（先实现）**：训练期按一定比例对Student音频加可控噪声或短时遮挡，A_s=degrade(A)；其余样本A_s=A。TA正视图`(V,A,r)`，负视图`(V,A_s,r)`，同一prompt、时长、采样率、声道和Student prefix。只有Student受到退化的样本才存在clean-audio特权；当A_s=A时该特权差值为0，不伪造Teacher优势。TR/TG都使用与Student相同的A_s，保证其路由差值仍分别来自类别与空间特权。TA为音频能力已验证更强的Omni模型/adapter，目标是事件识别、声音持续/先后、语音内容、音视频发声源匹配和噪声鲁棒性；帧内左右实例定位仍需视觉与分割监督。

音频TA不注入GT目标类别或GT mask，不将目标类别改写为“The audio contains c*”。TA独立类别核验使用`(V,A_s,r)`；GT只用于训练门控/验证。其他Teacher也在同一部署视图上独立核验，避免将其清晰音频优势混入基础可靠性比较。

**音频依赖单独检测**：当研究普通清晰音频时，另计算`(V,A,r)`与`(V,silence(A),r)`的Teacher差值，表示audio dependence。两边时长与媒体结构相同。它可作为音频相关token gate，但不能称为对Student的特权输入，且静音干预可能产生分布外反应。clean-vs-degraded与audio-vs-silence使用不同名称、尺度和消融。

**可选TA声学证据特权（普通输入下仍有特权）**：仅输入音频的独立专家提取经过审阅的事件证据E_A，再由可评估Student前缀的Omni Teacher接收`(V,A,r,E_A)`；负视图为`(V,A,r,E_unavailable)`，固定声学证据说明。E_A可以包括可靠的事件时间区间、音色/节律、持续/间歇、多声源、语音转录以及置信度；不包含由视频/GT推出的目标类别、左右位置或最终答案。时间标注和声音身份属于待生成/验证的伪标注，当前数据不假定已有GT；ASR只用于语音片段，不能替代音乐、动物和机械声音的事件理解。完整/削弱两边都保留原audio，测量的是额外声学事实贡献。此版需要新增证据生产与质检，不能把伪标注当可靠GT。

TA可靠性不能只由最终目标类别决定：目标可能靠视频猜中。增加音频事件query、原audio/错配audio对照、时间关系题和降噪验证；用独立验证集声学准确率校准a_A，对事件伪标注或去噪伪影不可靠的样本降权。Student错误audio描述即使最终类别正确，也允许TA提供纠错。无需新增语义规则把所有“sound”等词强制路由到TA，使用可靠性校准的双向证据差值作为主路由。

新增TA后，文本argmax扩大到TR、TG、TA及T0回退；原类别R_k、token D_k,t、权重和损失表达不变。不同介入强度必须校准；TA没有经验证的AURORA分割头时不进入mask route，即使它对SEG符号概率很高。

## 3. 一条Student轨迹，多个Teacher评分

每次迭代冻结采样快照theta_bar：

\[
y\sim p_{\bar\theta}(\cdot\mid x),\quad
p_t=p_\theta(\cdot\mid x,y_{<t}),\quad
q^+_{k,t}=p_{T_k}(\cdot\mid x_k^+,y_{<t}),\quad
q^-_{k,t}=p_{T_k}(\cdot\mid x_k^-,y_{<t}).
\]

Student只在原输入上自由采样一次。所有Teacher针对同一y并行评分，各正/负视图通过完整序列forward计算所有位置；不独立生成Teacher CoT后按token位置拼接。Teacher/RPC返回精确的Student采样token logp，不能以top-k遗漏代替。

最终类别可用于整条已完成轨迹的后验权重；该门控无需因果，因为只在训练loss中使用，但Teacher的每个next-token目标必须严格只看到y_<t，不能看到未来类别或未来CoT。类别门控、argmax和权重停止梯度。

## 4. 类别可靠性：三个不同的量

### 4.1 Student结论与GT

解析Student末尾结论hat c_S；m_S=1[hat c_S=c*且结论无否定/冲突]。未知记m_S=0，并单独日志。hat c_S本身不是GT。

### 4.2 Teacher独立核验分布

Teacher另做一个固定分类query：给原V、A、r，但不给GT类别、Student类别及Student CoT，只要求返回规范类别。无需自由生成长CoT；可通过规范类别候选字符串的条件序列概率计算分布。

\[
v_k(c\mid x)=\frac{\exp(\ell_k(c\mid x)/T_{\rm cal,k})}
{\sum_{c'\in\mathcal C\cup\{unknown\}}\exp(\ell_k(c'\mid x)/T_{\rm cal,k})}.
\]

ell_k是query下规范类别字符串加明确结束边界的联合log概率，候选类别统一，包含unknown；多token类别不是只取首token或简单不同词表token softmax。校准温度在独立验证集确定。若采用长度归一化，必须称为归一化分类score而非原始字符串概率。

hat c_k^verify=argmax_c v_k(c|x)。这提供独立的语义可靠性代理，不证明整条CoT或实例正确。

TR有GT类别时再从x_R+分类会产生标签照抄，不能用它当独立可靠性实证。分类query必须来自无特权视图。

定义：

\[
R_k=a_k[\epsilon_R+(1-\epsilon_R)v_k(c^*\mid x)].
\]

a_k是验证集确定的Teacher能力可靠性，epsilon_R>0保留能利用特权纠错的Teacher，避免无特权时答错就绝对禁用。另仅以a_k路由作为消融；当前样本分类信心也可能未校准，不保证正确。

### 4.3 同意与纠错

\[
A_k=1[\hat c_k^{verify}=\hat c_S],\quad
F_k=1[\hat c_S\ne c^*\ \land\ \hat c_k^{verify}=c^*].
\]

A_k用于日志，不作为硬门控；两个模型可共同判断错误。F_k标记可纠错Teacher。Student正确时优先保持，Student错误时允许与Student不同的Teacher进入。

## 5. token证据信号和路由

### 5.1 与S-OPD的联系

\[
\Delta_{k,t}=\log q^+_{k,t}(y_t)-\log q^-_{k,t}(y_t).
\]

附件S-OPD第6页式(4)使用原图/遮挡图的正delta门控，式(5)加权Student跨视图的对比损失。本文TR的类别特权正/负视图、TG的空间特权正/负视图是特权依赖，不等同于原文视觉依赖；单独保留原图/遮挡图版本作复现或消融。

如果Student token错了，有证据Teacher可能降低其概率，delta<0。主版使用双向幅值，保留符号：

\[
D_{k,t}=\min\left(d_{max},\frac{|\Delta_{k,t}|}{s_k+\epsilon}\right),
\quad g_{k,t}=I_{k,t}\,1[D_{k,t}>\tau_D]\,1[R_k>\tau_R].
\]

I表示严格token/前缀对齐、finite logp和有效介入；TR unknown占位仍有语言语义影响，TG画布对照仍有解释因素，须审计并消融。s_k取验证集冻结的|delta|稳健尺度，消除模型尖锐度/介入尺度部分差异。它仍不能将敏感度自动转成质量。正delta版本作为论文相近基线。

### 5.2 类别纠错和token阶段

令B_cat为Student最终类别词的token跨度，不包含SEG。明确类别错时提高该跨度监督：

\[
b_t=1+\gamma_c(1-m_S)1[t\in B_{cat}],\quad
S_{k,t}=g_{k,t}R_kD_{k,t}\big[1+\gamma_fF_k1[t\in B_{cat}]\big].
\]

主版不凭类别错就放大整段CoT，因正确的中间步骤可能仍有用。若没有可解析类别跨度，b_t=1，使用独立类别辅助损失恢复语义监督，不按GT词在任意文本中的出现位置补配对。

如后续能够可靠标注音频、视觉、关系推理token角色，可扩展到角色先验，但首版不假设现成token角色GT，不以句子关键词充当准确的能力标签。

### 5.3 硬路由和强度

有候选通过门控时k_t*=argmax_k S_k,t；若无候选，则回退T0。若T0不可用/已知不可靠，则该token蒸馏权重设0，保留GT类别和mask辅助监督；不随意指定一个无效Teacher。

\[
Q_t=q^+_{k_t^*,t},\quad
u_t=b_t\,[w_0+\min(S_{k_t^*,t},s_{max})],\quad
w_t=\operatorname{sg}\left(\frac{u_t}{|\mathcal B|^{-1}\sum_{j\in\mathcal B}u_j}\right).
\]

回退S=0。B是有效文本token集合，包含EOS/类别词；w0>0防止通用token失去监督。训练长度很短或B为空需显式处理。归一化保持每条有效轨迹mean weight=1；b_t只分配相对权重，不改变总text loss系数。最终类别是全轨迹监督，只有prefix next-token目标是因果。

各token可选择不同Teacher。但硬路由需要先计算所有候选Teacher分数，不能宣称省去Teacher forward。监控route switch、单Teacher塌缩；首版不强制均匀负载，因为优势Teacher本来可能占多数。

## 6. 路由的推导：从稀疏分配到argmax

固定teacher scores后，考虑每个token的分配alpha_t位于K维概率单纯形：

\[
\max_{\alpha_t\in\Delta_K}\sum_k\alpha_{k,t}S_{k,t}+\eta H(\alpha_t).
\]

拉格朗日函数对alpha求导：

\[
S_k-\eta(1+\log\alpha_k)+\lambda=0
\Rightarrow\alpha_{k,t}=\frac{e^{S_{k,t}/\eta}}{\sum_{j\in eligible}e^{S_{j,t}/\eta}}.
\]

eta->0时分配集中于最大score，即硬路由。平局按固定Teacher顺序处理。该推导说明给定代理score后的选择规则；并不证明score最大Teacher在真实任务上最好。R、D的可靠性需实验证实。

## 7. on-policy文本蒸馏损失及梯度

为与附件方向一致，主定义为reverse KL：

\[
\mathcal L_{MOPD}(\theta;\bar\theta)=
\mathbb E_{(x,z)\sim\mathcal D,y\sim p_{\bar\theta}(\cdot|x)}
\left[\frac1{|\mathcal B|}\sum_{t\in\mathcal B}w_t
D_{KL}(p_\theta(\cdot|x,y_{<t})\|Q_t)\right].
\]

theta_bar、Student前缀、门控、Teacher分布均固定，只对当前Student分布求导。这是Student当前采样状态上的局部蒸馏surrogate，不宣称包含整个轨迹分布随theta改变的全部梯度。每次更新后重新采样；多次重用旧rollout时需限制陈旧程度或增加明确的off-policy修正。

对于固定context，令p(a)=softmax(z)_a，Q(a)>0：

\[
K=\sum_a p(a)[\log p(a)-\log Q(a)].
\]

由softmax导数dp_a/dz_b=p_a(1[a=b]-p_b)：

\[
\frac{\partial K}{\partial z_b}
=p(b)\{\log[p(b)/Q(b)]-K\}.
\]

高于Teacher相对比例的token被压低，低于Teacher比例的候选被提高；Teacher可以纠正Student实际采样的错误，而不只赞同该采样token。

等价score-function表达：

\[
\nabla K=\mathbb E_{a\sim p}
[(1+\log p(a)-\log Q(a))\nabla\log p(a)].
\]

因E_p[grad logp]=0，常数1可作为baseline去掉。若用Student采样token估计，则需要这种梯度权重或正确OPD实现；直接对固定token平均(logp_S-logp_T)反传不是上述reverse KL梯度。推荐首版全词表逐位置KL以验证数学，再采用明确的top-k+tail近似，避免稀疏Teacher中未出现项设0导致reverse KL发散。

### 7.1 软组合与硬路由的关系

固定alpha时：

\[
\sum_k\alpha_k KL(p\|q_k)=KL(p\|q_{geo})-\log Z,
\quad q_{geo}(a)=\frac{\prod_kq_k(a)^{\alpha_k}}{Z}.
\]

所以加权reverse-KL对应几何组合Teacher，不是简单概率平均。alpha为one-hot时q_geo=q_selected，Z=1，得到硬路由目标。

若采用项目当前的forward KL：sum alpha KL(q_k||p)与KL(sum alpha q_k||p)在Student梯度上等价。必须报告loss方向；主版reverse-KL和现有forward-KL作为消融，不能混为同一个实现。

## 8. 显式类别辅助损失

token路由间接提高类别词监督，但已有错误类别字符串的后续prefix可能使正确类别无法在后续token局部修复。补充在最终类别字段起点之前的prefix h_cls，构造Student规范类别分布v_S(c|x,h_cls)，该prefix不包含Student已生成类别或SEG。

Teacher以完整特权视图和相同h_cls输出分类分布v_k^cls；Teacher独立核验分布v_k^verify不含Student CoT，二者用途不同。类别路由按可靠性及v_k^cls(c*)选择Teacher；TR有c*时此分布可能是oracle，必须如此标注。

\[
k_{cls}^*=\arg\max_{k\in eligible}R_kv_k^{cls}(c^*),\quad
\mathcal L_{cls}=KL(\operatorname{sg}(v_{k_{cls}^*}^{cls})\|v_S)
\quad\text{或}\quad -\log v_S(c^*).
\]

两者首版择一；oracle one-hot时forward KL=-log v_S(c*)，相当于直接GT类别监督而非额外Teacher知识。必须匹配GT分类辅助loss的baseline。对错误Student CoT给正确最终类别监督不等于认定整条CoT正确；前述token Teacher分布仍独立负责CoT纠错。

无字段/无有效类别断言时，固定格式分类query单独计算v_S并监督，不把改变后的GT完整CoT当Student rollout。

## 9. SEG的独立质量路由与mask损失

文本概率p([SEG])只能监督符号出现，不能监督其连续表示。文本B可以包含SEG符号的next-token协议监督，但能力分析与mask路由必须独立；语言token只选中TG不保证mask更好。

Student同一rollout上的表示：e_S=P_S(h_SEG)，mask概率m_S=sigmoid(D_S(F_S(V),e_S))。Teacher也在同一Student回答上replay并得到mask m_k。分割Teacher可为不同架构，只要求概率图对应相同原图坐标。

\[
J_k=mean_f IoU(1[m_k^{(f)}>0.5],M^{*(f)}),
\quad J_S=mean_f IoU(1[m_S^{(f)}>0.5],M^{*(f)}).
\]

gate使用stopgrad m_S。对具备LLM的Teacher，带GT类别时的类别一致不能验证实例质量；因此mask gate以J为主、类别只作附加的语义合法性/矛盾检查。纯分割Teacher不强行附加不存在的类别输出。

\[
g_k^{seg}=valid_k\,1[J_k>\tau_J]\,1[J_k-J_S>\mu],
\quad k_{seg}^*=\arg\max_{k:g_k^{seg}=1}J_k,
\quad w_{seg}=sg(clip(J_{k_{seg}^*}-J_S-\mu,0,1)).
\]

mu>=0是Teacher优势margin。没有Teacher优于Student时mask KD=0，但GT mask监督仍存在。Student类别错时不关闭GT mask，也不禁止准确Teacher mask来纠正。只要求可解码SEG；SEG缺失则maskKD不适用，保留文本协议/类别监督并记录失败率。

对齐mask概率空间：

\[
\mathcal L_{maskKD}=w_{seg}\frac1{FHW}\sum_{f,u}
KL(Ber(m_{k_{seg}^*}(f,u))\|Ber(m_S(f,u))).
\]

Bernoulli forward KL展开为t log(t/s)+(1-t)log[(1-t)/(1-s)]。Teacher固定时忽略其熵，等价于soft-target BCE：-t log s-(1-t)log(1-s)。对Student像素logit梯度为s-t，直接推动Teacher与Student的同一原坐标前景概率接近。

当前项目GT分割项可保持：

\[
\mathcal L_{GT}=2BCE(m_S,M^*)+0.5DiceLoss(m_S,M^*).
\]

mask/类别质量路由使用GT，是训练期oracle路由，不能称为无标注。它选择准确的软目标或更合理的结构表示，不保证比直接GT监督有额外收益；需GT-only、同训练量和同损失预算对照。

### 9.1 可选embedding蒸馏

不同Teacher投影/decoder空间不可直接MSE。若所有Teacher经过验证共享同一固定分割decoder和表示坐标，则可加w_seg*||e_S-sg(e_T)||^2，或归一化cosine loss。任意训练映射A_k需要另行学习并防止表示坍缩；主版不添加，以maskKD为可比输出监督。

## 10. 可选S-OPD学生感知项

先验证多Teacher门控本身，再加入TPC/PA。

用原V/共享patch遮挡V的Teacher差值delta_k,t^vis，选择可靠Teacher的正差值d_t^vis；不能直接把TR有无类别提示差值当视觉依赖。

\[
g_t^{vis}=1[d_t^{vis}>\tau_{vis}],\quad
\mathcal L_{TPC}=-\frac1{\max(1,\sum_tg_t^{vis})}
\sum_tg_t^{vis}d_t^{vis}KL(p_t\|sg(p_t^{masked})),
\quad
\mathcal L_{PA}=\frac1T\sum_tKL(p_t\|sg(p_t^{noisy})).
\]

mask perturbation和noise保留视频时序、维度、token预算，noise需保留任务内容。可选clip/hinge上限或annealing防止TPC无界最大化；这属于稳定化扩展，和论文原目标分开消融。mask空间的轻微噪声一致性可以另加，但不是语言KL的替代。

## 11. 完整目标与迭代算法

\[
\boxed{\mathcal L=
\lambda_{text}\mathcal L_{MOPD}
+\lambda_{cls}\mathcal L_{cls}
+\lambda_{GT}\mathcal L_{GT}
+\lambda_{seg}\mathcal L_{maskKD}
+\lambda_{TPC}\mathcal L_{TPC}
+\lambda_{PA}\mathcal L_{PA}}.
\]

主版先lambda_TPC=lambda_PA=0，保留两路质量监督。所有Teacher、类核验、GT分数、gate、routes、weights无梯度；Student LLM、投影头、允许训练的分割参数收到loss梯度。语言和mask梯度共享Student参数，仍可能冲突，需分支梯度余弦日志；不预设不同Teacher贡献天然可相加。

每步：

1. Student原输入rollout；解析结论类别、类别跨度、SEG协议及冲突。
2. Teacher并行正/负视图评分同一rollout，独立无特权query核验类别。
3. Student带梯度replay同一rollout，取得token distributions、SEG representation与mask。
4. 计算类别可靠性、signed delta、text routes、归一化weights。
5. 解码Teacher mask，GT质量评分，选择mask Teacher。
6. 构造正确类别query和辅助loss；可选Student masked/noisy views。
7. 汇总loss并更新；下一步重新采样Student轨迹。

跨Teacher的完整token分布需共同vocabulary和准确prefix对齐。首版验证全分布数学，正式运行采用精确sampled logp+top-k/tail目标时报告近似及覆盖率。采用不同tokenizer时重新设计span级蒸馏，不用SequenceMatcher等词片段视为严格prefix对应。

## 12. 实证验证与研究表述

必须验证各Teacher在独立验证集具有拟声称优势，尤其TG与TR在共同类别正确集合上是否有不同分割表现。特权提示使Teacher输出变化不等于Teacher更强。

必要消融：

- 单Teacher TR/TG/T0，等权多Teacher，token硬路由。
- 无类别门控、只看Student-Teacher agreement、GT校准类别门控；测试错误类别rollout的纠错率。
- 正delta、abs delta、门控不加权、门控加权，可靠性/尺度校准。
- 无maskKD、只GT mask、质量路由maskKD；GT分类loss相同的对照。
- 无TPC/PA、分别加入、联合；无特权Teacher、不同Teacher模型和同模型多视图。

固定数据/步数/rollout数量和权重预算，报告额外Teacher计算成本。用val选参数，test_s不调参；后续test_u/novel category检验泛化与特权类别依赖。报告类别、否定/矛盾、同类实例错误、共同正确固定集合IoU、第2-10帧IoU、SEG成功率及噪声鲁棒性。

日志：每Teacher独立类别分布与结论、是否GT类别已在prompt、Student末尾类别、m_S/A_k/F_k、每token delta符号/幅值/gate/route/weight、mask J_teacher/J_student/route、fallback、Teacher占比与route switches。

可验证的论文主张应是：类别校准选择更可靠的token监督，空间质量路由将语义和实例能力连接起来；多Teacher在Student自己的轨迹上产生互补收益。不能仅凭路由数量或embedding距离宣称学到了不同能力。

## 13. 数学与实现边界

- 算法所有GT使用限于train；类别独立核验不含GT但门控用GT监督，明示有标注。
- 末尾类别仅作训练后验权重，不泄露给Student部署输入；权重依赖future tokens需停止梯度，不当作causal路由网络。
- 基础RKL推导是固定Student采样context的局部目标；若追求全轨迹优化，需要另推导包含状态分布导数的policy-gradient目标。
- Teacher评分和重用Student回答生成mask属于replay监督，并非Teacher独立采样。多个Teacher并行评分不会消除模型内存和forward成本。
- 没有对独立Teacher的可靠性或新路由效果给出结果保证；本设计需要实现与消融。
