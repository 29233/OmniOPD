# 多特权视图 OPSD：以 Student 为共同基线

本修订优先于先前MOPD方案中的独立Teacher正负视图与T0回退设计。状态为设计，没有更改训练代码。

## 1. 定义

纯OPSD主版中的Teacher不是独立强模型，而是同一Student参数快照在不同特权输入上的无梯度分支：

\[
y\sim p_{\bar\theta}(\cdot|x_S),\quad
p_t^0=sg[p_{\bar\theta}(\cdot|x_S,y_{<t})],\quad
q_{k,t}=sg[p_{\bar\theta}(\cdot|x_k^+,y_{<t})].
\]

待更新Student为p_theta(.|x_S,y_<t)。在一次采样和更新阶段，theta_bar对应同一个完整模型状态，包括同一个adapter、projector、decoder。stopgrad只控制梯度，不应通过disable_adapter()把特权分支切成另一套参数。

共享的削弱视图就是Student实际采样输入x_S，不人工增加unknown类别、空白图片、静音、裁剪或视频遮挡。假如Student本来接受增强/退化输入，x_S就是这一实际输入；不再叠加第二次退化。

## 2. 三个完整视图

| 分支 | 特权视图x_k+ | 共同基线x_S | 能力目标 |
| --- | --- | --- | --- |
| TR | (V,A_s,r,GT target category c*) | (V,A_s,r) | 语义判断、类别纠错及相关指代推理 |
| TG | (V,A_s,r,first-frame GT target mask/marked original RGB) | (V,A_s,r) | 实例定位、SEG空间表示与mask |
| TA clean-audio | (V,A_original,r) | (V,A_s,r)，A_s为Student当前退化audio | 音频证据恢复、事件/时间理解、噪声鲁棒性 |
| TA acoustic-evidence | (V,A_s,r,E_A) | (V,A_s,r) | 利用可靠声学事件/时间事实；E_A需独立审阅 |

TA后两项是可选配置，不同时强制使用。若A_s=A_original、也不提供E_A，则TA与基线完全相同；无输入不对称就没有该分支的自蒸馏优势。音频证据不得由GT target label杜撰；ASR只覆盖speech。清晰音频特权训练混入普通音频样本，避免把整项任务无意改成仅抗噪评测。

模型所有分支同权时，独立无特权类别query在各分支结果相同，不能据此得到Teacher特异可靠性。它只能衡量Student当前类别能力。

## 3. 信号、类别与mask质量

\[
\Delta^{priv}_{k,t}=\log q_{k,t}(y_t)-\log p_t^0(y_t).
\]

差值表示特权视图相对普通Student的支持变化，包含附加prompt/media结构影响，不等于经过严格控制的纯特权内容效应。首版按视图验证集尺度校准，监控Student低概率导致的极大值。

正delta可用于“支持”门控，abs(delta)可保留“纠错”位置，但两者都不保证Teacher质量。Student错误token可能得到负delta；此时仍以q完整分布为目标，不能提高原错token。分数只在采样Student token上度量，可能遗漏Teacher对其他候选的变化，可用完整分布JS(q,p0)做消融。

对类别，hat c_S为rollout最后结论，和GT对照。Teacher类别判别query必须在未包含Student类别的prefix h_cls下计算：v_k(c|x_k+,h_cls)，基线v0(c|x_S,h_cls)。可定义特权类别优势：

\[
A_k^{cls}=v_k(c^*)-v_0(c^*).
\]

此优势可校准类别相关位置路由。TR已拿到GT类别，其类别优势可能是标签照抄；这可提供训练监督，但不能证明整条CoT、音频理解或实例定位更正确。不能从Teacher replay Student的完整回答中读出“Teacher自己的预测”，也不能只按Teacher与Student一致门控。

各分支整体可靠性a_k应来自独立验证的特权收益/矛盾/能力评测；在同权OPSD里，它表示输入视图的经验可靠性，不是不同模型的可靠性。TG需额外以同一Student轨迹上的GT mask IoU优势J_k-J_S做质量门控。TA可靠性需音频事件/时间与噪声指标，不能仅凭最终类别。未知/矛盾类别单独处理。

路由候选score可以用g_k,t*a_k*clip(|delta|/s_k)并在错误最终类别跨度加入非负类别纠错优势，所有门控/尺度/优势stopgrad。类别质量因子不应无差别否决整条CoT中有价值的局部纠错；mask route独立。

## 4. 路由和损失

\[
k_t^*=\arg\max_{k\in eligible} S_{k,t},\quad
\mathcal L_{text}=\frac1{|B|}\sum_{t\in B}w_t
KL(p_\theta(\cdot|x_S,y_{<t})\|q_{k_t^*,t}).
\]

B为有效候选token集合，w在B内归一化并停止梯度；B为空时该loss为带图零值，继续GT类别/分割监督。主版不再设独立T0。当所有特权分支无可靠优势时可跳过KD；用p0作回退，在theta=theta_bar时KL和梯度都为0，不能声称产生纠错。若对旧快照/EMA做额外约束，则属于时间自蒸馏，明确为另一项可选regularizer。

同一模型下，尽管初始参数相同，只要q_k因特权输入不同于p，KL可产生非零Student梯度；Teacher必须detach，否则反向修改特权目标可能产生不同优化问题。RKL局部梯度与旧文档推导不变。当前项目forward-KL/JSD可以先作为基线，loss方向另做消融。

mask loss以Student rollout后的SEG解码为条件，选择Teacher提示质量更好的分支，蒸馏mask概率并保留GT mask。相同projector/decoder快照使SEG feature更可比，但仍需GT质量门控，不能仅以embedding变化大认定有益。

## 5. 额外对照仅用于归因

可选计算TG同预算普通RGB、TR matched unavailable字段、TA静音对照，以拆解prompt/image增加和实际标注的效果；这些是控制实验，不替代主版Student基线。例如：

\[
\log q_G^{marked}-\log p^0
=(\log q_G^{marked}-\log q_G^{RGB})
+(\log q_G^{RGB}-\log p^0).
\]

前项更接近目标标记贡献，后项为额外RGB和说明结构影响。原先实验中额外RGB本身有收益，这个消融很重要。遮挡Student视觉分支只在加入S-OPD TPC时需要，不是OPSD最小必需项。

## 6. 项目实现审计

当前scripts/train_opsd_swift.sh对相同model路径建立Teacher alias，明确避免ms-swift隐式self-distillation关闭adapter，以便加载SFT Teacher adapter。那是既有冻结Teacher实现；若新方案要求每步Teacher与当前Student同权，不能只凭“同基座/同初始化”称为已满足。

后续实现需明确选择：当前参数detach分支（本主版）、冻结初始SFT Teacher、或EMA Teacher。后两者涉及参数差异，delta混合输入优势与模型优势。没有在本轮修改这些行为。
