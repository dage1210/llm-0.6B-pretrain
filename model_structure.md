一、规模选型：0.6B 还是 1B

<img width="1331" height="419" alt="image" src="https://github.com/user-attachments/assets/842f30f9-81c8-4ff3-8bcb-4c44e8a5853b" />

两者层数相同、head_dim 相同、attention 参数量恰好相同（0.6B 用 16 头 ×128 的宽 q 投影补齐了较窄的 hidden）。真正的差异在 hidden 宽度和 MLP 容量。两个配置都是 Qwen3 架构——相比 Qwen2 增加 QK-Norm、去掉 QKV bias。

二、训练预计对比
<img width="1317" height="620" alt="image" src="https://github.com/user-attachments/assets/8a452ba2-2112-48fc-bb24-b48f25eb0545" />




三、吞吐与时间预算

单 token 训练计算量 ≈ 6N + 6·L·h·s（N 含 lm_head，s = 2048）。开全量 activation checkpointing 时重算开销约 ×1.33。4090 的 bf16 峰值按 165 TFLOPS 计。

<img width="1288" height="203" alt="image" src="https://github.com/user-attachments/assets/ef236b6e-a54b-4af1-a297-50db95b725ee" />

E1 实测比原估算慢 6%，200B tokens 从 48 天修正为 51 天。
<img width="1355" height="622" alt="image" src="https://github.com/user-attachments/assets/fabd56ff-4edd-4f8b-abed-417270825cf3" />


四、为什么选择0.6B

推荐：0.6B 先行
质量差距（等算力下约 0.03 nats）小到在这个规模上会被数据配比的影响完全淹没，而 0.6B 在工程上有四个可能的优势：

- 全流程闭环从 79 天缩到 51 天（实测修正），问题早 28 天暴露。这是团队在这套硬件上的第一次完整预训练，流程风险高于模型容量风险。
- 与 Qwen3-0.6B 完全同构——同 hidden、同层数、同 head 配置、同 151936 词表。可以直接下载官方权重，用同一批 held-out 数据算 PPL，随时知道自己训到了官方的百分之多少。训练一旦不对劲，这个对照能立刻区分是「配方错了」还是「数据不够」。1B 是自定义配置，没有任何现成参照。
- 显存宽裕到能关掉 activation checkpointing——E1 实测否定：0.6B 在 micro 4 关重算也 OOM，这条优势不存在。两个规模都只能开重算，吞吐比仍约 1.54 倍（45.2k vs 29.4k），其余三条理由不受影响。
- Phase 2 每组实验（0.8B token）实测 4.9 小时，1B 估算要 8.8 小时，六组实验能省出两天多。

另外，0.6B 跑通后，1B 的配方几乎可以整套复用（同架构、同数据管线、学习率按宽度比缩放），第二轮的风险和调试成本都会低得多。如果算力窗口允许，这是最稳的两步走。

