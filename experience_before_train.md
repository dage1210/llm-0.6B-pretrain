<img width="1316" height="515" alt="image" src="https://github.com/user-attachments/assets/e062e9e4-3afe-4ae7-8452-4a7c2bcf2c17" />


# E1 实测结果
所有组固定 tokens/step = 524,288（micro × accum = 64），同一 seed、同一数据顺序，因此 loss 可直接横比。每组 24 步，吞吐取稳态窗口（第 17–24 步）。
<img width="1307" height="619" alt="image" src="https://github.com/user-attachments/assets/0519ff1c-fc8e-401b-aa84-52922c505551" />

E1 里发现的五件事
- micro 8 之后吞吐完全饱和。fp32 下 8→16 持平，bf16 下 8→16→32 反而略降、显存持续上涨。GPU 已被喂饱，瓶颈不在并行度，更大的 batch 只会浪费显存。
- compile 的收益取决于参数精度。fp32 下 +8.8%，bf16 下 -5.9%。bf16 参数 + 重算的组合可能让图不稳定、触发重编译。重复测量的噪声始终在 0.5% 以内，这个差距是真实的。
- eager attention 直接 OOM，反证了 sdpa 在走高效后端。eager 会物化 O(T²) 的注意力矩阵，sdpa 没有。所以不需要额外安装 flash-attn，省掉一个编译依赖。
- bf16 路径的第一版实现有 bug，是小规模实验抓出来的。首轮 bf16 结果吞吐漂亮，但 loss 9.30（基线 8.30）、梯度范数 49.1（基线 0.548），差 90 倍。原因是 opt.zero_grad() 只清了 fp32 master 的梯度，backward 实际累加到的 bf16 模型梯度从不清零、逐步累积。修复后 loss 8.2996、梯度范数 0.512，与基线对齐。如果跳过这一步直接上全量，这个 bug 会在第一天就毁掉整个 run，且表现为「学得慢」而非崩溃，很难被察觉。
- 顺带修了 bf16 下 resume 丢精度的问题。原实现 checkpoint 只存 bf16 模型参数，恢复时 fp32 master 从 bf16 重建，每次中断都会损失精度。现在 checkpoint 同时保存 fp32 master。对要连续跑 51 天的任务，中断恢复不是可选项。

# E2 学习率

<img width="1317" height="202" alt="image" src="https://github.com/user-attachments/assets/1ec59a02-3ed2-429f-9352-1433fa8cac69" />

三组都没有任何 loss 尖峰，梯度范数从未超过 1.0；学习率越高越好，差距远大于同配置重复跑的噪声（step 100 处两次运行相差 0.0075）。但 1.2e-3 相对 6e-4 的领先在稳步收窄：step 400 领先 0.19，500 为 0.13，600 为 0.10，700 只剩 0.085——典型的短跑偏差，高学习率早期占优，优势随训练变长而缩小。全量训练长 250 倍、恒定学习率阶段约 180B tokens。峰值学习率定为 1e-3：比短跑最优略低，作为对长跑稳定性的对冲，由 E5 验证。

# E4 官方基线（Qwen3-0.6B-Base）
<img width="1307" height="160" alt="image" src="https://github.com/user-attachments/assets/9c7cf72d-5dc8-49ba-a51e-058f2c4ac69e" />

对照必须用 Base 版：对话版经过后训练，在网页文本上的 loss 高约 0.45 nats，拿它当标尺会把自训模型的进度高估将近半个 nat。评测窗口与训练 eval 完全一致（scripts/eval_ppl.py），数值可直接对比。E2 结束时自训模型为 4.05，全量训练的每个里程碑都会与这组数字对照。




