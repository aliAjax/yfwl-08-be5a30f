# 失序二维频谱的稀疏底片

夜里两点，频谱复核员收到同一块材料的三次扫描。每次扫描都经过了二维 Walsh–Hadamard 变换，但扫描台把系数行、系数列分别打乱，又给两轴各加了一组未知符号相位。眼前只剩三张没有原轴标签的整数频谱，以及一组由原始系数矩阵导出的聚合校验。

入口文件固定为 `/app/walsh_recover.py`：

```text
python3 /app/walsh_recover.py --input <输入路径> --output <结果路径> --ledger <账本路径>
```

`frames` 给三张去标号的二维频谱。频谱矩阵的元素位置仍然固定，系数行和系数列则在采集阶段被未知置换与符号相位改写。三张频谱必须对应同一个 8×8 稀疏整数系数矩阵。

输入的根对象只允许：

```text
schema_version, case_id, size, sparsity, coefficient_abs_max,
frames, axis_checks, limits
```

`schema_version` 必须是整数 `1`。`case_id` 是非空字符串。`size` 固定为 `8`。`sparsity` 是 `8` 至 `12` 的整数。`coefficient_abs_max` 是 `1` 至 `20` 的整数。

`frames` 恰好含三张频谱。`frames` 的每一项字段固定为：

```text
frame_id, spectrum
```

`frame_id` 非空且三张互异。每帧的 `spectrum` 是 8×8 频点表：外层有 8 条记录，每条记录含 8 个频点读数，单元格落在闭区间 `[-4096, 4096]` 内。令 `H` 为 8 阶 Sylvester Hadamard 矩阵，首行全为 `1`。若某帧的观测轴到规范轴的置换为 `row_permutation`、`column_permutation`，规范轴上的符号为 `row_signs`、`column_signs`，则频谱满足：

```text
spectrum = H diag(row_signs) P_row C P_col diag(column_signs) H
```

其中 `P_row` 的第 `o` 行第 `i` 列在规范行 `i` 落到观测行 `o` 时为 `1`。三张频谱的置换和符号独立，底片矩阵 `C` 相同。

`axis_checks` 字段固定为：

```text
row_energy_moments, column_energy_moments,
row_sign_parities, column_sign_parities, global_sign_parity
```

行能量定义为 `sum(abs(C[i][j]))`。把规范行号加一作为采样权重，`row_energy_moments[q]` 记录 q 阶加权后的行能量总和，`q` 取 `0` 至 `7`。列侧沿用这一做法，写入 `column_energy_moments`。八个行能量必须互异，八个列能量也必须互异。

`row_sign_parities[i]` 是第 `i` 行所有非零系数符号位的异或，负数记 `1`、正数记 `0`。`column_sign_parities[j]` 对第 `j` 列采用同样规则。每个奇偶值只能是整数 `0` 或 `1`。

`global_sign_parity` 是满足 `(row + 2 * column) % 5 == 0` 的非零项符号位异或，用来消除支撑图中的偶环反号自由度。

`limits` 的字段固定为：

```text
max_size, max_sparsity, max_frames
```

三个上限依次是 `8`、`12`、`3`。`sparsity` 最高可取 `12`；读到 `13` 或更大值时不再继续，过程以状态码 `3` 结束。

底片矩阵每行、每列至少有一个非零项；每个非零值的绝对值不超过 `coefficient_abs_max`。三帧逆变换得到的绝对值支撑必须完全相同，规范化后的支持权重之和就是全局支持度。

成功结果写到 `--output`。根字段固定为：

```text
schema_version, case_id, matrix, frames, support_score
```

`matrix` 恰好含 `sparsity` 项，按 `row`、`column` 递增。元素字段固定为：

```text
row, column, value
```

`row`、`column` 是 8×8 规范网格的整数下标，取值为 `0` 至 `7`；`value` 是非零整数。`support_score` 等于全部 `abs(value)` 之和。

输出 `frames` 按 `frame_id` 递增。输出 `frames` 的元素字段取以下五个名字：

```text
frame_id, row_permutation, column_permutation, row_signs, column_signs
```

`row_permutation[o]` 是观测行 `o` 对应的规范行。`column_permutation[o]` 是观测列 `o` 对应的规范列。两个数组都必须是 `0` 至 `7` 的排列。符号数组每项为 `1` 或 `-1`。

支撑图按非零项连接行列。每个支撑连通分量可同时翻转全部行、列符号而不改变频谱。规定每个分量内最小行号的 `row_signs` 必须为 `1`，从而全局也有 `row_signs[0]=1`。

`--output` 写成单行 UTF-8 JSON：对象键按字典序排列，分隔符旁不添空格，文件以唯一换行收尾。成功复核不向标准输出写字符，标准错误也保持空白。

写入账本的校验事件字段限制为：

```text
case_id, event, input_sha256, result_sha256, schema_version
```

`event` 为 `completed`。原始票据和复核文件各生成一条小写 SHA-256 摘要。同一票据重复复核只会复用既有账本内容，旧账本字节保持不变。新的成功批次要追加一行，每写完一条事件立即追加单个换行。

读取、解码或解析票据失败时立即结束，进程状态码为 `3`。JSON 不允许重复键覆盖旧值；出现重复键同样退出 `3`。字段集合需要逐层核对。每个值的类型也要核对。公开数值范围同样要核对。任一项不符就中止复核并返回 `3`。

逆变换若不能还原成整数底片，复核到此为止，状态码为 `3`。能量矩与谱面矛盾时，同样返回 `3`。行、列能量若打平，规范轴不再唯一，仍以 `3` 退出。

符号奇偶相互冲突时结束，状态码为 `3`。符号方程若留下多个解，也不输出底片。三帧的绝对值支撑不能逐格重合时，同样按 `3` 退回。

被退回的调用不向控制台写字符，`--output` 与 `--ledger` 都维持调用前的字节。

提交新底片时先写旁路文件，再用一次替换落到目标名。替换若失败，旧结果和旧账本仍可原样读取，旁路文件也不会残留。
