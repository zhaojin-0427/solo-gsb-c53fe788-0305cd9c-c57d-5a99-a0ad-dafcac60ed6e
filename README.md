# 陶片接缝全局拼合服务 · Potsherd Global Seam Assembly

一个用 **Python + FastAPI + NumPy + OpenCV** 实现的 HTTP 服务。它根据成对边界点
为陶片估计**仅含旋转 + 平移**的相对位姿，并在**同一组全局位姿**下选出
**总分最高**的接缝集合——全局精确优化，**绝不逐边贪心**。

---

## 1. 它解决什么问题

每个候选接缝给出两片陶片边界上的若干匹配点对、一个正的接缝分数、位姿误差上限
（像素）与实体掩膜重叠像素上限。服务需要：

1. 由点对闭式估计相对刚体运动（Procrustes / Kabsch，二维，只允许正常旋转，
   不允许反射）；
2. 决定**采用哪些接缝**，使得存在**一组统一的全局片段位姿**，令所有被采用
   接缝同时满足：
   - 每个边界点对在全局位姿下的残差 ≤ 该接缝 `pose_error_limit`；
   - 闭环误差（接缝自带的点对拟合位姿与全局位姿推导的相对位姿之差）≤ 上限；
   - 两片陶片 alpha 实体掩膜在该相对位姿下的重叠像素数 ≤ `overlap_limit`；
3. 在所有可行集合中取**分数总和最大**者；总分相同时，取
   **已选接缝 ID 升序列表字典序最小**的集合；
4. 输出每个**连通拼合簇**的片段位姿、采用与淘汰接缝（含原因），并把该簇按
   **同一位姿**渲染成一张透明 PNG（Base64 返回）；**孤立片段各自成簇**。

不合格的接缝**不会参与成图**。

> **为什么不是逐边贪心？** 边之间通过共享片段位姿相互耦合：采用某条接缝会
> 改变闭环与所有相邻接缝的残差。本服务用**分支限界（branch & bound）在接缝
> 子集空间上做精确全局搜索**：目标是子集总分之最大，可行性由“是否存在一组
> 全局位姿”这一全局条件判定。搜索带有两项**可靠（不会改变最优解）**的加速：
> 候选图连通分量分别求解；搜索树上的每个节点用一个**单调性的位姿 LP 松弛**
> （12 边形外逼近 + 位姿修正一阶线性化）提前剪枝——LP 不可行则任何超集都不
> 可行；只有叶节点才做精确的 Gauss–Newton 全局位姿求解与掩膜光栅化校验。
>
> **多簇临时子集的 LP 变量空间。** 候选图整体连通、但完整集合不可行而进入
> 子集搜索后，某个临时已选集合可能由**多个互不相连的连通分量**组成——例如
> 同时含一个三片闭合环（A-B-C）和另一条独立接缝（D-E），而连接它们的桥接
> 缝此刻尚未被选中。每个分量以各自的最小 ID 片段为锚点（锚点不设修正变量），
> 因而各分量的修正变量个数不同（环 6 个、独立缝 3 个）。松弛对**所有分量的
> 全部非锚点节点统一编号**到一个变量向量中，各分量的约束行只写自己的列，
> 构成**块对角 LP**——等价于对各分量分别求解 LP 的合取，且行宽一致，不会把
> 宽度不同的行堆进同一个数组（旧实现会抛
> `ValueError: ... inhomogeneous shape` 导致接口 500）。
>
> **淘汰原因的完整子图复核。** 报告每条未选接缝的原因时，服务把该候选与
> **它影响的完整已选子图**（其两端点所在已选连通分量的全部已选边；桥接
> 两个已选簇时两侧都纳入）在同一组全局位姿下重新拟合，再按**实际失败项**
> 裁定原因：增广集合不可行时给出真实的残差/闭环/重叠超限（诊断值与原因
> 一致），可行时才报 `score_optimization_not_selected`。只取候选两端的一跳
> 已选边会把已选路径 A-B-C-D 截断、漏掉 B-C，使本应因闭环超限淘汰的候选
> A-D 被误报为分数未选——本服务不会这样做（见第 4 节的闭环候选示例）。

---

## 2. 一键启动（Docker Compose）

```bash
docker compose up --build
```

服务监听在 http://localhost:8000 ：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 健康检查，返回 `{"status":"ok"}` |
| POST | `/assemblies` | 提交陶片与候选接缝，返回全局拼合结果 |
| GET  | `/docs` | 交互式 OpenAPI 文档（可直接发请求） |

停止：

```bash
docker compose down
```

### 不用 Docker 的本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

跑测试：

```bash
pytest -q
```

---

## 3. 请求格式 `POST /assemblies`

```jsonc
{
  "fragments": [
    {
      "id": "shard-A",                       // 片段唯一 ID
      "image_base64": "iVBORw0KGgo...."       // 透明 PNG（alpha 通道），可带 data: 前缀
    }
  ],
  "seams": [
    {
      "id": "seam-AB",                        // 接缝唯一 ID
      "fragment_a": "shard-A",
      "fragment_b": "shard-B",
      "score": 12.5,                          // 正分数，越大越好
      "pose_error_limit": 3.0,                // 点对残差/闭环误差上限（像素）
      "overlap_limit": 200,                   // 实体掩膜重叠像素上限
      "pairs": [                              // 至少 2 对，且不能退化（共点）
        {"a": [71.0, 12.0], "b": [3.0, 12.0]},
        {"a": [72.4, 22.0], "b": [4.4, 22.0]}
      ]
    }
  ]
}
```

坐标为各片段**自身图像坐标系**中的像素坐标 `[x, y]`（原点左上角，x 向右、
y 向下）。点对 `a`、`b` 分别是边界点在 `fragment_a`、`fragment_b` 上的位置。

### 422（不可处理）情形

- 点对少于 2 对，或点对退化（全部重合，旋转不可观）；
- 片段 PNG 解码失败 / Base64 非法；
- 片段 alpha 实体掩膜为空；
- 接缝引用了不存在的片段、片段自连、ID 重复。

这些都返回 HTTP `422`，body 为 `{"detail": "...原因..."}`。

---

## 4. 实际请求与响应示例

仓库自带一个可直接生成真实请求并调用服务的脚本：

```bash
# 1) 生成 examples/request.json（3 张透明 PNG：两片可拼合 + 一片孤立）
python examples/make_example.py

# 2) 对正在运行的服务发起真实 POST，落地 examples/response.json 与渲染 PNG
python examples/make_example.py --call
```

或用 curl：

```bash
curl -s -X POST http://localhost:8000/assemblies \
  -H 'Content-Type: application/json' \
  --data @examples/request.json | python -m json.tool
```

实际响应（`examples/response.json`；PNG 为真实 Base64，下方截断展示）：

```json
{
  "clusters": [
    {
      "cluster_id": 1,
      "fragment_ids": ["shard-A", "shard-B"],
      "poses": [
        {"fragment_id": "shard-A", "theta": 0.0, "tx": 0.0, "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "shard-B", "theta": 0.0, "tx": 67.0, "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0}
      ],
      "adopted_seams": [
        {"id": "seam-AB", "fragment_a": "shard-A", "fragment_b": "shard-B",
         "score": 12.5, "pair_residual_max": 0.0, "loop_error": 0.0,
         "overlap_pixels": 0}
      ],
      "rejected_seams": [],
      "total_score": 12.5,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAK8AAACvCAYAAACLko51AAAG...（透明 PNG）"
    },
    {
      "cluster_id": 2,
      "fragment_ids": ["shard-C"],
      "poses": [
        {"fragment_id": "shard-C", "theta": 0.0, "tx": 0.0, "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0}
      ],
      "adopted_seams": [],
      "rejected_seams": [],
      "total_score": 0.0,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAAADgAAAA4CAYAAACohjseAAAB...（透明 PNG）"
    }
  ],
  "adopted_seam_ids": ["seam-AB"],
  "rejected_seams": [],
  "total_score": 12.5
}
```

`--call` 还会把每个簇按同一位姿渲染的透明 PNG 写到
`examples/cluster_1.png`、`examples/cluster_2.png`，可直接打开查看：
`cluster_1.png` 是两片陶片在统一全局位姿下拼回的完整陶板，`cluster_2.png`
是孤立的第三片。

### 淘汰接缝的原因码（`reason`）

| reason | 含义 |
|--------|------|
| `pairwise_pose_residual_exceeds_limit` | 该接缝点对自身的刚体拟合残差已超上限（全局位姿无法修复） |
| `pose_residual_exceeds_limit` | 与已采用接缝放在同一组全局位姿后，点对残差超上限 |
| `loop_closure_exceeds_limit` | 闭环误差（相对位姿不自洽）超上限 |
| `mask_overlap_exceeds_limit` | 该相对位姿下实体掩膜重叠像素数超上限 |
| `score_optimization_not_selected` | 本身可行，但为使总分最大而未被精确搜索选中 |

淘汰记录同时回填测得的 `pair_residual_max` / `loop_error` /
`overlap_pixels`（不适用时为 `null`）。淘汰接缝只出现在报告里，
**不会出现在任何簇的渲染图中**。

> **淘汰原因如何复核（与最优集同一套全局判据）。** 每条未入选候选缝
> 都与**它实际影响的完整已选子图**一起重估：即已选接缝图中与候选两
> 端点相连通的**全部**已选边（候选连接两个已选簇时两侧都纳入），加
> 上候选本身，在同一组全局位姿下重新拟合。只取候选两端的一跳已选边
> 会把已选路径 A-B-C-D 截成两段、丢掉 B-C，使本应闭环超限的候选
> A-D 被误报为 `score_optimization_not_selected`——本服务不会这样做。
> 增广集合不可行时，按**实际超限的约束项**裁定原因：候选自身超限就报
> 候选自身的实测值；若增广拟合把某条原已选边推超其上限而候选本身未
> 超限，则按该已选边实际失败的项（点对残差 / 闭环 / 掩膜重叠）给出原
> 因，并回填该失败项的实测诊断值——不会笼统误报掩膜重叠，诊断值与
> 原因始终一致。只有增广集合确实可行时才报
> `score_optimization_not_selected`。复核只读取候选连通的已选子图，
> 不相关簇的位姿与淘汰归属不受影响。

### 多簇子集搜索示例（闭环簇 + 独立接缝）

这是修复前会让 `POST /assemblies` 返回 500 的场景：候选接缝图整体连通
（A-B-C 与 D-E 由低分桥接缝 C-D 相连），A-B-C 构成一个几何自洽的三片
**闭环**，但闭环弦 `bc` 掩膜重叠超限，完整集合不可行；搜索临时放下桥接缝
时，已选集合同时含一个三片闭环簇和一条独立接缝（D-E），两簇修正变量个数
不同。脚本生成该真实请求（5 张矩形透明 PNG、5 条接缝）并调用服务：

```bash
python examples/make_multicluster_example.py --call
curl -s -X POST http://localhost:8000/assemblies \
  -H 'Content-Type: application/json' \
  --data @examples/multi_request.json | python -m json.tool
```

实际响应（`examples/multi_response.json`；服务正常返回 200，PNG 为真实
Base64，下方截断展示）。搜索舍弃重叠超限的高分弦 `bc`（9 分），保留
`ab+ac`（16 分）与桥链 `cd+de`（8 分），取到可行的最高总分 **24**，
并诚实给出淘汰原因与实测重叠像素：

```json
{
  "clusters": [
    {
      "cluster_id": 1,
      "fragment_ids": ["A", "B", "C", "D", "E"],
      "poses": [
        {"fragment_id": "A", "theta": 0.0, "tx": 0.0,   "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "B", "theta": 0.0, "tx": 39.0,  "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "C", "theta": 0.0, "tx": 0.0,   "ty": 39.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "D", "theta": 0.0, "tx": 79.0,  "ty": 39.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "E", "theta": 0.0, "tx": 158.0, "ty": 39.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0}
      ],
      "adopted_seams": [
        {"id": "ab", "fragment_a": "A", "fragment_b": "B", "score": 8.0,
         "pair_residual_max": 0.0, "loop_error": 0.0, "overlap_pixels": 40},
        {"id": "ac", "fragment_a": "A", "fragment_b": "C", "score": 8.0,
         "pair_residual_max": 0.0, "loop_error": 0.0, "overlap_pixels": 40},
        {"id": "cd", "fragment_a": "C", "fragment_b": "D", "score": 1.0,
         "pair_residual_max": 0.0, "loop_error": 0.0, "overlap_pixels": 40},
        {"id": "de", "fragment_a": "D", "fragment_b": "E", "score": 7.0,
         "pair_residual_max": 0.0, "loop_error": 0.0, "overlap_pixels": 0}
      ],
      "rejected_seams": [
        {"id": "bc", "fragment_a": "B", "fragment_b": "C", "score": 9.0,
         "reason": "mask_overlap_exceeds_limit",
         "pair_residual_max": 0.2465, "loop_error": 0.2465,
         "overlap_pixels": 40}
      ],
      "total_score": 24.0,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAAANUAAABeCAYAAAC0N1GY...（透明 PNG）"
    }
  ],
  "adopted_seam_ids": ["ab", "ac", "cd", "de"],
  "rejected_seams": [
    {"id": "bc", "reason": "mask_overlap_exceeds_limit", "...": "..."}
  ],
  "total_score": 24.0
}
```

单簇闭环、纯生成树（无闭环）与完全孤立的片段仍走各自的正常路径
（树只校掩膜重叠、孤立片段各自成簇）。

### 闭环候选的淘汰归因示例（A-B-C-D 路径 + A-D 候选）

这是淘汰原因曾被误报的场景：最优已选集是路径 **A-B-C-D**（接缝
`ab`、`bc`、`cd`），未选候选 `ad` 的点对自身拟合残差为 0（通过成对
预筛），但它隐含的 A→D 相对位姿 `(39, 12)` 与链式推导的 `(117, 0)`
矛盾约 79 px——与已选路径构成**位姿不自洽的闭环**。旧实现只取候选
两端的一跳已选边（`ab`、`cd`）重拟合，漏掉 `bc`，增广图退化成可精确
拟合的树，于是把 `ad` 误报为 `score_optimization_not_selected`。修复后
服务划定候选影响的**完整已选子图**（`ab+bc+cd` 与 `ad` 一起在同一组
全局位姿下重拟合），闭环不可行，按实际超限项给出诚实原因。脚本生成
真实请求（4 张 40×40 透明 PNG、4 条接缝）并调用服务：

```bash
python examples/make_loop_rejection_example.py --call
curl -s -X POST http://localhost:8000/assemblies \
  -H 'Content-Type: application/json' \
  --data @examples/loop_request.json | python -m json.tool
```

实际响应（`examples/loop_response.json`；PNG 为真实 Base64，截断展示）。
最优集保留路径 `ab+bc+cd`（总分 **24**），`ad` 以
`pose_residual_exceeds_limit` 淘汰，诊断值（16.21 px > 3.0 上限）与
原因一致：

```json
{
  "clusters": [
    {
      "cluster_id": 1,
      "fragment_ids": ["A", "B", "C", "D"],
      "poses": [
        {"fragment_id": "A", "theta": 0.0, "tx": 0.0,   "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "B", "theta": 0.0, "tx": 39.0,  "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "C", "theta": 0.0, "tx": 78.0,  "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0},
        {"fragment_id": "D", "theta": 0.0, "tx": 117.0, "ty": 0.0,
         "pair_residual_rms": 0.0, "pair_residual_max": 0.0}
      ],
      "adopted_seams": [
        {"id": "ab", "fragment_a": "A", "fragment_b": "B", "score": 8.0,
         "pair_residual_max": 0.0, "loop_error": 0.0, "overlap_pixels": 40},
        {"id": "bc", "fragment_a": "B", "fragment_b": "C", "score": 8.0,
         "pair_residual_max": 0.0, "loop_error": 0.0, "overlap_pixels": 40},
        {"id": "cd", "fragment_a": "C", "fragment_b": "D", "score": 8.0,
         "pair_residual_max": 0.0, "loop_error": 0.0, "overlap_pixels": 40}
      ],
      "rejected_seams": [
        {"id": "ad", "fragment_a": "A", "fragment_b": "D", "score": 5.0,
         "reason": "pose_residual_exceeds_limit",
         "pair_residual_max": 16.2146, "loop_error": 16.2146,
         "overlap_pixels": 1046}
      ],
      "total_score": 24.0,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAKwAAAA3CAYAAACRkKFY...（透明 PNG）"
    }
  ],
  "adopted_seam_ids": ["ab", "bc", "cd"],
  "rejected_seams": [
    {"id": "ad", "reason": "pose_residual_exceeds_limit", "...": "..."}
  ],
  "total_score": 24.0
}
```

候选桥接两个已选簇时，两侧连通分量的全部已选边都纳入复核（不相关簇
的位姿与淘汰归属保持不变）；增广集合不可行时一律按实际失败项裁定，
绝不报 `score_optimization_not_selected`。

---

## 5. 模块结构

```
app/
  geometry.py    闭式刚体（旋转+平移）拟合、位姿复合/相对位姿
  posegraph.py   生成树初始化 + 全局 Gauss–Newton 位姿图求解（所有接缝同时拟合）
  relax.py       位姿可行性的 LP 松弛（单调性、可靠剪枝；多分量块对角变量空间）
  masks.py       alpha 掩膜、相对位姿下的实体重叠计数、透明 PNG 渲染
  assembler.py   解码/建模、可行性判定、分支限界精确全局选缝
  service.py     端到端编排：成簇、采用/淘汰归因、渲染与响应
  schemas.py     Pydantic 请求/响应模型
  main.py        FastAPI 应用（/health、/assemblies，422 处理）
examples/        真实请求/响应生成脚本与产物（含多簇闭环与闭环候选淘汰归因场景）
tests/           API、几何、全局优化器（含对拍暴力枚举）与多片段闭环测试
```

### 位姿约定

片段位姿 `(theta, t)` 把片段局部坐标映到簇/世界坐标：
`p_world = R(theta) · p_local + t`。每个簇以该簇最小 ID 的片段为锚点
（位姿为恒等）。接缝内部保存由点对拟合的、把 `fragment_b` 局部坐标映到
`fragment_a` 局部坐标的相对位姿 `Rr = R_aᵀ R_b, tr = R_aᵀ(t_b − t_a)`。
