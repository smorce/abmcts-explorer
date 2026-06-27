# AB-MCTS-M（Mixed Model）仕様書

本書は、Sakana AI の [TreeQuest](https://github.com/SakanaAI/treequest) が提供する
**AB-MCTS-M（Adaptive Branching Monte Carlo Tree Search / Mixed Model）** の仕様と、
本リポジトリでの利用方法をまとめたものである。

- 対象読者: 本リポジトリで探索ループを設計・拡張する開発者
- 関連実装: `experiments/arc2/run.py`, `experiments/arc2/utils.py`, `experiments/arc2/prompt.py`
- 依存: `treequest[abmcts-m,vis]`（`pyproject.toml` で定義済み）

---

## 1. 概要

### 1.1 AB-MCTS とは

AB-MCTS（Adaptive Branching MCTS）は、LLM の推論時スケーリング（inference-time scaling）を目的とした探索手法である。
通常の MCTS が「固定された branching factor で子ノードを展開する」のに対し、AB-MCTS は外部フィードバック（スコア）を使い、
各選択ステップで以下を **動的・逐次的** に決める。

- **wider（幅方向）**: ルートや既存ノードから新しい候補を作る
- **deeper（深さ方向）**: 既存の有望な候補をさらに改善・深掘りする

TreeQuest は 2 つの実装を提供する。

| アルゴリズム | クラス | 事後分布の扱い | 追加依存 |
| --- | --- | --- | --- |
| AB-MCTS-A | `tq.ABMCTSA` | ノード単位の独立分布 | なし |
| AB-MCTS-M | `tq.ABMCTSM` | **混合効果モデル（mixed-effects model）** | `treequest[abmcts-m]` / `treequest[all]` |

### 1.2 AB-MCTS-M（Mixed Model）の特徴

AB-MCTS-M は各選択ステップで **PyMC による混合モデル（mixed modeling）を当てはめる**。
これにより、観測スコアのばらつきを 2 階層に分けて捉える。

1. **初期生成のばらつき（between-group / アクション間の変動）**
   どの生成・修正・採点アクションが有望かという、グループ（アクション）ごとの効果。
2. **その後の refinement のばらつき（within-group / アクション内の stochasticity）**
   同じアクションでも、LLM の temperature やサンプリングで結果が揺らぐランダム成分。

混合モデルは新規分岐を表す **GEN ノード**（= まだ生成していない「新しい子を作る」行動）の予測分布も扱うため、
「既存ノードを深掘りすべきか / 新しい候補を生むべきか」を確率的に判断できる。

> ⚠️ 各 step で PyMC のフィッティングが走るため、AB-MCTS-M は **1 ステップが重い**。
> 性能・計算コストのトレードオフは「7. パフォーマンスと運用上の注意」を参照。

---

## 2. 依存関係とインストール

AB-MCTS-M は PyMC を用いるため、追加 extra が必要。

```bash
# 単体で試す場合
pip install "treequest[abmcts-m]"
# もしくは全部入り
pip install "treequest[all]"
```

本リポジトリでは `uv` と `pyproject.toml` で管理しており、AB-MCTS-M の extra は導入済み。

```37:37:pyproject.toml
    "treequest[abmcts-m,vis]>=0.1.0",
```

> OneDrive 配下のためハードリンク不可。`uv` 実行時は必ず `--link-mode=copy`（または `set UV_LINK_MODE=copy`）を付ける。
> 文字化け対策として `$env:PYTHONUTF8='1'` も推奨。

---

## 3. コア API

TreeQuest の AB-MCTS-M は探索木（`search_tree`）を **不変オブジェクト的に受け渡す**設計で、
各操作は新しい木を返す。状態はアルゴリズムインスタンスではなく木に保持される。

| API | 役割 |
| --- | --- |
| `algo = tq.ABMCTSM(...)` | アルゴリズムの初期化 |
| `search_tree = algo.init_tree()` | 空の探索木を生成 |
| `search_tree = algo.step(search_tree, generate_fns)` | **1 ノードだけ** 追加（内部で ask→generate→tell） |
| `search_tree, trials = algo.ask_batch(search_tree, batch_size, actions)` | 展開すべき (親, アクション) を `batch_size` 件まとめて取得 |
| `search_tree = algo.tell(search_tree, trial_id, (state, score))` | 生成結果を木に反映 |
| `tq.top_k(search_tree, algo, k=1)` | スコア上位 k の `(state, score)` を取得 |
| `algo.get_state_score_pairs(search_tree)` | 全ノードの `(state, score)` を取得（件数カウント等に利用） |
| `tq.render(search_tree, ...)` | 探索木を HTML 等で可視化（`treequest[vis]`） |

### 3.1 生成関数（generate_fns）の契約

- `generate_fns` は `{ action_label: callable }` の辞書。各 `action_label` は TreeQuest から見た **1 つの探索アクション**。
- 各生成関数のシグネチャは `fn(parent_state: State | None) -> tuple[State, float]`。
  - `parent_state is None` → ルートからの初期生成。
  - それ以外 → `parent_state` を親とする refinement / 修正など。
- 戻り値の `score` は **必ず `[0, 1]` に正規化** する（探索の比較基準になるため）。

---

## 4. 実行モデル（step の意味）

### 4.1 「N 個のアクションを渡す = 最初に N 個の子を全展開」ではない

`generate_fns` に N 個の variation を渡しても、TreeQuest は最初に深さ 1 の子を N 個すべて作るわけではない。
`step()` 1 回で増えるノードは **必ず 1 個**。

```
step 1: root から 1 個生成（N アクションのうち 1 つが選ばれる）
step 2: root に兄弟を足す(wider) か、step1 のノードを深掘り(deeper) するかを AB-MCTS-M が判断
step 3: 同様に毎回 1 個だけ追加
```

つまり深さ 1 が N 個そろう前に、depth 2 / depth 3 へ進むことがある。

generate_fns:
```model_name ✕ prompt_style ✕ action_type ✕ scorer_name
o4-mini        × direct     × sample × exact
gemini-2.5-pro × structured × sample × heuristic
deepseek-r1    × reasoning  × sample × consistency
o4-mini        × reasoning  × refine × exact
o4-mini        × critic     × repair × heuristic
gemini-2.5-pro × structured × refine × consistency
gemini-2.5-pro × critic     × repair × exact
```
これで7個の variation になる。 variation = aciton 。

### 4.2 2 段階の選択（場所の選択 と アクションの選択）

AB-MCTS-M（multiarm bandit Thompson strategy）の選択は **階層の異なる 2 つの軸** からなる。
この 2 つは排他ではなく、役割が違う。

| 軸 | 何を決めるか |
| --- | --- |
| **deeper / wider** | どの場所（親ノード）を展開するか |
| **action / variation** | その場所で、どの `generate_fn` で子ノードを生成するか |

各ノードで AB-MCTS-M はまず「既存 child を選ぶ」か「GEN ノード（= 新規生成）を選ぶ」かを Thompson Sampling 的に判断する。

```text
root
├── child A
├── child B
└── GEN        ← この仮想ノードが「新しい子を作る」行動
```

- **既存 child が選ばれた（deeper）** → 「その child に降りる」だけ。
  **この瞬間には `generate_fns` のアクションはまだ選ばれない**（新ノードを作っていないため `generate_fns[action](parent_state)` は呼ばれない）。
  降りた先のノードで、再び同じ「child を選ぶか GEN を選ぶか」の判断を繰り返す。
- **GEN が選ばれた（wider）、または葉ノードに到達した** → ここで初めて
  「どの action / variation で新ノードを作るか」が決まり、`generate_fn` が呼ばれる。

つまり、最終的に 1 つの `step()` で新ノードを 1 つ追加するには、**必ず以下の 2 つが両方決まる**。

```text
parent node : どのノードを親にするか（deeper を辿った末の到達点）
action      : どの generate_fn で子ノードを生成するか（GEN/葉で選択）
```

### 4.2.1 選択の擬似コード

```python
node = root
while node.children:
    selected = select_child_or_gen(node)   # Thompson Sampling
    if selected == "GEN":
        action = select_action(generate_fns)  # ここで初めて action 確定
        return node, action
    else:
        node = selected_child               # deeper: action はまだ不要
# 葉に到達した場合
action = select_action(generate_fns)
return node, action
```

TreeQuest の実装では、内部の `_select_child()` が

- **文字列（action 名）を返す** → そのノードから新ノードを生成（GEN）
- **child index を返す** → 既存 child へ降りる（このとき `action=None` で返り、ループがその child に進む）

という分岐になっている。`ask_batch()` が返す `Trial` には常に `trial.action` と `trial.parent_state` が入り、
生成実行時には必ず `generate_fns[trial.action](trial.parent_state)` という形になる
（＝生成が走る時点では常に action が確定している）。

### 4.2.2 deeper を辿ってから生成する例

```text
step 10:

root        ─ deeper → child_2
child_2     ─ deeper → child_2_1
child_2_1   ─ GEN    → 新ノードを生成

  action = gemini-2.5-pro × critic × repair × exact
  generate_fns[action](child_2_1.state)
```

このように deeper を複数回辿った末の到達点（`child_2_1`）が parent となり、
その場所で選ばれた action の `generate_fn` が呼ばれて新ノードが 1 つ追加される。

### 4.3 すべてのアクションを最初に試したい場合（warm start）

固定 branching を AB-MCTS-M に任せるのではなく、最初だけ明示的に各アクションを 1 回ずつ展開する。

```python
# warm start: 各 action を root から 1 回ずつ
for action, fn in generate_fns.items():
    search_tree, trial = algo.ask(search_tree, [action])
    result = fn(trial.parent_state)
    search_tree = algo.tell(search_tree, trial.trial_id, result)

# 以降は AB-MCTS-M に任せる
for _ in range(remaining_budget):
    search_tree = algo.step(search_tree, generate_fns)
```

ただし AB-MCTS-M の思想は「固定幅で全展開」ではなく「幅と深さを逐次適応選択」であることに注意。

---

## 5. variation 設計（model × prompt × action × scorer）

`generate_fns` のキーを「モデル名」だけでなく **model × prompt × action × scorer** の組み合わせにすると、
それぞれが別の探索アクションになる。

```python
generate_fns = {
    variant.label: partial(generate_with_variant, variant=variant)
    for variant in VARIANTS
}
```

### 5.1 ばらつきの 2 階層

| 階層 | 対応 | 実 LLM での例 |
| --- | --- | --- |
| アクション内 stochasticity | 同一アクション内のランダム性 | temperature / sampling の揺らぎ |
| アクション間 variation | model / prompt / action / scorer の違い | 解き方そのものの違い |

AB-MCTS-M は混合モデルでこの 2 階層を分離し、「どの種類の生成・修正・採点が有望か」を探索中に学習する。

### 5.2 プロンプト設計 = 探索戦略設計

variation を入れるなら、**失敗モードの異なるプロンプトを少数** 入れるのが効果的。
名前だけ違って中身が似ているプロンプトは、AB-MCTS-M から見て重複アクションになり探索空間を悪化させる。

推奨するアクションの軸（例）:

| プロンプト/アクション | 役割 |
| --- | --- |
| `direct` | 低コストで即答する |
| `reasoning` | 手順を明示して解く |
| `decompose` | 問題を小問題に分けて統合する |
| `critic` / `criticize` | 既存解答の誤りを探す |
| `repair` | 誤りを特定して修正する |
| `verify` | 最終答えの正しさを判定する |
| `alternative` / `new_angle` | 別解・別視点を作る |

- `refine`（より良くする）と `repair`（誤りを直す）は区別すると有効。
  `refine` は元の誤答を引きずりやすく、`repair` はエラー訂正に向く。
- 避けたい例（探索アクションとして重複しやすい）:
  「丁寧に / 慎重に / よく考えて / ステップバイステップで / 詳細に解いてください」。

### 5.3 スコアリングの正規化（重要な落とし穴）

`scorer` をアクションごとに変える場合でも、TreeQuest に返す `score` は
**同じ意味の `[0, 1]` スケールに揃える**こと。

- スコアの意味がアクションごとにズレると、探索が「良い回答」ではなく「甘い採点器」を選ぶ方向に偏る。
- 「長い解答ほど高得点」のようなスコアラーは、長文プロンプトを報酬化してしまう（悪い例）。

```python
# 悪い例: 良い解答ではなく長い解答を報酬化している
if len(answer) > 500:
    score += 0.2
```

良いスコアラーは、プロンプト由来の文体差に左右されず、タスクの成功そのものを測る。

| タスク | 測るべき指標 |
| --- | --- |
| 数学 | 正答か / 計算過程に矛盾がないか / 最終答えが明示されているか |
| コード生成 | テストが通るか / 型エラーがないか / 仕様充足 / 余計な副作用がないか |
| 文章生成 | 要件充足 / 禁止事項違反の有無 / 読者・目的への適合 |

---

## 6. 本リポジトリでの利用

DeepResearch 実験（`experiments/arc2`）は AB-MCTS-M を用いて研究レポートを探索的に生成する。

### 6.1 アクション定義

```19:25:experiments/arc2/utils.py
DEFAULT_ACTIONS: tuple[ResearchAction, ...] = (
    "new_angle",
    "deepen",
    "criticize",
    "revise",
)
PERSPECTIVES: tuple[str, ...] = ("市場", "技術", "戦略", "リスク", "実装")
```

各アクションは `partial` で `generate_fn` にバインドされ、`generate_fns` を構成する。

```670:688:experiments/arc2/run.py
    generate_fns = {
        action: partial(
            generate_fn,
            action=action,
            topic=topic,
            ...
        )
        for action in actions
    }
```

### 6.2 ask_batch / tell ループ（AB-MCTS-M 用）

AB-MCTS-M は step が重いため、README 推奨の `ask_batch()` / `tell()` 形式でバッチ実行する。

```699:710:experiments/arc2/run.py
    if algo_name == "ABMCTSM":
        num_steps = max(1, (max_num_nodes + batch_size - 1) // batch_size)
        for step in tqdm(range(num_steps)):
            node_start = time.time()
            search_tree, trials = algo.ask_batch(search_tree, batch_size, actions)
            for trial in trials:
                result = generate_fns[trial.action](trial.parent_state)
                search_tree = algo.tell(search_tree, trial.trial_id, result)
            node_times.append(time.time() - node_start)
            n_states = count_states(search_tree, algo)
            research_logger.log_event("batch_completed", step=step + 1, n_states=n_states)
            save_checkpoint(search_tree, save_dir, n_states)
```

その他のアルゴリズム（AB-MCTS-A 等）は `step()` ループを用いる。

```713:715:experiments/arc2/run.py
        for step in tqdm(range(max(0, max_num_nodes - initial_states))):
            node_start = time.time()
            search_tree = algo.step(search_tree, generate_fns)
```

### 6.3 アルゴリズム初期化（max_process_workers）

AB-MCTS-M は PyMC のフィッティングを並列化できる。本リポジトリでは初期化失敗時に
`max_process_workers=batch_size` でフォールバックする。

```472:479:experiments/arc2/run.py
def build_algorithm(algo_name: str, params: dict[str, Any], batch_size: int) -> tq.Algorithm:
    algo_cls = getattr(tq, algo_name)
    try:
        return algo_cls(**params)
    except TypeError:
        if algo_name == "ABMCTSM":
            return algo_cls(max_process_workers=batch_size)
        raise
```

### 6.4 状態（State）

本リポジトリの `State` は `NodeState`（`experiments/arc2/utils.py`）。
TreeQuest が要求する `score` に加え、`action` / `perspective` / `facet` / `depth` / `findings` 等の
メタ情報を保持し、ロギング・可視化・後段のプロンプト構築に再利用する。
`score` は `score_review()` で `[0, 1]` にクランプされる（重複ペナルティ・カバレッジボーナス込み）。

---

## 7. パフォーマンスと運用上の注意

- **step が重い**: AB-MCTS-M は各選択で PyMC を当てはめるため、`step()` 単体は遅くなりやすい。
  `ask_batch()` / `tell()` でバッチ化し、`batch_size <= 5` あたりから試すのが無難。
- **LLM 呼び出しがボトルネック**: 実運用では各 trial が LLM 呼び出し（本リポジトリでは planner / researcher / reviewer の複数回）を伴うため、
  バッチ内で生成を並列化すると効率的。
- **チェックポイント**: 探索木は pickle 化可能。`save_checkpoint()` で `checkpoint_latest.pkl` 等を保存し、`checkpoint_path` から再開できる。
- **可視化**: `tq.render(..., format="html")` で探索木を HTML 出力（`treequest[vis]`）。
- **Windows / OneDrive**: `uv run --link-mode=copy`、`$env:PYTHONUTF8='1'` を推奨。

---

## 8. 最小サンプル

### 8.1 単一アクション（step ループ）

```python
from dataclasses import dataclass
import random
import treequest as tq


@dataclass
class State:
    answer: str
    score: float
    depth: int


def generate(parent_state: State | None) -> tuple[State, float]:
    if parent_state is None:
        score = random.random()
        state = State(answer=f"initial, score={score:.3f}", score=score, depth=0)
    else:
        score = min(1.0, max(0.0, parent_state.score + random.uniform(-0.1, 0.2)))
        state = State(
            answer=f"refined from [{parent_state.answer}]",
            score=score,
            depth=parent_state.depth + 1,
        )
    return state, state.score


def main():
    algo = tq.ABMCTSM()              # treequest[abmcts-m] / [all] が必要
    search_tree = algo.init_tree()
    generate_fns = {"llm_generate_or_refine": generate}

    for i in range(30):              # 生成予算
        search_tree = algo.step(search_tree, generate_fns)
        if (i + 1) % 5 == 0:
            best_state, best_score = tq.top_k(search_tree, algo, k=1)[0]
            print(f"[{i + 1}] best={best_score:.3f} depth={best_state.depth}")

    best_state, best_score = tq.top_k(search_tree, algo, k=1)[0]
    print("BEST:", best_score, best_state.answer)


if __name__ == "__main__":
    main()
```

### 8.2 複数 variation（model × prompt × action × scorer）

`generate_fns` を `partial(generate_with_variant, variant=variant)` で構成し、
各 variant.label を別アクションとして渡す。詳細な variation 設計は「5. variation 設計」を参照。

```python
from functools import partial

generate_fns = {
    variant.label: partial(generate_with_variant, variant=variant)
    for variant in VARIANTS
}

algo = tq.ABMCTSM()
search_tree = algo.init_tree()
for _ in range(budget):
    search_tree = algo.step(search_tree, generate_fns)
```

各 variant.label は TreeQuest から見て独立した探索アクションになる。

```
o4-mini        × direct     × sample × exact
gemini-2.5-pro × structured × sample × heuristic
deepseek-r1    × reasoning  × sample × consistency
o4-mini        × reasoning  × refine × exact
o4-mini        × critic     × repair × heuristic
gemini-2.5-pro × structured × refine × consistency
gemini-2.5-pro × critic     × repair × exact
```

---

## 9. 用語集

| 用語 | 意味 |
| --- | --- |
| GEN ノード | 「新しい子を生成する」行動を表す仮想ノード。wider 選択に対応 |
| wider / deeper | 幅方向（新規候補）/ 深さ方向（既存候補の深掘り） |
| mixed-effects model | アクション間（between-group）とアクション内（within-group）の分散を分離するモデル |
| variation / action | `generate_fns` の各キー。model × prompt × action × scorer の組み合わせ |
| score | TreeQuest に返す `[0, 1]` の評価値。探索の比較基準 |
| budget | 生成予算（追加するノード数の上限） |

---

## 参考

- TreeQuest: <https://github.com/SakanaAI/treequest>
- AB-MCTS（Sakana AI）: Adaptive Branching Monte Carlo Tree Search
