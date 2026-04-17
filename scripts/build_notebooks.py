from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import nbformat as nbf


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_eda_notebook() -> nbf.NotebookNode:
    summary = read_json(ROOT / "eda_summary.json")
    runtime_summary = read_json(ROOT / "artifacts" / "reports" / "eda_runtime_summary.json")
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# EDA: Ecommerce Recommender\n"
            "Исследование ориентировано на выбор memory-safe стратегии рекомендаций для задачи оптимизации `addtocart`."
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Ключевые выводы\n"
            "- данные событийные и требуют `time-based split`\n"
            "- основной позитивный сигнал: `addtocart`, более сильный дополнительный: `transaction`\n"
            "- распределение активности длиннохвостое, поэтому система должна иметь fallback для короткой истории\n"
            "- item properties полезны точечно, без полного расплющивания в wide table"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "ROOT = Path('..').resolve()\n"
            "summary = json.loads((ROOT / 'eda_summary.json').read_text(encoding='utf-8'))\n"
            "runtime_summary_path = ROOT / 'artifacts' / 'reports' / 'eda_runtime_summary.json'\n"
            "runtime_summary = json.loads(runtime_summary_path.read_text(encoding='utf-8')) if runtime_summary_path.exists() else {}\n"
            "summary, runtime_summary"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "pd.DataFrame([\n"
            "    {'metric': 'events_rows', 'value': summary['tables']['events_rows']},\n"
            "    {'metric': 'users', 'value': summary['uniques']['users']},\n"
            "    {'metric': 'items_in_events', 'value': summary['uniques']['items_in_events']},\n"
            "    {'metric': 'item_properties_rows', 'value': summary['tables']['item_properties_rows']},\n"
            "])"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "event_share = pd.Series(summary['event_shares_pct']).sort_values(ascending=False)\n"
            "plt.figure(figsize=(6, 4))\n"
            "sns.barplot(x=event_share.index, y=event_share.values, palette='crest')\n"
            "plt.title('Event share, %')\n"
            "plt.xlabel('event')\n"
            "plt.ylabel('share %')\n"
            "plt.tight_layout()"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Интерпретация для моделирования\n"
            "Высокая доля `view` при редких `addtocart` и `transaction` означает, что просмотры используются как контекст и сигнал для генерации кандидатов, но целевая оптимизация должна идти по событиям с более высокой ценностью."
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "implications = pd.DataFrame({'implication': summary['implications']})\n"
            "implications"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Memory-safe правила пайплайна\n"
            "- загрузка только нужных колонок и downcasting типов\n"
            "- отсутствие full join `events x item_properties`\n"
            "- агрегация до уровня `user-item` перед построением item-to-item связей\n"
            "- использование parquet для промежуточных артефактов"
        )
    )
    if runtime_summary:
        cells.append(
            nbf.v4.new_code_cell(
                "pd.DataFrame(list(runtime_summary.items()), columns=['metric', 'value'])"
            )
        )

    nb["cells"] = cells
    return nb


def build_modeling_notebook() -> nbf.NotebookNode:
    evaluation = read_json(ROOT / "artifacts" / "reports" / "evaluation_summary.json")
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            "# Modeling And Experiments\n"
            "Ноутбук фиксирует экспериментальный контур, `time-based` разбиение и сравнение baseline-моделей с итоговым retrieval-подходом."
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "\n"
            "ROOT = Path('..').resolve()\n"
            "summary_path = ROOT / 'artifacts' / 'reports' / 'evaluation_summary.json'\n"
            "summary = json.loads(summary_path.read_text(encoding='utf-8'))\n"
            "pd.DataFrame(summary['all_metrics']).T.sort_values('recall_at_k', ascending=False)"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Что сравнивалось\n"
            "- `global_popularity`: популярные товары по train-окну\n"
            "- `history_baseline`: персональная история с fallback на popularity\n"
            "- `weighted_item2item`: итоговая модель на агрегированном item-to-item co-occurrence"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "eval_frame = pd.read_parquet(ROOT / 'artifacts' / 'data' / 'evaluation_frame.parquet')\n"
            "eval_frame.head()"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Итог\n"
            f"Лучшая модель: `{evaluation.get('best_model_name', 'n/a')}`."
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "metrics = pd.DataFrame(summary['all_metrics']).T\n"
            "metrics[['recall_at_k', 'hit_rate_at_k', 'ndcg_at_k']].sort_values('recall_at_k', ascending=False)"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "Итоговый подход выбран как наиболее сильный по `Recall@K` при умеренных вычислительных затратах и без перехода к тяжелым dense-представлениям."
        )
    )
    nb["cells"] = cells
    return nb


def main() -> None:
    notebooks_dir = ROOT / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    nbf.write(build_eda_notebook(), notebooks_dir / "01_eda.ipynb")
    nbf.write(build_modeling_notebook(), notebooks_dir / "02_modeling.ipynb")


if __name__ == "__main__":
    main()
