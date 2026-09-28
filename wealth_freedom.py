#!/usr/bin/env python3
"""财富自由公式计算器。

命令行 + 可视化。所有金额均以今天的购买力（实际口径）展示：
用户输入名义参数，程序内部按通胀折算为实际收益率后复利。

双击运行（不带命令行参数）→ 逐项问答；带参数运行 → 命令行直通。

公式：
    实际年收益率   r    = (1 + 名义收益率) / (1 + 通胀率) - 1
    实际收入增长率 g    = (1 + 名义收入增长率) / (1 + 通胀率) - 1
    月利率         r_m  = (1 + r)^(1/12) - 1
    自由门槛       T    = 年支出 / 提取率
    第 t 年年储蓄       = 年储蓄0 * (1 + g)^t
    积累期终值     FV   = P*(1+r_m)^N + PMT*((1+r_m)^N - 1)/r_m   # 仅 g = 0 时
    消耗期递推     A'   = A*(1+r) - 年支出

年支出支持两种输入：直接填年总支出（简式），或逐项填写明细（求和即年支出）。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
import unicodedata
from dataclasses import dataclass
from typing import NamedTuple

MONTHS_PER_YEAR = 12
MAX_YEARS = 100
MAX_MONTHS = MAX_YEARS * MONTHS_PER_YEAR
WAN = 10000

DEFAULT_RETURN = 8.0
DEFAULT_INFLATION = 3.0
DEFAULT_WITHDRAWAL = 4.0
DEFAULT_LIFE_EXPECTANCY = 85

CHART_FILENAME = "wealth_freedom.png"

ASSERT_TOLERANCE = 1e-6

# 支出明细目录：(类目, 条目, 默认频率, 全国基准参考值/元)。
# 基准值仅为量级估算，不来自任何官方统计口径，只用于降低填写门槛。
EXPENSE_CATALOG = (
    ("住房", "房租或房贷月供", "月", 2000.0),
    ("住房", "物业水电燃气", "月", 400.0),
    ("餐饮", "日常吃饭", "月", 1800.0),
    ("餐饮", "外出聚餐", "月", 400.0),
    ("交通", "通勤", "月", 300.0),
    ("交通", "车辆油费保养保险", "年", 12000.0),
    ("购物", "衣物", "月", 500.0),
    ("购物", "洗护日用品", "月", 200.0),
    ("购物", "电子产品", "年", 4000.0),
    ("健康", "保险", "年", 6000.0),
    ("健康", "体检与常备药", "月", 200.0),
    ("旅行", "年度旅行预算", "年", 10000.0),
    ("其他", "人情往来", "年", 5000.0),
    ("其他", "订阅与娱乐", "月", 200.0),
    ("其他", "教育", "年", 6000.0),
)

# 城市档乘数（相对全国基准 1.0），同样为粗略估算，只影响明细模式的参考值。
CITY_TIERS = {
    "一线": 1.3,
    "新一线": 1.0,
    "二线": 0.85,
    "三四线": 0.7,
}

# 问答时城市档的展示文案。
CITY_PROMPTS = {
    "一线": "一线（北上广深）",
    "新一线": "新一线",
    "二线": "二线",
    "三四线": "三四线及以下",
}


class TargetUnreachable(Exception):
    """在当前参数下无法达成财富自由。"""


class ExpenseItem(NamedTuple):
    """一条分项支出。"""

    category: str      # 类目
    item: str          # 条目
    amount: float      # 金额（元）
    period: str        # 频率：「月」或「年」


@dataclass(frozen=True)
class Context:
    """一次计算的全部口径参数。"""

    age: int                  # 当前年龄（岁）
    life_expectancy: int      # 预期寿命（岁）
    nominal: float            # 名义年化收益率 %
    inflation: float          # 通胀率 %
    withdrawal: float         # 安全提取率 %
    real: float               # 实际年收益率（小数）
    r_month: float            # 实际月收益率（小数）
    annual_income: float      # 年收入（元），未知时为 None
    annual_expense: float     # 年支出（元）
    annual_save: float        # 第 0 年年储蓄（元）
    principal: float          # 现有可投资资产（元）
    target: float             # 自由门槛（元）
    goal_years: float         # 目标年限，仅 --years 显式给出时非 None
    income_growth: float      # 名义年收入增长率 %
    real_income_growth: float # 实际年收入增长率（小数）
    expense_items: tuple      # 分项支出列表，简式模式为 None
    city_tier: str            # 城市档，命令行或简式模式为 None

    @property
    def monthly_save(self):
        """第 0 年的每月储蓄（元）。"""
        return self.annual_save / MONTHS_PER_YEAR


class SensitivityRow(NamedTuple):
    """敏感度分析的一行。"""

    name: str          # 参数名
    change: str        # 扰动幅度
    age: float         # 扰动后的达成年龄，无法达标时为 None
    delta: float       # 相对基准的年数变化，无法比较时为 None
    adverse: str       # 该扰动为不利时的完整描述


# --------------------------------------------------------------------------
# 参数层
# --------------------------------------------------------------------------


def non_negative(text):
    """argparse 类型：非负浮点数。"""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"不是合法数字：{text}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"不能为负数：{text}")
    return value


def positive(text):
    """argparse 类型：正浮点数。"""
    value = non_negative(text)
    if value == 0:
        raise argparse.ArgumentTypeError(f"必须大于 0：{text}")
    return value


def positive_int(text):
    """argparse 类型：正整数。"""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"不是合法整数：{text}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"必须为正整数：{text}")
    return value


def rate_list(text):
    """argparse 类型：逗号分隔的收益率列表。"""
    try:
        rates = [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"不是合法的收益率列表：{text}") from None
    if not rates:
        raise argparse.ArgumentTypeError(f"收益率列表为空：{text}")
    if any(rate < 0 for rate in rates):
        raise argparse.ArgumentTypeError("收益率不能为负数")
    return rates


def growth_rate(text):
    """argparse 类型：名义收入增长率 %，允许为负但不能到 -100%。"""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"不是合法数字：{text}") from None
    if value <= -100:
        raise argparse.ArgumentTypeError(f"必须大于 -100：{text}")
    return value


def expense_item(text):
    """argparse 类型：分项支出，格式 `类目:条目:金额:频率`（金额单位：元）。"""
    parts = text.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"格式应为 类目:条目:金额:频率（金额单位元），收到：{text}"
        )
    category, name, amount_text, period = (part.strip() for part in parts)
    if not category or not name:
        raise argparse.ArgumentTypeError(f"类目与条目不能为空：{text}")
    if period not in ("月", "年"):
        raise argparse.ArgumentTypeError(f"频率只能是「月」或「年」，收到：{period}")
    try:
        amount = float(amount_text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"金额不是合法数字：{amount_text}") from None
    if amount < 0:
        raise argparse.ArgumentTypeError(f"金额不能为负数：{amount_text}")
    return ExpenseItem(category, name, amount, period)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="wealth_freedom.py",
        description="财富自由公式计算器（所有金额均为今天的购买力）",
    )
    parser.add_argument(
        "--age", type=positive_int, default=None, metavar="岁",
        help="当前年龄（必填）",
    )
    parser.add_argument(
        "--annual-income", type=non_negative, default=None, metavar="万",
        help="年收入，单位万元；给出后年储蓄 = 年收入 - 年支出",
    )
    parser.add_argument(
        "--annual-expense", type=non_negative, default=None, metavar="万",
        help="年支出，单位万元（未给 --expense-item 时必填）",
    )
    parser.add_argument(
        "--expense-item", type=expense_item, action="append", default=None,
        metavar="类目:条目:金额:频率",
        help="分项支出，金额单位元，可重复；给出后年支出 = 各项年化之和",
    )
    parser.add_argument(
        "--assets", type=non_negative, default=0.0, metavar="万",
        help="现有可投资资产，单位万元（默认 0）",
    )
    parser.add_argument(
        "--monthly-save", type=non_negative, default=0.0, metavar="元",
        help="每月可存金额，单位元；未给 --annual-income 时用它折算年储蓄",
    )
    parser.add_argument(
        "--life-expectancy", type=positive_int, default=DEFAULT_LIFE_EXPECTANCY,
        metavar="岁", help=f"预期寿命（默认 {DEFAULT_LIFE_EXPECTANCY}）",
    )
    parser.add_argument(
        "--return", dest="return_rate", type=non_negative, default=DEFAULT_RETURN,
        metavar="%", help=f"名义年化收益率 %% （默认 {DEFAULT_RETURN}）",
    )
    parser.add_argument(
        "--inflation", type=non_negative, default=DEFAULT_INFLATION, metavar="%",
        help=f"通胀率 %% （默认 {DEFAULT_INFLATION}）",
    )
    parser.add_argument(
        "--income-growth", type=growth_rate, default=None, metavar="%",
        help="名义年收入增长率 %%；默认取通胀率（实际不涨不减），可填 0 或负数",
    )
    parser.add_argument(
        "--withdrawal", type=positive, default=DEFAULT_WITHDRAWAL, metavar="%",
        help=f"安全提取率 %% （默认 {DEFAULT_WITHDRAWAL}）",
    )
    parser.add_argument(
        "--years", dest="goal_years", type=positive, default=None, metavar="年",
        help="目标年限，给出后在结论中反推所需储蓄",
    )
    parser.add_argument(
        "--scenario", type=rate_list, default=None, metavar="6,8,10",
        help="情景对比的名义收益率列表，逗号分隔",
    )
    parser.add_argument(
        "--no-table", action="store_true", help="不打印逐年明细",
    )
    parser.add_argument(
        "--no-chart", action="store_true", help="不生成折线图",
    )
    parser.add_argument(
        "--selftest", action="store_true", help="运行内置校验用例后退出",
    )
    parser.add_argument(
        "--no-pause", action="store_true",
        help="双击启动时也不等待回车（脚本化调用时使用）",
    )
    args = parser.parse_args(argv)
    if args.selftest:
        return args

    if args.expense_item and args.annual_expense is not None:
        parser.error("不能同时指定 --expense-item 与 --annual-expense")

    missing = []
    if args.age is None:
        missing.append("--age")
    if args.annual_expense is None and not args.expense_item:
        missing.append("--annual-expense")
    if missing:
        parser.error("缺少必填参数：" + "、".join(missing))
    return args


def params_from_args(args):
    """把命令行参数整理成统一的参数字典。"""
    items = args.expense_item
    if items:
        annual_expense = expense_items_total(items)
        if annual_expense <= 0:
            raise ValueError("分项支出合计为 0，请至少填写一项")
    else:
        annual_expense = args.annual_expense * WAN

    if args.annual_income is not None:
        annual_income = args.annual_income * WAN
        annual_save = annual_income - annual_expense
    else:
        annual_income = None
        annual_save = args.monthly_save * MONTHS_PER_YEAR

    income_growth = args.inflation if args.income_growth is None else args.income_growth

    return {
        "age": args.age,
        "principal": args.assets * WAN,
        "annual_income": annual_income,
        "annual_expense": annual_expense,
        "annual_save": annual_save,
        "nominal": args.return_rate,
        "inflation": args.inflation,
        "withdrawal": args.withdrawal,
        "income_growth": income_growth,
        "life_expectancy": args.life_expectancy,
        "goal_years": args.goal_years,
        "expense_items": tuple(items) if items else None,
        "city_tier": None,
    }


def build_context(params):
    """校验参数字典并构造 Context；不合法时抛 ValueError。"""
    if params["life_expectancy"] <= params["age"]:
        raise ValueError("预期寿命必须大于当前年龄")
    if params["annual_save"] < 0:
        raise ValueError("入不敷出，年储蓄为负，无法积累")
    if params["income_growth"] <= -100:
        raise ValueError("收入增长率必须大于 -100%")

    real = real_rate(params["nominal"], params["inflation"])
    real_g = real_income_growth(params["income_growth"], params["inflation"])
    return Context(
        age=params["age"],
        life_expectancy=params["life_expectancy"],
        nominal=params["nominal"],
        inflation=params["inflation"],
        withdrawal=params["withdrawal"],
        real=real,
        r_month=monthly_rate(real),
        annual_income=params["annual_income"],
        annual_expense=params["annual_expense"],
        annual_save=params["annual_save"],
        principal=params["principal"],
        target=fire_target(params["annual_expense"], params["withdrawal"]),
        goal_years=params["goal_years"],
        income_growth=params["income_growth"],
        real_income_growth=real_g,
        expense_items=params.get("expense_items"),
        city_tier=params.get("city_tier"),
    )


def ask_int(prompt, default=None, minimum=None, hint=None):
    """交互式读取整数；有默认值时直接回车采用默认值。"""
    while True:
        suffix = f"（默认 {default}）" if default is not None else ""
        raw = input(f"{prompt}{suffix}：").strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("  请输入整数。")
            continue
        if minimum is not None and value < minimum:
            print(f"  {hint or f'不能小于 {minimum}。'}")
            continue
        return value


def ask_float(prompt, default=None, minimum=None, maximum=None, hint=None):
    """交互式读取浮点数；有默认值时直接回车采用默认值。"""
    while True:
        suffix = f"（默认 {default}）" if default is not None else ""
        raw = input(f"{prompt}{suffix}：").strip()
        if not raw and default is not None:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  请输入数字。")
            continue
        if minimum is not None and value < minimum:
            print(f"  {hint or f'不能小于 {minimum}。'}")
            continue
        if maximum is not None and value > maximum:
            print(f"  {hint or f'不能大于 {maximum}。'}")
            continue
        return value


def ask_choice(prompt, options, default):
    """选项问答：options 为 [(键, 说明)]，回车采用 default，返回所选键。"""
    for key, label in options:
        print(f"  {key}) {label}")
    keys = "/".join(key for key, _ in options)
    while True:
        raw = input(f"{prompt}（默认 {default}）：").strip()
        if not raw:
            return default
        if any(raw == key for key, _ in options):
            return raw
        print(f"  请输入 {keys} 之一。")


def ask_expense_items(multiplier):
    """逐项询问明细支出，返回 ExpenseItem 列表；直接回车表示该项为 0。"""
    print()
    print(f"逐项填写支出，直接回车表示该项为 0（金额单位：元，"
          f"参考值 = 全国基准 × {multiplier}）。")
    items = []
    last_category = None
    for category, name, period, baseline in EXPENSE_CATALOG:
        if category != last_category:
            print(f"[{category}]")
            last_category = category
        unit = "元/月" if period == "月" else "元/年"
        amount = ask_float(
            f"  {name}（{unit}，参考 {baseline * multiplier:.0f}）",
            default=0.0, minimum=0.0,
        )
        if amount > 0:
            items.append(ExpenseItem(category, name, amount, period))
    print()
    return items


def ask_expense(income_yuan):
    """询问支出模式，返回 (年支出元, 明细列表或 None, 城市档或 None)。"""
    while True:
        mode = ask_choice(
            "支出模式",
            [("1", "简式：直接填写年总支出"),
             ("2", "明细：按类目逐项填写各项支出")],
            default="1",
        )
        if mode == "1":
            expense_wan = ask_float(
                "年支出（万元）", minimum=0.0, maximum=income_yuan / WAN,
                hint="年支出不能大于年收入（入不敷出，无法积累）。",
            )
            return expense_wan * WAN, None, None

        tiers = list(CITY_TIERS)
        tier = ask_choice(
            "城市档",
            [(str(index + 1), CITY_PROMPTS[key]) for index, key in enumerate(tiers)],
            default="2",
        )
        tier = tiers[int(tier) - 1]
        items = ask_expense_items(CITY_TIERS[tier])
        total = expense_items_total(items)
        if total <= 0:
            print("  明细合计为 0，请至少填写一项或选择简式。\n")
            continue
        if total > income_yuan:
            print("  明细合计已超过年收入（入不敷出，无法积累），请重新填写。\n")
            continue
        return total, tuple(items), tier


def ask_params():
    """双击运行时逐项问答，返回与命令行路径一致的参数字典。"""
    print("=== 财富自由推演 ===")
    print("带默认值的项直接回车即可采用默认值。\n")

    age = ask_int("当前年龄（岁）", minimum=1)
    asset_wan = ask_float("当前资产（万元）", minimum=0.0)
    income_wan = ask_float("年收入（万元）", minimum=0.0)
    income_yuan = income_wan * WAN

    expense_yuan, items, tier = ask_expense(income_yuan)

    nominal = ask_float("名义年化收益率（%）", default=DEFAULT_RETURN, minimum=0.0)
    inflation = ask_float("通胀率（%）", default=DEFAULT_INFLATION, minimum=0.0)
    withdrawal = ask_float("安全提取率（%）", default=DEFAULT_WITHDRAWAL, minimum=0.1)
    income_growth = ask_float(
        "名义年收入增长率（%）", default=inflation, minimum=-99.99,
        hint="收入增长率必须大于 -100%。",
    )
    life = ask_int(
        "预期寿命（岁）", default=DEFAULT_LIFE_EXPECTANCY, minimum=age + 1,
        hint=f"预期寿命必须大于当前年龄 {age} 岁。",
    )
    print()

    return {
        "age": age,
        "principal": asset_wan * WAN,
        "annual_income": income_yuan,
        "annual_expense": expense_yuan,
        "annual_save": income_yuan - expense_yuan,
        "nominal": nominal,
        "inflation": inflation,
        "withdrawal": withdrawal,
        "income_growth": income_growth,
        "life_expectancy": life,
        "goal_years": None,
        "expense_items": items,
        "city_tier": tier,
    }


# --------------------------------------------------------------------------
# 计算层（纯函数，无 I/O）
# --------------------------------------------------------------------------


def real_rate(nominal_pct, inflation_pct):
    """名义收益率折算为实际收益率（小数）。"""
    return (1 + nominal_pct / 100) / (1 + inflation_pct / 100) - 1


def real_income_growth(nominal_growth_pct, inflation_pct):
    """名义收入增长率折算为实际收入增长率（小数）。"""
    return (1 + nominal_growth_pct / 100) / (1 + inflation_pct / 100) - 1


def monthly_rate(annual_real):
    """实际年收益率折算为实际月收益率（小数）。"""
    return (1 + annual_real) ** (1 / MONTHS_PER_YEAR) - 1


def fire_target(annual_expense, withdrawal_pct):
    """自由门槛：年支出 / 安全提取率。"""
    return annual_expense / (withdrawal_pct / 100)


def saving_rate(annual_income, annual_expense):
    """储蓄率 = 年储蓄 / 年收入（小数）；年收入未知或为 0 时返回 None。"""
    if not annual_income:
        return None
    return (annual_income - annual_expense) / annual_income


def annualize(amount, period):
    """把按月/年记录的金额折算为年化金额（元）。"""
    return amount * MONTHS_PER_YEAR if period == "月" else amount


def expense_items_total(items):
    """分项支出的年化金额之和（元）。"""
    return sum(annualize(item.amount, item.period) for item in items)


def expense_breakdown(items):
    """分项明细，按年化金额降序，返回 [(类目, 条目, 年化金额, 占比)]。"""
    total = expense_items_total(items)
    rows = sorted(
        ((item.category, item.item, annualize(item.amount, item.period)) for item in items),
        key=lambda row: row[2],
        reverse=True,
    )
    return [(category, name, amount, amount / total if total else 0.0)
            for category, name, amount in rows]


def baseline_expense(catalog_row, multiplier):
    """参考值 = 条目全国基准 × 城市乘数（元）。"""
    return catalog_row[3] * multiplier


def months_to_target(principal, monthly_save, target, r_month):
    """闭式解：达到目标所需月数。

    N = ln((T*r_m + PMT) / (P*r_m + PMT)) / ln(1 + r_m)
    """
    if principal >= target:
        return 0.0

    if r_month == 0.0:
        if monthly_save <= 0:
            raise TargetUnreachable("无收益且无储蓄")
        months = (target - principal) / monthly_save
    else:
        if r_month <= -1.0:
            raise TargetUnreachable("实际收益率低于 -100%")
        denominator = principal * r_month + monthly_save
        if denominator == 0.0:
            raise TargetUnreachable("储蓄不足以抵消购买力缩水")
        growth = (target * r_month + monthly_save) / denominator
        if growth <= 0.0:
            raise TargetUnreachable("储蓄不足以抵消购买力缩水")
        months = math.log(growth) / math.log1p(r_month)

    if not math.isfinite(months) or months <= 0.0:
        raise TargetUnreachable("储蓄不足以抵消购买力缩水")
    if months > MAX_MONTHS:
        raise TargetUnreachable(f"所需时间超过 {MAX_YEARS} 年")
    return months


def required_monthly_saving(principal, target, months, r_month):
    """闭式解：给定月数达成目标所需的每月储蓄。

    PMT = (T - P*(1+r_m)^N) * r_m / ((1+r_m)^N - 1)
    """
    if principal >= target:
        return 0.0
    if months <= 0:
        return math.inf
    if r_month == 0.0:
        return (target - principal) / months

    growth = (1.0 + r_month) ** months
    denominator = growth - 1.0
    if denominator == 0.0:
        return (target - principal) / months
    return max(0.0, (target - principal * growth) * r_month / denominator)


def simulate_months(principal, monthly_save, r_month, months):
    """逐月推进（月末定投），返回每月末的资产列表。"""
    series = []
    assets = principal
    for _ in range(months):
        assets += assets * r_month + monthly_save
        series.append(assets)
    return series


def accumulate_one_year(assets, monthly_save, r_month):
    """积累一年（月末定投），返回 (年末资产, 当年收益)。"""
    earned = 0.0
    for _ in range(MONTHS_PER_YEAR):
        interest = assets * r_month
        assets += interest + monthly_save
        earned += interest
    return assets, earned


def annual_save_at(ctx, year_index):
    """第 year_index 年（从 0 起）的年储蓄（元），按实际收入增长率逐年复利。"""
    return ctx.annual_save * (1 + ctx.real_income_growth) ** year_index


def reach_age(ctx, nominal_pct=None, inflation_pct=None, annual_save=None,
              income_growth_pct=None):
    """给定（可覆盖的）参数下的达成年龄；无法达标返回 None。

    实际收入增长率为 0 时走闭式解；不为 0 时储蓄逐年变化，闭式解失效，
    改为逐月迭代（月储蓄按年阶梯上调）。
    """
    if nominal_pct is None:
        nominal_pct = ctx.nominal
    if inflation_pct is None:
        inflation_pct = ctx.inflation
    if annual_save is None:
        annual_save = ctx.annual_save
    if income_growth_pct is None:
        income_growth_pct = ctx.income_growth

    r_month = monthly_rate(real_rate(nominal_pct, inflation_pct))
    growth = real_income_growth(income_growth_pct, inflation_pct)

    if abs(growth) < ASSERT_TOLERANCE:
        try:
            months = months_to_target(
                ctx.principal, annual_save / MONTHS_PER_YEAR, ctx.target, r_month
            )
        except TargetUnreachable:
            return None
        return ctx.age + months / MONTHS_PER_YEAR

    if ctx.principal >= ctx.target:
        return ctx.age
    assets = ctx.principal
    for month in range(1, MAX_MONTHS + 1):
        year_index = (month - 1) // MONTHS_PER_YEAR
        monthly_save = annual_save * (1 + growth) ** year_index / MONTHS_PER_YEAR
        assets += assets * r_month + monthly_save
        if assets >= ctx.target:
            return ctx.age + month / MONTHS_PER_YEAR
    return None


def build_lifeline(ctx):
    """按整数年龄逐年推演一生。

    返回 [(年龄, 年末资产, 当年储蓄, 当年支取, 当年收益, 阶段)]，
    年龄从「当前年龄 + 1」推进到「预期寿命」；资产耗尽则提前结束。
    未退休 → 积累期（按月复利、按月储蓄），首次跨越门槛那一年标「达成」；
    退休不可逆 → 退休期（按年复利、按年支取），资产回落到门槛以下仍继续支取。
    """
    rows = []
    assets = ctx.principal
    retired = ctx.principal >= ctx.target
    for age in range(ctx.age + 1, ctx.life_expectancy + 1):
        if retired:
            earned = assets * ctx.real
            withdraw = ctx.annual_expense
            assets = assets + earned - withdraw
            saved = 0.0
            phase = "退休"
        else:
            saved = annual_save_at(ctx, age - ctx.age - 1)
            assets, earned = accumulate_one_year(
                assets, saved / MONTHS_PER_YEAR, ctx.r_month
            )
            withdraw = 0.0
            if assets >= ctx.target:
                phase = "达成"
                retired = True
            else:
                phase = "积累"
        rows.append((age, assets, saved, withdraw, earned, phase))
        if assets < 0:
            break
    return rows


def accumulation_trajectory(ctx, nominal_pct):
    """给定名义收益率的积累期逐年轨迹 [(年龄, 年末资产)]，达标即止。"""
    if ctx.principal >= ctx.target:
        return []

    r_month = monthly_rate(real_rate(nominal_pct, ctx.inflation))
    rows = []
    assets = ctx.principal
    for age in range(ctx.age + 1, ctx.life_expectancy + 1):
        saved = annual_save_at(ctx, age - ctx.age - 1)
        assets, _ = accumulate_one_year(assets, saved / MONTHS_PER_YEAR, r_month)
        rows.append((age, assets))
        if assets >= ctx.target:
            break
    return rows


def depletion_age(rows):
    """资产首次转负的年龄；未耗尽返回 None。"""
    for age, assets, *_ in rows:
        if assets < 0:
            return age
    return None


def rate_scenarios(ctx, rates):
    """不同名义收益率下的达成年龄，返回 [(名义%, 实际%, 达成年龄 or None)]。"""
    return [
        (rate, real_rate(rate, ctx.inflation), reach_age(ctx, nominal_pct=rate))
        for rate in rates
    ]


def sensitivity(ctx):
    """参数扰动对达成年龄的影响，返回 SensitivityRow 列表。"""
    base = reach_age(ctx)
    cases = [
        ("名义收益率", "+1.0%", {"nominal_pct": ctx.nominal + 1.0}, None),
        ("名义收益率", "-1.0%", {"nominal_pct": ctx.nominal - 1.0},
         "名义收益率比预期低 1 个百分点"),
        ("年储蓄", "+10%", {"annual_save": ctx.annual_save * 1.1}, None),
        ("年储蓄", "-10%", {"annual_save": ctx.annual_save * 0.9},
         "年储蓄比预期低 10%"),
        ("收入增长率", "+1.0%", {"income_growth_pct": ctx.income_growth + 1.0}, None),
        ("收入增长率", "-1.0%", {"income_growth_pct": ctx.income_growth - 1.0},
         "收入增长率比预期低 1 个百分点"),
        ("通胀率", "+1.0%", {"inflation_pct": ctx.inflation + 1.0},
         "通胀率比预期高 1 个百分点"),
    ]

    rows = []
    for name, change, kwargs, adverse in cases:
        age = reach_age(ctx, **kwargs)
        delta = None if (age is None or base is None) else age - base
        rows.append(SensitivityRow(name, change, age, delta, adverse))
    return rows


def cross_check(principal, monthly_save, target, r_month, months_exact):
    """用逐月迭代校验闭式解，不一致直接抛断言错误。"""
    if months_exact <= 0.0:
        if principal < target:
            raise AssertionError("闭式解为 0 个月，但资产未达门槛")
        return

    months = math.ceil(months_exact)
    series = simulate_months(principal, monthly_save, r_month, months)
    tolerance = max(abs(target), 1.0) * ASSERT_TOLERANCE
    if series[-1] < target - tolerance:
        raise AssertionError(f"闭式解 {months_exact:.6f} 个月，但第 {months} 个月末仍未达标")
    if len(series) >= 2 and series[-2] >= target + tolerance:
        raise AssertionError(
            f"闭式解 {months_exact:.6f} 个月，但第 {months - 1} 个月末已提前达标"
        )


def cross_check_lifeline(ctx, rows, free_age):
    """校验达成年龄落在逐年明细表标记「达成」的那一年内。"""
    if free_age is None or free_age >= ctx.life_expectancy:
        return
    if ctx.principal >= ctx.target:
        return
    for age, _assets, _saved, _withdraw, _earned, phase in rows:
        if phase == "达成":
            if not age - 1 < free_age <= age:
                raise AssertionError(
                    f"达成年龄 {free_age:.4f} 岁与逐年表的达成年 {age} 岁不符"
                )
            return
    raise AssertionError(f"达成年龄 {free_age:.4f} 岁，但逐年表未标记达成年")


# --------------------------------------------------------------------------
# 展示层（只排版，不计算）
# --------------------------------------------------------------------------


def display_width(text):
    """按终端显示宽度计算字符串长度，中日韩字符按 2 列计。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text, width):
    return text + " " * max(0, width - display_width(text))


def format_wan(yuan):
    return f"{yuan / WAN:.1f}万"


def format_money(yuan):
    return f"{yuan:,.0f}"


def format_summary(ctx):
    multiple = 1 / (ctx.withdrawal / 100)
    rate = saving_rate(ctx.annual_income, ctx.annual_expense)

    if ctx.annual_income is None:
        saving_line = f"年支出 {format_wan(ctx.annual_expense)}"
        if ctx.annual_save:
            saving_line += f"  年储蓄 {format_wan(ctx.annual_save)}"
    else:
        saving_line = (
            f"年收入 {format_wan(ctx.annual_income)}  年支出 {format_wan(ctx.annual_expense)}"
            f"  →  年储蓄 {format_wan(ctx.annual_save)}"
        )
        if rate is not None:
            saving_line += f"  储蓄率 {rate * 100:.1f}%"

    lines = [
        "=== 口径 ===",
        f"年龄 {ctx.age} 岁  预期寿命 {ctx.life_expectancy} 岁",
        f"名义 {ctx.nominal:.1f}%  通胀 {ctx.inflation:.1f}%"
        f"  →  实际收益率 {ctx.real * 100:.3f}%",
        f"名义收入增长率 {ctx.income_growth:.1f}%"
        f"  →  实际收入增长率 {ctx.real_income_growth * 100:.3f}%",
        saving_line,
        f"年支出 {format_wan(ctx.annual_expense)} × (1/{ctx.withdrawal:.1f}%)"
        f" = {multiple:.1f} 倍  →  自由门槛 {format_wan(ctx.target)}（今天购买力）",
    ]
    if ctx.expense_items is not None:
        if ctx.city_tier:
            lines.append(
                f"支出明细模式：{ctx.city_tier}城市"
                f"（消费水平 ≈ 全国基准的 {CITY_TIERS[ctx.city_tier]:.1f} 倍，"
                f"参考值为粗略估算）"
            )
        else:
            lines.append("支出明细模式：命令行分项合计（参考值不参与计算）")
    return "\n".join(lines)


def format_expense_breakdown(rows):
    """分项支出表：按年化金额降序列出，末行为合计。"""
    total = sum(row[2] for row in rows)
    lines = [
        "=== 支出明细 ===",
        pad(" 年化金额(万)", 15) + pad("占比", 9) + pad("类目", 8) + "条目",
    ]
    for category, name, amount, ratio in rows:
        lines.append(
            pad(f"{amount / WAN:>9.1f}", 15) + pad(f"{ratio * 100:.1f}%", 9)
            + pad(category, 8) + name
        )
    lines.append(pad(f"{total / WAN:>9.1f}", 15) + pad("100.0%", 9) + "合计")
    return "\n".join(lines)


def format_conclusion(ctx, rows, sens):
    lines = ["=== 结论 ==="]
    free_age = reach_age(ctx)

    if free_age is None or free_age >= ctx.life_expectancy:
        lines.append(
            f"• 按实际收益率 {ctx.real * 100:.3f}% 推算，"
            f"在预期寿命 {ctx.life_expectancy} 岁内无法达成财富自由"
        )
    else:
        lines.append(f"• 按实际收益率 {ctx.real * 100:.3f}% 推算，预计 {free_age:.1f} 岁达成财富自由")
        lines.append(
            f"• 距离预期寿命 {ctx.life_expectancy} 岁还有 {ctx.life_expectancy - free_age:.1f} 年，"
            f"退休后每年支取 {format_wan(ctx.annual_expense)}"
        )

    depleted = depletion_age(rows)
    if depleted is None:
        if rows:
            last_age, last_assets = rows[-1][0], rows[-1][1]
            lines.append(f"• 推演到 {last_age} 岁资产仍有 {format_wan(last_assets)}，不会耗尽")
    else:
        lines.append(
            f"• 警告：资产将在 {depleted} 岁耗尽，"
            f"比预期寿命早 {ctx.life_expectancy - depleted} 岁"
        )

    worst = None
    for row in sens:
        if row.adverse and row.delta is not None and row.delta > 0:
            if worst is None or row.delta > worst.delta:
                worst = row
    if worst is not None:
        lines.append(f"• 若{worst.adverse}，达成时间推迟 {worst.delta:.1f} 年")

    if ctx.goal_years is not None:
        months = max(1, round(ctx.goal_years * MONTHS_PER_YEAR))
        needed = required_monthly_saving(ctx.principal, ctx.target, months, ctx.r_month)
        if not math.isfinite(needed):
            lines.append(f"• 目标 {ctx.goal_years:.1f} 年后自由：期限过短，无法实现")
        elif needed <= 0.0:
            lines.append(f"• 目标 {ctx.goal_years:.1f} 年后自由：现有资产已足够")
        else:
            lines.append(
                f"• 目标 {ctx.goal_years:.1f} 年后自由：每月至少需存 {format_money(needed)} 元"
            )
    return "\n".join(lines)


def format_sensitivity(ctx, rows):
    base = reach_age(ctx)
    base_text = "无法达标" if base is None else f"{base:.1f} 岁"
    lines = [
        "=== 敏感度分析 ===",
        pad("参数", 14) + pad("变化", 10) + pad("达成年龄", 12) + "相对基准",
        pad("基准", 14) + pad("—", 10) + pad(base_text, 12) + "—",
    ]
    for row in rows:
        age_text = "无法达标" if row.age is None else f"{row.age:.1f} 岁"
        delta_text = "—" if row.delta is None else f"{row.delta:+.1f} 年"
        lines.append(
            pad(row.name, 14) + pad(row.change, 10) + pad(age_text, 12) + delta_text
        )
    return "\n".join(lines)


def format_scenarios(rows):
    lines = [
        "=== 情景对比 ===",
        pad("名义收益率", 14) + pad("实际收益率", 14) + "达成年龄",
    ]
    for rate, real, age in rows:
        age_text = "无法达标" if age is None else f"{age:.1f} 岁"
        lines.append(pad(f"{rate:.1f}%", 14) + pad(f"{real * 100:.3f}%", 14) + age_text)
    return "\n".join(lines)


def format_lifeline(rows):
    lines = [
        "=== 逐年明细 ===",
        pad("年龄", 8) + pad("年末资产(万)", 16) + pad("当年储蓄(万)", 16)
        + pad("当年支取(万)", 16) + pad("当年收益(万)", 16) + "阶段",
    ]
    for age, assets, saved, withdraw, earned, phase in rows:
        lines.append(
            pad(str(age), 8) + pad(format_wan(assets), 16) + pad(format_wan(saved), 16)
            + pad(format_wan(withdraw), 16) + pad(format_wan(earned), 16) + phase
        )
    return "\n".join(lines)


def chart_path():
    if "__file__" in globals():
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        base_dir = os.getcwd()
    return os.path.join(base_dir, CHART_FILENAME)


def draw_chart(ctx, rows, rates, path):
    """绘制生命周期折线图，保存 PNG 并弹窗显示。"""
    try:
        import matplotlib

        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[提示] 未安装 matplotlib，已跳过折线图"
              "（可运行 pip install -r requirements.txt 安装）")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        [row[0] for row in rows], [row[1] / WAN for row in rows],
        color="#0F7B4A", linewidth=2, label="基准情形资产",
    )

    for rate in rates:
        trajectory = accumulation_trajectory(ctx, rate)
        if trajectory:
            ax.plot(
                [item[0] for item in trajectory], [item[1] / WAN for item in trajectory],
                linewidth=1.2, linestyle="--", label=f"名义 {rate:.1f}%",
            )

    ax.axhline(ctx.target / WAN, color="#888888", linestyle=":", linewidth=1,
               label=f"自由门槛 {format_wan(ctx.target)}")
    free_age = reach_age(ctx)
    if free_age is not None and free_age <= ctx.life_expectancy:
        ax.axvline(free_age, color="#888888", linestyle=":", linewidth=1,
                   label=f"达成年龄 {free_age:.1f} 岁")

    ax.set_xlabel("年龄（岁）")
    ax.set_ylabel("资产（万元，今天购买力）")
    ax.set_title("财富自由推演")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\n折线图已保存：{path}")

    try:
        plt.show()
    except Exception:
        print("[提示] 当前环境无法弹窗显示，请直接查看已保存的 PNG")


# --------------------------------------------------------------------------
# 内置校验
# --------------------------------------------------------------------------


def sample_params():
    """设计文档中的样例输入（年龄 30、资产 20万、收入 15万、支出 6万）。"""
    return {
        "age": 30,
        "principal": 200_000.0,
        "annual_income": 150_000.0,
        "annual_expense": 60_000.0,
        "annual_save": 90_000.0,
        "nominal": DEFAULT_RETURN,
        "inflation": DEFAULT_INFLATION,
        "withdrawal": DEFAULT_WITHDRAWAL,
        "income_growth": DEFAULT_INFLATION,
        "life_expectancy": 85,
        "goal_years": None,
        "expense_items": None,
        "city_tier": None,
    }


def run_selftest():
    cases = []

    # 1. 零收益退化为纯除法：无收益时 N = (T - P) / PMT
    months = months_to_target(100_000.0, 1_000.0, 220_000.0, 0.0)
    cases.append(("零收益退化", abs(months - 120.0) < 1e-9, f"{months:.6f} 个月"))

    # 2. 已知答案定投题：终值反解回原月数
    r_month = monthly_rate(0.06)
    target = 1_000.0 * (((1 + r_month) ** 120 - 1) / r_month)
    months = months_to_target(0.0, 1_000.0, target, r_month)
    cases.append(("定投终值反解", abs(months - 120.0) < 1e-6, f"{months:.6f} 个月"))

    # 3. 解析解与迭代解一致
    principal, monthly_save, goal = 200_000.0, 5_000.0, 1_500_000.0
    r_month = monthly_rate(real_rate(DEFAULT_RETURN, DEFAULT_INFLATION))
    try:
        months = months_to_target(principal, monthly_save, goal, r_month)
        cross_check(principal, monthly_save, goal, r_month, months)
        cases.append(("解析/迭代一致", True, f"{months:.4f} 个月"))
    except AssertionError as exc:
        cases.append(("解析/迭代一致", False, str(exc)))

    # 4. 购买力负增长且储蓄不足时应报错
    try:
        months_to_target(principal, monthly_save, goal, monthly_rate(-0.10))
        cases.append(("负增长不可达", False, "未报错"))
    except TargetUnreachable as exc:
        cases.append(("负增长不可达", True, str(exc)))

    # 5. 年储蓄与储蓄率派生
    rate = saving_rate(150_000.0, 60_000.0)
    ok = rate is not None and abs(rate - 0.6) < 1e-12
    cases.append(("储蓄率派生", ok, f"储蓄率 {rate * 100:.1f}%"))

    # 6. 消耗期耗尽：净值负增长 + 高支取，必然耗尽
    starving = build_context({
        "age": 60, "principal": 13_000_000.0,
        "annual_income": None, "annual_expense": 500_000.0,
        "annual_save": 0.0,
        "nominal": 0.0, "inflation": DEFAULT_INFLATION, "withdrawal": DEFAULT_WITHDRAWAL,
        "income_growth": DEFAULT_INFLATION,
        "life_expectancy": 95, "goal_years": None,
    })
    depleted = depletion_age(build_lifeline(starving))
    cases.append(("消耗期耗尽", depleted is not None, f"耗尽年龄 {depleted}"))

    # 7. 消耗期未耗尽：设计文档样例输入
    healthy = build_context(sample_params())
    rows = build_lifeline(healthy)
    cases.append(
        ("消耗期未耗尽", depletion_age(rows) is None,
         f"{rows[-1][0]} 岁仍有 {format_wan(rows[-1][1])}")
    )

    # 8. 敏感度单调性：收益率上升时达成年龄不得变大
    base_age = reach_age(healthy)
    better_age = reach_age(healthy, nominal_pct=healthy.nominal + 1.0)
    ok = (
        base_age is not None and better_age is not None and better_age <= base_age
    )
    cases.append(("敏感度单调性", ok, f"{base_age:.2f} → {better_age:.2f}"))

    # 9. 分项年化与合计：3,000 元/月 + 20,000 元/年 = 56,000 元
    rent = ExpenseItem("住房", "房租", 3000.0, "月")
    trip = ExpenseItem("旅行", "旅行", 20000.0, "年")
    total = expense_items_total([rent, trip])
    cases.append(("分项年化与合计", abs(total - 56_000.0) < 1e-9, f"合计 {total:,.0f} 元"))

    # 10. 分项占比
    parts = expense_breakdown([rent, trip])
    ok = (
        abs(parts[0][3] - 36_000.0 / 56_000.0) < 1e-9
        and abs(parts[1][3] - 20_000.0 / 56_000.0) < 1e-9
    )
    cases.append(("分项占比", ok, f"{parts[0][3] * 100:.1f}% / {parts[1][3] * 100:.1f}%"))

    # 11. 城市乘数参考值：全国基准 2,000 × 1.3 = 2,600
    reference = baseline_expense(EXPENSE_CATALOG[0], 1.3)
    cases.append(("城市乘数参考值", abs(reference - 2600.0) < 1e-9, f"2,000 × 1.3 = {reference:,.0f}"))

    # 12. 收入增长加速积累：名义增长率 +5% 应早于名义增长率 0
    earlier = reach_age(healthy, income_growth_pct=healthy.inflation + 5.0)
    flat = reach_age(healthy, income_growth_pct=0.0)
    ok = earlier is not None and flat is not None and earlier < flat
    cases.append(("收入增长加速", ok, f"{earlier:.2f} < {flat:.2f} 岁"))

    # 13. 收入负增长减速：名义增长率 -5% 应晚于名义增长率 0
    later = reach_age(healthy, income_growth_pct=healthy.inflation - 5.0)
    ok = flat is not None and later is not None and later > flat
    cases.append(("收入负增长减速", ok, f"{later:.2f} > {flat:.2f} 岁"))

    # 14. 增长口径一致性：达成年龄必须落在逐年表的「达成」年内
    consistent = []
    for growth in (DEFAULT_INFLATION, DEFAULT_INFLATION + 3.0):
        params = sample_params()
        params["income_growth"] = growth
        ctx = build_context(params)
        try:
            cross_check_lifeline(ctx, build_lifeline(ctx), reach_age(ctx))
            consistent.append(True)
        except AssertionError:
            consistent.append(False)
    cases.append((
        "增长口径一致性", all(consistent),
        f"名义增长率 {DEFAULT_INFLATION}% / {DEFAULT_INFLATION + 3.0}% 均与逐年表一致",
    ))

    # 15. 默认跟随通胀率：实际收入增长率为 0，结果与第 2 版基准一致
    zero = abs(real_income_growth(DEFAULT_INFLATION, DEFAULT_INFLATION)) < 1e-15
    age = reach_age(healthy)
    ok = zero and age is not None and 40.1 < age < 40.3
    cases.append(("默认跟随通胀率", ok, f"实际收入增长率 0，达成 {age:.2f} 岁"))

    failed = sum(1 for _, ok, _ in cases if not ok)
    for name, ok, detail in cases:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print(f"{len(cases) - failed}/{len(cases)} 通过")
    return 1 if failed else 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def launched_by_double_click():
    """判断是否由资源管理器双击启动。

    双击 .py 时 Windows 会为 python.exe 单独新建控制台，控制台上只挂着
    自身一个进程；从 cmd / PowerShell 启动时，宿主进程也在同一个控制台上。
    """
    if os.name != "nt":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        buffer = (ctypes.c_uint32 * 8)()
        return kernel32.GetConsoleProcessList(buffer, 8) <= 1
    except Exception:
        return False


def pause_before_exit(disabled=False):
    """双击启动时等待回车，让报错信息留在屏幕上。"""
    if disabled or not launched_by_double_click():
        return
    try:
        input("按回车键关闭窗口...")
    except (EOFError, KeyboardInterrupt):
        pass


def run_cli(argv=None):
    """入口包装：无论成功还是失败，双击启动的窗口都会停留。"""
    no_pause = "--no-pause" in (sys.argv[1:] if argv is None else argv)
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception:
        print("\n发生未预期的错误：", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        pause_before_exit(disabled=no_pause)


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)

    scenario_rates = None
    show_table = True
    show_chart = True
    if raw:
        args = parse_args(raw)
        if args.selftest:
            return run_selftest()
        scenario_rates = args.scenario
        show_table = not args.no_table
        show_chart = not args.no_chart

    try:
        params = params_from_args(args) if raw else ask_params()
        ctx = build_context(params)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    # 储蓄恒定时闭式解必须与逐月迭代吻合，否则说明公式实现有误
    if abs(ctx.real_income_growth) < ASSERT_TOLERANCE:
        try:
            months = months_to_target(
                ctx.principal, ctx.monthly_save, ctx.target, ctx.r_month
            )
        except TargetUnreachable:
            pass
        else:
            cross_check(ctx.principal, ctx.monthly_save, ctx.target, ctx.r_month, months)

    rows = build_lifeline(ctx)
    cross_check_lifeline(ctx, rows, reach_age(ctx))
    sens = sensitivity(ctx)

    print(format_summary(ctx))
    if ctx.expense_items is not None:
        print()
        print(format_expense_breakdown(expense_breakdown(ctx.expense_items)))
    print()
    print(format_conclusion(ctx, rows, sens))
    print()
    print(format_sensitivity(ctx, sens))
    if scenario_rates:
        print()
        print(format_scenarios(rate_scenarios(ctx, scenario_rates)))
    if show_table:
        print()
        print(format_lifeline(rows))
    if show_chart:
        draw_chart(ctx, rows, scenario_rates or [], chart_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())