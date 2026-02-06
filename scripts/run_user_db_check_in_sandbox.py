from __future__ import annotations

from agent.sandbox import execute_python

CODE = r'''# 读取贵州茅台股价数据，验证数据库连接
import pandas as pd

# 贵州茅台代码
ts_code = "600519.SH"

print("=" * 60)
print("贵州茅台股价数据验证")
print("=" * 60)

# 方法1：使用store.daily读取最近数据
print("\n1. 使用store.daily()读取最近10个交易日数据:")
try:
    df = store.daily(ts_code)
    if df.empty:
        print("   ❌ 未获取到数据")
    else:
        df = df.sort_values("trade_date", ascending=False)
        recent_data = df.head(10)
        print(f"   ✅ 成功获取数据，共{len(df)}条记录")
        print(f"   最新交易日: {recent_data.iloc[0]['trade_date']}")
        print(f"   最新收盘价: {recent_data.iloc[0]['close']:.2f}元")

        # 显示最近5个交易日
        print("\n   最近5个交易日数据:")
        display_cols = ["trade_date", "open", "high", "low", "close", "pct_chg", "vol"]
        display_df = recent_data.head(5)[display_cols].copy()
        display_df["trade_date"] = pd.to_datetime(display_df["trade_date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
        display_df["vol(万手)"] = (display_df["vol"] / 100).round(2)
        display_df = display_df[["trade_date", "open", "high", "low", "close", "pct_chg", "vol(万手)"]]
        print(display_df.to_string(index=False))

except Exception as e:
    print(f"   ❌ store.daily()出错: {str(e)}")

# 方法2：使用store.daily_basic读取估值数据
print("\n2. 使用store.daily_basic()读取估值数据:")
try:
    basic_df = store.daily_basic(ts_code)
    if basic_df.empty:
        print("   ❌ 未获取到估值数据")
    else:
        basic_df = basic_df.sort_values("trade_date", ascending=False)
        recent_basic = basic_df.head(1)
        print(f"   ✅ 成功获取估值数据，共{len(basic_df)}条记录")
        print(f"   最新PE(TTM): {recent_basic.iloc[0]['pe_ttm']:.2f}")
        print(f"   最新PB: {recent_basic.iloc[0]['pb']:.2f}")
        print(f"   总市值: {recent_basic.iloc[0]['total_mv']/10000:.2f}亿元")
        print(f"   流通市值: {recent_basic.iloc[0]['circ_mv']/10000:.2f}亿元")

except Exception as e:
    print(f"   ❌ store.daily_basic()出错: {str(e)}")

# 方法3：读取股票基本信息
print("\n3. 读取股票基本信息:")
try:
    # 尝试读取股票基本信息
    stock_info = store.read("stock_basic", where={"ts_code": ts_code})
    if stock_info.empty:
        print("   ❌ 未获取到股票基本信息")
    else:
        print(f"   ✅ 成功获取股票基本信息")
        info = stock_info.iloc[0]
        print(f"   股票名称: {info['name']}")
        print(f"   上市日期: {info['list_date']}")
        print(f"   行业: {info['industry']}")
        print(f"   市场: {info['market']}")

except Exception as e:
    print(f"   ❌ 读取股票基本信息出错: {str(e)}")

# 方法4：测试其他数据表
print("\n4. 测试其他数据表访问:")
try:
    # 测试指数数据
    index_data = store.read("index_basic", limit=3)
    print(f"   ✅ 指数数据表可访问，样本数: {len(index_data)}")

    # 测试ETF数据
    etf_data = store.read("fund_basic", limit=3)
    print(f"   ✅ ETF数据表可访问，样本数: {len(etf_data)}")

    # 测试交易日数据
    trading_days = store.trading_days("20240101", "20240110")
    print(f"   ✅ 交易日数据可访问，2024年1月1-10日交易日数: {len(trading_days)}")

except Exception as e:
    print(f"   ❌ 其他数据表访问出错: {str(e)}")

print("\n" + "=" * 60)
print("数据库连接测试总结")
print("=" * 60)

# 总结
if 'df' in locals() and not df.empty:
    print("✅ 数据库连接正常，数据可正常读取")
    print(f"   贵州茅台数据范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
    print(f"   总数据条数: {len(df)}")

    # 数据新鲜度检查
    latest_date = df['trade_date'].max()
    print(f"   最新数据日期: {latest_date}")

    # 检查数据是否完整
    date_range = pd.date_range(start="2020-01-01", end="2026-02-04", freq='D')
    trading_days_count = len([d for d in date_range if d.weekday() < 5])  # 粗略估计
    data_coverage = len(df) / trading_days_count * 100 if trading_days_count > 0 else 0
    print(f"   数据覆盖率(估算): {data_coverage:.1f}%")

else:
    print("❌ 数据库连接或数据读取存在问题")

print("\n建议:")
print("1. 如果所有测试都通过 ✅，说明数据库连接正常")
print("2. 如果部分测试失败 ❌，可能是特定数据表的问题")
print("3. 贵州茅台作为A股标杆，其数据通常最完整")
'''


def main() -> None:
    out = execute_python(CODE, session_id="db-check", timeout_seconds=180)
    print("success:", out.get("success"))
    if out.get("output"):
        print("output:\n", out.get("output"))
    if out.get("error"):
        print("error:\n", out.get("error"))
    if out.get("result"):
        print("result:\n", out.get("result"))


if __name__ == "__main__":
    main()
