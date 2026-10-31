"""
run_intraday.py - starts the intraday scanner AND logs every entry alert to the sheet.
intraday.py itself is not changed.
"""
import intraday
import tradelog

_original = intraday.build_alert


def logged_build_alert(ticker, name, side, mode, entry, sl, label, ts, extra="", sec_line=""):
    text = _original(ticker, name, side, mode, entry, sl, label, ts, extra, sec_line)
    if text and not intraday.TEST_ONCE:           # test runs are never logged
        try:
            symbol, _, rest = name.partition(" (")
            sector = rest[:-1] if rest.endswith(")") else rest
            risk = abs(entry - sl)
            target = entry + intraday.RR * risk if side == "BUY" else entry - intraday.RR * risk
            day = ts.strftime("%Y-%m-%d")
            tradelog.add_rows("Intraday", [{
                "ID": f"{day}|{symbol}|INTRADAY",
                "Date": day,
                "Time": ts.strftime("%H:%M"),
                "Strategy": "Intraday 100/200 EMA",
                "Stock": symbol,
                "Sector": sector,
                "Side": side,
                "Entry Type": mode,
                "Entry": round(entry, 2),
                "SL": round(sl, 2),
                "Target": round(target, 2),
                "Status": "OPEN",
                "Exit Price": "",
                "Exit Time": "",
                "R": "",
                "Notes": "",
            }])
        except Exception as e:
            print("Could not log trade:", e)
    return text


intraday.build_alert = logged_build_alert

if __name__ == "__main__":
    intraday.run()
