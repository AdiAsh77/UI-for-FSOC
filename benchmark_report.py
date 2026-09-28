"""
benchmark_report.py

Turns one completed Benchmark Performance-1 run (see Main.run_bench() /
_bench_finish_all() in dashboard.py) into the two mandatory-style deliverables
called out in the SIH problem statement (PS #26169): a Performance Log
(spreadsheet) and a short written report. Both are built entirely from the
real per-frame stats backend.TrackingThread.stats_ready emitted while each
scenario ran — nothing here is fabricated or resampled.

Kept as its own module (rather than folded into dashboard.py) so the PyQt
file stays focused on the UI, and so these can be re-run or unit-tested
without importing PyQt/backend at all.

Requires: openpyxl, python-docx  (pip install openpyxl python-docx)
"""

import os
import statistics
from datetime import datetime


# ================================================================
# Column layout shared with dashboard.py's Data-Check tab
# (CR / FM / SPEC_THRESH in dashboard.py) — duplicated here as plain
# data so this module has no PyQt/dashboard import dependency.
# ================================================================

CRITERIA = ['Acquisition (s)', 'Tracking error (px)', 'Target loss', 'Speed (FPS)', 'Update (ms)', 'Lock retention (%)', 'Tracking RMSE (px)']
DECIMALS = [2, 2, 0, 0, 1, 1, 2]
# (limit, lower_is_better) — same numbers as SPEC_THRESH in dashboard.py /
# the problem statement's Performance Specifications table. None = no pass/fail line.
SPEC_THRESH = [(2.0, True), (10.0, True), (1.0, True), (20.0, False), None, None, (10.0, True)]
N_CRIT = len(CRITERIA)


def _fmt(value, decimals):
    return f"{value:.{decimals}f}" if decimals else f"{int(round(value))}"


def _passfail(col_index, value):
    th = SPEC_THRESH[col_index]
    if th is None:
        return None
    limit, lower_is_better = th
    ok = (value <= limit) if lower_is_better else (value >= limit)
    return ok


def _safe_filename(name):
    name = (name or 'Benchmark').strip() or 'Benchmark'
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, '_')
    return name


# ================================================================
# EXCEL — Performance Log
# ================================================================

def export_excel(entry, frames_by_scene, out_dir):
    """entry: dict(name, type, date, cols, rows) — rows are the same 7-tuples
    dashboard.py's Data-Check tab already uses: (scenario, ap, acq, err, loss, fps, upd).
    frames_by_scene: {scenario_name: [stats_ready dict, ...]} — the raw per-frame
    stats collected for that scenario during the run, in order.
    Returns the full path written."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.chart import BarChart, Reference
    from openpyxl.utils import get_column_letter

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _safe_filename(entry['name']) + '.xlsx')

    wb = Workbook()
    bold = Font(bold=True)
    head_fill = PatternFill('solid', fgColor='DCE6F1')
    ok_fill = PatternFill('solid', fgColor='DCF5E8')
    fail_fill = PatternFill('solid', fgColor='FBDCDE')

    rows = entry['rows']
    n = len(rows)

    # ---- Summary sheet -----------------------------------------------
    ws = wb.active
    ws.title = 'Summary'
    ws.append(['Benchmark name', entry['name']])
    ws.append(['Type', entry['type']])
    ws.append(['Date', entry['date']])
    ws.append(['Scenarios run', n])
    total_duration = sum(len(f) and (f[-1].get('elapsed') or 0.0) for f in frames_by_scene.values())
    ws.append(['Total simulated duration (s)', round(total_duration, 1)])
    ws.append([])
    ws.append(['Average across scenarios'])
    ws['A7'].font = bold
    for j, crit in enumerate(CRITERIA):
        avg = sum(r[j + 1] for r in rows) / n if n else 0.0
        row = [crit, round(avg, DECIMALS[j])]
        th = SPEC_THRESH[j]
        if th is not None:
            ok = _passfail(j, avg)
            row.append('PASS' if ok else 'FAIL')
        ws.append(row)
        if th is not None:
            ws.cell(row=ws.max_row, column=3).fill = ok_fill if ok else fail_fill
    for c, w in zip('AB', (28, 20)):
        ws.column_dimensions[c].width = w

    # ---- Scenario results sheet (this is the data behind the Data-Check
    # "By scenario" / "By run" Compliance charts — same criteria, one row
    # per scenario) --------------------------------------------------
    ws2 = wb.create_sheet('Scenario results')
    # Lock retention (%) is already one of the CRITERIA columns, so it isn't
    # repeated in the trailing extras below.
    header = ['Scenario'] + CRITERIA + ['Sim. duration (s)', 'Accuracy (%)', 'Target loss (%)']
    ws2.append(header)
    for c in range(1, len(header) + 1):
        cell = ws2.cell(row=1, column=c)
        cell.font = bold
        cell.fill = head_fill
    for r in rows:
        name = r[0]
        frames = frames_by_scene.get(name, [])
        last = frames[-1] if frames else {}
        duration = last.get('elapsed', 0.0) or 0.0
        acc = last.get('accuracy', 0.0) or 0.0
        loss_pct = last.get('target_loss_pct', 0.0) or 0.0
        ws2.append([name] + [r[j + 1] for j in range(N_CRIT)] + [round(duration, 1), round(acc, 1), round(loss_pct, 1)])
        for j in range(N_CRIT):
            th = SPEC_THRESH[j]
            if th is not None:
                cell = ws2.cell(row=ws2.max_row, column=j + 2)
                cell.fill = ok_fill if _passfail(j, r[j + 1]) else fail_fill
    for c in range(1, len(header) + 1):
        ws2.column_dimensions[get_column_letter(c)].width = 18
    ws2.column_dimensions['A'].width = 26

    # ---- Native Excel bar charts, one per criterion — the graphs asked
    # for in Data-Check's Compliance charts panel, built straight off the
    # Scenario results sheet above -------------------------------------
    charts_ws = wb.create_sheet('Compliance charts')
    row_cursor = 1
    for j, crit in enumerate(CRITERIA):
        chart = BarChart()
        chart.title = crit
        chart.y_axis.title = crit
        chart.x_axis.title = 'Scenario'
        data = Reference(ws2, min_col=j + 2, max_col=j + 2, min_row=1, max_row=n + 1)
        cats = Reference(ws2, min_col=1, min_row=2, max_row=n + 1)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.height, chart.width = 8, 16
        charts_ws.add_chart(chart, f'A{row_cursor}')
        row_cursor += 17

    # ---- Raw per-frame log — one sheet, all scenarios stacked, exactly
    # the real values backend.TrackingThread reported (no resampling) --
    ws3 = wb.create_sheet('Raw per-frame log')
    ws3.append(['Scenario', 'Elapsed (s)', 'FPS', 'Pan (deg)', 'Tilt (deg)', 'Error (px)', 'RMSE (px)', 'Lock retention (%)', 'Accuracy (%)', 'Target losses', 'Acquisition time (s)'])
    for c in range(1, 12):
        cell = ws3.cell(row=1, column=c); cell.font = bold; cell.fill = head_fill
    for name, frames in frames_by_scene.items():
        for f in frames:
            ws3.append([
                name,
                round(f.get('elapsed', 0.0) or 0.0, 3),
                round(f.get('fps', 0.0) or 0.0, 1),
                round(f.get('pan', 0.0) or 0.0, 2),
                round(f.get('tilt', 0.0) or 0.0, 2),
                round(f['error'], 2) if f.get('error') is not None else '',
                round(f['rmse'], 2) if f.get('rmse') is not None else '',
                round(f.get('lock_retention', 0.0) or 0.0, 1),
                round(f.get('accuracy', 0.0) or 0.0, 1),
                f.get('target_losses', 0) or 0,
                round(f['acquisition_time'], 3) if f.get('acquisition_time') is not None else '',
            ])
    ws3.freeze_panes = 'A2'

    wb.save(path)
    return path


# ================================================================
# WORD — short benchmark report (Performance Log narrative form,
# referencing the problem statement's Benchmark Performance-1 stage)
# ================================================================

def export_word(entry, frames_by_scene, out_dir):
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _safe_filename(entry['name']) + '.docx')

    rows = entry['rows']
    n = len(rows)
    doc = Document()

    title = doc.add_heading(entry['name'], level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(f"{entry['type']}  ·  {entry['date']}")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in sub.runs:
        run.italic = True

    doc.add_heading('Overview', level=1)
    total_duration = sum(len(f) and (f[-1].get('elapsed') or 0.0) for f in frames_by_scene.values())
    doc.add_paragraph(
        f"This report covers Benchmark Performance-1 (in-house scenario suite) of the coarse-alignment "
        f"virtual camera tracking system, as described in SIH problem statement PS #26169. {n} scenario"
        f"{'s' if n != 1 else ''} were run end-to-end through the live Sim View pipeline "
        f"(backend.TrackingThread, driving the same Unity gimbal used interactively), each for the "
        f"simulation duration configured in its own scene file, for a combined {total_duration:.1f} s "
        f"of simulated tracking. Every figure below is taken directly from the per-frame detection, "
        f"Kalman-filter and pan/tilt telemetry produced during those runs — none of it is estimated "
        f"or interpolated afterwards."
    )

    doc.add_heading('Summary (average across scenarios)', level=1)
    tbl = doc.add_table(rows=1, cols=3)
    tbl.style = 'Light Grid Accent 1'
    hdr = tbl.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text = 'Criterion', 'Average', 'vs. spec'
    for j, crit in enumerate(CRITERIA):
        avg = sum(r[j + 1] for r in rows) / n if n else 0.0
        cells = tbl.add_row().cells
        cells[0].text = crit
        cells[1].text = _fmt(avg, DECIMALS[j])
        th = SPEC_THRESH[j]
        if th is None:
            cells[2].text = '—'
        else:
            limit, lower = th
            ok = _passfail(j, avg)
            cells[2].text = f"{'PASS' if ok else 'FAIL'} ({'≤' if lower else '≥'} {limit:g})"

    doc.add_heading('Per-scenario results', level=1)
    doc.add_paragraph(
        "The table below lines every checked scenario up against every criterion at a glance; the "
        "section that follows breaks each scenario out individually with its own pass/fail call "
        "against the PS #26169 performance specification."
    )
    tbl2 = doc.add_table(rows=1, cols=len(CRITERIA) + 1)
    tbl2.style = 'Light Grid Accent 1'
    for c, h in zip(tbl2.rows[0].cells, ['Scenario'] + CRITERIA):
        c.text = h
    for r in rows:
        cells = tbl2.add_row().cells
        cells[0].text = r[0]
        for j in range(N_CRIT):
            cells[j + 1].text = _fmt(r[j + 1], DECIMALS[j])

    # ---- Individual scenario breakdown — each scenario gets its own heading, a
    # short narrative summarising how it went, and a full criterion-by-criterion
    # reading-vs-spec table with an explicit PASS/FAIL call on every row that has
    # a hard spec threshold. Requested in addition to (not instead of) the
    # combined table above.
    doc.add_heading('Individual scenario results', level=1)
    doc.add_paragraph(
        "Each of the following breaks a single scenario's run out on its own, so its behaviour can be "
        "read without cross-referencing the summary tables above."
    )
    for r in rows:
        name = r[0]
        frames = frames_by_scene.get(name, [])
        duration = (frames[-1].get('elapsed') or 0.0) if frames else 0.0
        n_frames = len(frames)
        checked = [(j, crit) for j, crit in enumerate(CRITERIA) if SPEC_THRESH[j] is not None]
        passed = [crit for j, crit in checked if _passfail(j, r[j + 1])]
        failed = [crit for j, crit in checked if not _passfail(j, r[j + 1])]

        doc.add_heading(name, level=2)
        lead = f"Ran for {duration:.1f} s ({n_frames} frames processed). " if n_frames else ""
        if not checked:
            verdict = "No criteria with a hard spec threshold were selected for this run."
        elif not failed:
            verdict = f"This scenario met all {len(checked)} of its checked performance targets."
        elif not passed:
            verdict = f"This scenario missed every checked performance target: {', '.join(failed)}."
        else:
            verdict = (f"This scenario met {len(passed)} of {len(checked)} checked targets, "
                       f"falling short on {', '.join(failed)}.")
        doc.add_paragraph(lead + verdict)

        t = doc.add_table(rows=1, cols=3)
        t.style = 'Light Grid Accent 1'
        th_cells = t.rows[0].cells
        th_cells[0].text, th_cells[1].text, th_cells[2].text = 'Criterion', 'Reading', 'vs. spec'
        for j, crit in enumerate(CRITERIA):
            cells = t.add_row().cells
            cells[0].text = crit
            cells[1].text = _fmt(r[j + 1], DECIMALS[j])
            th = SPEC_THRESH[j]
            if th is None:
                cells[2].text = '—'
            else:
                limit, lower = th
                ok = _passfail(j, r[j + 1])
                cells[2].text = f"{'PASS' if ok else 'FAIL'} ({'≤' if lower else '≥'} {limit:g})"

    doc.add_heading('Test methodology', level=1)
    doc.add_paragraph(
        "Each checked scenario in the Benchmark tab's Scenarios list was loaded through "
        "backend.TrackingThread.parsed() — the same call Sim View's own \"Load scene\" button makes — "
        "which resets the Kalman filter, acquisition state and all running metrics, sends the scene "
        "to Unity, and resumes tracking. Acquisition time, tracking-error RMSE, lock retention, target "
        "loss count and processing FPS were then read from backend.py's own stats_ready stream (the "
        "same values shown live on the Sim View tab) once each scenario's configured simulation time "
        "had elapsed. Lock retention (%) and Tracking RMSE (px) are reported as their own criteria — "
        "rather than folded into a single proxy score — because PS #26169's Benchmark Performance-2 "
        "evaluation table names \"RMSE\" and \"Lock retention rate\" as separate metrics alongside "
        "acquisition/re-acquisition time and FPS."
    )

    doc.add_heading('Performance targets (from PS #26169)', level=1)
    doc.add_paragraph(
        "Acquisition time ≤ 2 s  ·  Tracking error ≤ 10 px  ·  Target loss < 5 %  ·  "
        "Re-acquisition ≤ 1 s  ·  Processing speed ≥ 20 FPS."
    )

    doc.save(path)
    return path


def export_reports(entry, frames_by_scene, out_dir):
    """Writes both deliverables and returns (excel_path, word_path).
    Raises whatever openpyxl/python-docx raise if either package is missing
    or the write fails — callers should catch and report that to the user."""
    xlsx_path = export_excel(entry, frames_by_scene, out_dir)
    docx_path = export_word(entry, frames_by_scene, out_dir)
    return xlsx_path, docx_path