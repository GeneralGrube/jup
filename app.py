from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

PLANNING_YEAR = 2027
BASE_DIR = Path(__file__).resolve().parent
MONTHS = [
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]
PRIORITY_LABELS = {
    "JOKER": "Joker",
    "HIGH": "Hoch",
    "LOW": "Niedrig",
    "PLACEHOLDER": "Platzhalter",
}
PRIORITY_FROM_LABEL = {label: code for code, label in PRIORITY_LABELS.items()}
SHEET_HEADERS = [
    "entry_id",
    "employee_id",
    "start_date",
    "end_date",
    "absence_code",
    "priority",
    "workdays_count",
    "is_submitted",
    "last_modified_utc",
    "is_deleted",
]


def load_master_data() -> dict[str, Any]:
    employees = json.loads((BASE_DIR / "employees.json").read_text(encoding="utf-8"))
    teams = json.loads((BASE_DIR / "teams_and_skills.json").read_text(encoding="utf-8"))
    absence_types = pd.read_csv(BASE_DIR / "absence_types.csv")
    holidays = pd.read_csv(BASE_DIR / "holidays.csv")
    school_holidays = pd.read_csv(BASE_DIR / "school_holidays.csv")
    blackout = pd.read_csv(BASE_DIR / "blackout_periods.csv")

    return {
        "employees": employees,
        "teams": teams,
        "absence_types": absence_types,
        "holidays": holidays,
        "school_holidays": school_holidays,
        "blackout": blackout,
    }


def get_employee_map() -> dict[str, dict[str, Any]]:
    return {str(item["employee_id"]): item for item in load_master_data()["employees"]}


WEEKDAY_ALIASES = {
    "montag": 0,
    "mo": 0,
    "monday": 0,
    "dienstag": 1,
    "di": 1,
    "tuesday": 1,
    "mittwoch": 2,
    "mi": 2,
    "wednesday": 2,
    "donnerstag": 3,
    "do": 3,
    "thursday": 3,
    "freitag": 4,
    "fr": 4,
    "friday": 4,
    "samstag": 5,
    "sa": 5,
    "saturday": 5,
    "sonntag": 6,
    "so": 6,
    "sunday": 6,
}


def get_standard_absence_weekdays(employee: dict[str, Any]) -> set[int]:
    """Return configured standard-absence weekdays as Monday=0 through Sunday=6."""
    weekdays: set[int] = set()
    for value in employee.get("std_absences", []):
        if isinstance(value, int) and 0 <= value <= 6:
            weekdays.add(value)
            continue
        normalized = str(value).strip().lower()
        if normalized.isdigit() and 0 <= int(normalized) <= 6:
            weekdays.add(int(normalized))
            continue
        if normalized not in WEEKDAY_ALIASES:
            raise ValueError(
                f"Ungültiger Wochentag in std_absences: {value!r}"
            )
        weekdays.add(WEEKDAY_ALIASES[normalized])
    return weekdays


def get_absence_meta() -> dict[str, dict[str, Any]]:
    rows = load_master_data()["absence_types"].to_dict(orient="records")
    return {row["code"]: row for row in rows}


def get_absence_label(code: str) -> str:
    return str(get_absence_meta().get(code, {}).get("label", code))


def get_absence_options() -> list[tuple[str, str]]:
    return [
        (str(row["code"]), str(row["label"]))
        for row in load_master_data()["absence_types"].to_dict(orient="records")
    ]


def get_team_label(team_id: str) -> str:
    return next(
        (
            str(team["label"])
            for team in load_master_data()["teams"]["teams"]
            if str(team["team_id"]) == str(team_id)
        ),
        team_id,
    )


def get_skill_label(skill_id: str) -> str:
    return next(
        (
            str(skill["label"])
            for skill in load_master_data()["teams"]["sub_skills"]
            if str(skill["skill_id"]) == str(skill_id)
        ),
        skill_id,
    )


def get_min_employees(scope_type: str, scope_id: str) -> list[int]:
    collection = (
        load_master_data()["teams"]["teams"]
        if scope_type == "team"
        else load_master_data()["teams"]["sub_skills"]
    )
    configured = next(
        (
            item.get("min_employees")
            for item in collection
            if str(item.get("team_id", item.get("skill_id"))) == str(scope_id)
        ),
        None,
    )
    if not isinstance(configured, list) or len(configured) != 5:
        raise ValueError(
            f"Für {scope_type} {scope_id} wird min_employees mit fünf Werten benötigt."
        )
    return [int(value) for value in configured]


def get_holiday_adjust(scope_type: str, scope_id: str) -> list[int]:
    collection = (
        load_master_data()["teams"]["teams"]
        if scope_type == "team"
        else load_master_data()["teams"]["sub_skills"]
    )
    configured = next(
        (
            item.get("holiday_adjust", [0, 0, 0, 0, 0])
            for item in collection
            if str(item.get("team_id", item.get("skill_id"))) == str(scope_id)
        ),
        [0, 0, 0, 0, 0],
    )
    if not isinstance(configured, list) or len(configured) != 5:
        raise ValueError(
            f"Für {scope_type} {scope_id} wird holiday_adjust mit fünf Werten benötigt."
        )
    return [int(value) for value in configured]


def is_school_holiday(target: date) -> bool:
    return any(
        _date_in_range(
            target,
            normalize_date(row["start_date"]),
            normalize_date(row["end_date"]),
        )
        for row in load_master_data()["school_holidays"].to_dict(orient="records")
    )


def get_holiday_dates() -> set[date]:
    df = load_master_data()["holidays"]
    return {pd.to_datetime(row["date"]).date() for _, row in df.iterrows()}


def get_blackout_ranges() -> list[tuple[date, date, str]]:
    df = load_master_data()["blackout"]
    ranges = []
    for _, row in df.iterrows():
        start = pd.to_datetime(row["start_date"]).date()
        end = pd.to_datetime(row["end_date"]).date()
        ranges.append((start, end, str(row["reason"])))
    return ranges


def workdays_in_range(start: date, end: date, holiday_dates: set[date] | None = None) -> int:
    if start > end:
        return 0
    if holiday_dates is None:
        holiday_dates = get_holiday_dates()
    total = 0
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in holiday_dates:
            total += 1
        current += timedelta(days=1)
    return total


def date_range(start: date, end: date) -> list[date]:
    current = start
    out: list[date] = []
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


@st.cache_resource
def get_vacation_worksheet() -> gspread.Worksheet:
    config = dict(st.secrets["connections"]["gsheets"])
    service_account_info = {
        key: config[key]
        for key in (
            "type",
            "project_id",
            "private_key_id",
            "private_key",
            "client_email",
            "client_id",
            "auth_uri",
            "token_uri",
            "auth_provider_x509_cert_url",
            "client_x509_cert_url",
            "universe_domain",
        )
        if key in config
    }
    credentials = Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    client = gspread.authorize(credentials)
    worksheet = client.open_by_url(config["spreadsheet"]).worksheet(config["worksheet"])
    current_headers = worksheet.row_values(1)
    if current_headers != SHEET_HEADERS:
        if current_headers and current_headers != SHEET_HEADERS:
            raise ValueError(
                "The vacation_requests worksheet has unexpected headers. "
                f"Expected {SHEET_HEADERS}, found {current_headers}."
            )
        worksheet.update("A1:J1", [SHEET_HEADERS], value_input_option="RAW")
    return worksheet


def entry_to_row(entry: dict[str, Any]) -> list[Any]:
    return [
        entry["entry_id"],
        str(entry["employee_id"]),
        entry["start_date"],
        entry["end_date"],
        entry["absence_code"],
        entry["priority"],
        int(entry["workdays_count"]),
        bool(entry.get("is_submitted", False)),
        entry["last_modified_utc"],
        bool(entry.get("is_deleted", False)),
    ]


def row_to_entry(row: dict[str, Any]) -> dict[str, Any]:
    def as_bool(value: Any) -> bool:
        return str(value).strip().lower() in {"true", "1", "yes"}

    return {
        "entry_id": str(row.get("entry_id", "")),
        "employee_id": str(row.get("employee_id", "")),
        "start_date": str(row.get("start_date", "")),
        "end_date": str(row.get("end_date", "")),
        "absence_code": str(row.get("absence_code", "")),
        "priority": str(row.get("priority", "")),
        "workdays_count": int(row.get("workdays_count") or 0),
        "is_submitted": as_bool(row.get("is_submitted", False)),
        "last_modified_utc": str(row.get("last_modified_utc", "")),
        "is_deleted": as_bool(row.get("is_deleted", False)),
    }


def fetch_all_entries() -> list[dict[str, Any]]:
    entries = [
        row_to_entry(row)
        for row in get_vacation_worksheet().get_all_records(
            expected_headers=SHEET_HEADERS
        )
    ]
    return [entry for entry in entries if not entry["is_deleted"]]


def get_pending_entries() -> list[dict[str, Any]]:
    return st.session_state.setdefault("pending_entries", [])


def entries_with_pending_persistence(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending = get_pending_entries()
    pending_ids = {entry["entry_id"] for entry in pending}
    return [entry for entry in entries if entry["entry_id"] not in pending_ids] + pending


def persist_pending_entries() -> None:
    pending = list(get_pending_entries())
    for entry in pending:
        save_entry(entry)
    st.session_state["pending_entries"] = []


def save_entry(entry: dict[str, Any]) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    if not entry.get("entry_id"):
        entry["entry_id"] = str(uuid.uuid4())
    entry["last_modified_utc"] = now
    worksheet = get_vacation_worksheet()
    values = worksheet.get_all_values()
    entry_ids = [row[0] for row in values[1:] if row]
    row_index = entry_ids.index(entry["entry_id"]) + 2 if entry["entry_id"] in entry_ids else None
    if row_index is None:
        worksheet.append_rows([entry_to_row(entry)], value_input_option="RAW")
    else:
        worksheet.update(
            f"A{row_index}:J{row_index}",
            [entry_to_row(entry)],
            value_input_option="RAW",
        )


def delete_entry(entry_id: str) -> None:
    worksheet = get_vacation_worksheet()
    values = worksheet.get_all_values()
    entry_ids = [row[0] for row in values[1:] if row]
    if entry_id not in entry_ids:
        raise ValueError(f"Cannot delete unknown vacation entry: {entry_id}")
    row_index = entry_ids.index(entry_id) + 2
    row = values[row_index - 1]
    row[8] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row[9] = True
    worksheet.update(f"A{row_index}:J{row_index}", [row[:10]], value_input_option="RAW")


def delete_employee_entries(employee_id: str, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        if str(entry["employee_id"]) == str(employee_id):
            delete_entry(entry["entry_id"])


@st.dialog("Abwesenheit löschen")
def delete_single_entry_dialog(entry_id: str) -> None:
    st.write(f"Möchten Sie die Abwesenheit **{entry_id}** wirklich löschen?")
    if st.button("Löschen", type="primary"):
        delete_entry(entry_id)
        st.success("Abwesenheit gelöscht.")
        st.rerun()


@st.dialog("Alle Abwesenheiten löschen")
def delete_all_entries_dialog(employee_id: str, entries: list[dict[str, Any]]) -> None:
    st.write("Möchten Sie wirklich alle Ihre Abwesenheiten löschen?")
    st.warning("Diese Aktion kann nicht rückgängig gemacht werden.")
    if st.button("Alle löschen", type="primary"):
        delete_employee_entries(employee_id, entries)
        st.success("Alle Abwesenheiten wurden gelöscht.")
        st.rerun()


def get_employee_entries(employee_id: str, entries: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    entries = entries or fetch_all_entries()
    return [e for e in entries if str(e["employee_id"]) == str(employee_id)]


def format_employee_label(employee_id: str) -> str:
    employee = get_employee_map().get(str(employee_id), {})
    return f"{employee.get('employee_id', employee_id)}"


@st.cache_data
def get_year_calendar() -> pd.DataFrame:
    start = date(PLANNING_YEAR, 1, 1)
    end = date(PLANNING_YEAR, 12, 31)
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur)
        cur += timedelta(days=1)
    df = pd.DataFrame({"date": dates})
    df["weekday"] = df["date"].map(lambda d: d.weekday())
    df["is_weekday"] = df["weekday"].map(lambda x: x < 5)
    return df


def compute_employee_statistics(employee_id: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    employee = get_employee_map().get(str(employee_id), {})
    absence_meta = get_absence_meta()
    holiday_dates = get_holiday_dates()
    all_recreational = []
    high_priority_days = 0
    joker_entry: dict[str, Any] | None = None
    for entry in entries:
        if entry["is_deleted"] or str(entry["employee_id"]) != str(employee_id):
            continue
        if entry["absence_code"] in {"VAC_REC", "COMP_TIME"}:
            all_recreational.append(entry)
        try:
            workdays = int(entry["workdays_count"])
        except (TypeError, ValueError):
            workdays = workdays_in_range(
                datetime.strptime(entry["start_date"], "%Y-%m-%d").date(),
                datetime.strptime(entry["end_date"], "%Y-%m-%d").date(),
                holiday_dates,
            )
        if entry["priority"] in {"JOKER", "HIGH"}:
            high_priority_days += workdays
        if entry["priority"] == "JOKER":
            joker_entry = entry

    recreation_days = sum(
        int(entry["workdays_count"])
        for entry in entries
        if not entry["is_deleted"]
        and str(entry["employee_id"]) == str(employee_id)
        and absence_meta.get(entry["absence_code"], {}).get("is_recreational_vacation") in {True, "TRUE", "true"}
    )
    par_leave_seen = any(
        str(entry["absence_code"]).upper() == "PAR_LEAVE"
        for entry in entries
        if not entry["is_deleted"] and str(entry["employee_id"]) == str(employee_id)
    )
    return {
        "employee": employee,
        "recreation_days": recreation_days,
        "high_priority_days": high_priority_days,
        "joker_entry": joker_entry,
        "has_parental_leave": par_leave_seen,
    }


def normalize_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        if "T" in value:
            value = value.split("T")[0]
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date value: {value!r}")


def overlap_exists(entry_a: dict[str, Any], entry_b: dict[str, Any]) -> bool:
    a_start = normalize_date(entry_a["start_date"])
    a_end = normalize_date(entry_a["end_date"])
    b_start = normalize_date(entry_b["start_date"])
    b_end = normalize_date(entry_b["end_date"])
    return not (a_end < b_start or b_end < a_start)


def validate_entry(employee_id: str, form: dict[str, Any], existing_entries: list[dict[str, Any]], current_entry_id: str | None = None) -> list[str]:
    errors: list[str] = []
    start = normalize_date(form["start_date"])
    end = normalize_date(form["end_date"])
    if start > end:
        errors.append("Fehler: Das Startdatum darf nicht nach dem Enddatum liegen.")
        return errors

    if start.year != PLANNING_YEAR or end.year != PLANNING_YEAR:
        errors.append("Die Abwesenheit muss innerhalb des Planungsjahres 2027 liegen.")

    holiday_dates = get_holiday_dates()
    blackout_ranges = get_blackout_ranges()
    overlap = any(
        str(e["employee_id"]) == str(employee_id)
        and e["entry_id"] != current_entry_id
        and overlap_exists(form, e)
        for e in existing_entries
    )
    if overlap:
        errors.append("Fehler: Der gewählte Zeitraum überschneidet sich mit einer bereits erfassten Abwesenheit. Bitte passen Sie die Daten an.")

    if form["priority"] in {"LOW", "PLACEHOLDER"}:
        for blackout_start, blackout_end, reason in blackout_ranges:
            if not (end < blackout_start or start > blackout_end):
                errors.append(
                    f"Achtung: Der gewählte Zeitraum überschneidet sich mit einer Urlaubssperre ({reason}). Tage innerhalb dieser Sperre erfordern zwingend die Priorität 'Hoch'. Um nur die betroffenen Tage als 'Hoch' zu belasten, teilen Sie Ihren Urlaub bitte manuell in separate Einträge vor, während und nach der Sperre auf."
                )
                break

    if form["priority"] == "JOKER" and form["absence_code"] != "VAC_REC":
        errors.append("Ein Joker kann nur für Erholungsurlaub verwendet werden.")

    if form["priority"] == "JOKER":
        if (end - start).days + 1 > 14:
            errors.append(
                "Ein Joker darf maximal 14 aufeinanderfolgende Kalendertage umfassen. Wenn Sie einen längeren zusammenhängenden Urlaub planen, legen Sie die maximal 14 Tage als Joker-Eintrag an und erfassen Sie die verbleibenden Tage bitte als separaten Eintrag (z. B. mit Priorität 'Hoch' oder 'Niedrig')."
            )
        if any(
            str(e["employee_id"]) == str(employee_id)
            and e["entry_id"] != current_entry_id
            and e["priority"] == "JOKER"
            for e in existing_entries
        ):
            errors.append("Es darf pro Mitarbeiter nur ein Joker-Eintrag pro Jahr geben.")

    high_total = sum(
        int(e["workdays_count"]) if e["priority"] in {"JOKER", "HIGH"} else 0
        for e in existing_entries
        if str(e["employee_id"]) == str(employee_id) and e["entry_id"] != current_entry_id
    )
    if form["priority"] in {"JOKER", "HIGH"}:
        high_total += workdays_in_range(start, end, holiday_dates)
    if high_total > 20:
        errors.append("Die Gesamtzahl der Arbeitstage für Joker und Hoch überschreitet das Limit von 20 Tagen.")

    return errors


def build_recreational_summary(employee_id: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    workdays = 0
    high_days = 0
    joker = None
    par_leave = False
    for entry in entries:
        if str(entry["employee_id"]) != str(employee_id) or entry["is_deleted"]:
            continue
        if entry["absence_code"] == "PAR_LEAVE":
            par_leave = True
        if entry["absence_code"] == "VAC_REC":
            workdays += int(entry["workdays_count"])
        if entry["priority"] in {"JOKER", "HIGH"}:
            high_days += int(entry["workdays_count"])
        if entry["priority"] == "JOKER":
            joker = entry
    return {
        "planned_recreation_days": workdays,
        "high_priority_days": high_days,
        "joker": joker,
        "has_parental_leave": par_leave,
    }


def validate_submission(employee_id: str, entries: list[dict[str, Any]]) -> tuple[bool, str | None, str | None]:
    summary = build_recreational_summary(employee_id, entries)
    if summary["planned_recreation_days"] > 30:
        return True, None, "Sie haben {0} Tage Erholungsurlaub geplant und überschreiten damit den üblichen Anspruch von 30 Tagen.".format(summary["planned_recreation_days"])
    if summary["planned_recreation_days"] < 30 and not summary["has_parental_leave"]:
        return False, "Der Erholungsurlaub wurde nicht vollständig verplant, geben Sie ggf. Tage gegen Jahresende mit der Priorität 'Platzhalter' an.", None
    if summary["planned_recreation_days"] < 30 and summary["has_parental_leave"]:
        return True, None, "Es wurden nur {0} Tage statt des üblichen Anspruchs von 30 Urlaubstagen geplant. Bei Angabe von Elternzeiten kann der Urlaubsanspruch reduziert sein. Bitte prüfen Sie, ob der Jahresurlaub vollständig verplant wurde.".format(summary["planned_recreation_days"])
    if summary["high_priority_days"] > 20:
        return False, "Die Gesamtzahl der Arbeitstage für Joker und Hoch überschreitet das Limit von 20 Tagen.", None
    return True, None, None


def get_palette(priority: str) -> str:
    colors = {
        "JOKER": "#1a237e",
        "HIGH": "#1565c0",
        "LOW": "#64b5f6",
        "PLACEHOLDER": "#bbdefb",
    }
    return colors.get(priority, "#808080")


def _date_in_range(target: date, start: date, end: date) -> bool:
    return start <= target <= end


def _day_background(target: date) -> str:
    holiday_dates = get_holiday_dates()
    if target in holiday_dates:
        return "#fce4d6"
    for start, end, _ in get_blackout_ranges():
        if _date_in_range(target, start, end):
            return "#f4cccc"
    for row in load_master_data()["school_holidays"].to_dict(orient="records"):
        if _date_in_range(
            target,
            normalize_date(row["start_date"]),
            normalize_date(row["end_date"]),
        ):
            return "#fff2cc"
    if target.weekday() >= 5:
        return "#e7e6e6"
    return "#ffffff"


def availability_matrix(
    entries: list[dict[str, Any]],
    employee_ids: list[str],
    min_employees: list[int],
    holiday_adjust: list[int],
    display_mode: str,
    own_employee_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    employee_ids = sorted({str(employee_id) for employee_id in employee_ids})
    active_entries = [
        entry for entry in entries
        if not entry.get("is_deleted", False)
        and str(entry["employee_id"]) in employee_ids
    ]
    values: dict[str, list[Any]] = {}
    styles: dict[str, list[str]] = {}
    for month_number, month_name in enumerate(MONTHS, start=1):
        row_values: list[Any] = []
        row_styles: list[str] = []
        for day_number in range(1, 32):
            try:
                target = date(PLANNING_YEAR, month_number, day_number)
            except ValueError:
                row_values.append(pd.NA)
                row_styles.append("background-color: #f2f2f2; color: #999999;")
                continue
            holiday = target in get_holiday_dates()
            weekday = target.weekday()
            required = min_employees[weekday] if weekday < 5 and not holiday else 0
            if required and is_school_holiday(target):
                required = max(0, required + holiday_adjust[weekday])
            if target.weekday() >= 5 or holiday:
                row_values.append(pd.NA)
                row_styles.append("background-color: #d9d9d9; color: #777777;")
                continue
            standard_absent_ids = {
                employee_id
                for employee_id in employee_ids
                if target.weekday()
                in get_standard_absence_weekdays(
                    get_employee_map().get(employee_id, {})
                )
            }
            absent_ids = standard_absent_ids | {
                str(entry["employee_id"])
                for entry in active_entries
                if _date_in_range(
                    target,
                    normalize_date(entry["start_date"]),
                    normalize_date(entry["end_date"]),
                )
            }
            available = len(employee_ids) - len(absent_ids)
            difference = available - required
            own_entry = next(
                (
                    entry for entry in active_entries
                    if str(entry["employee_id"]) == str(own_employee_id)
                    and _date_in_range(
                        target,
                        normalize_date(entry["start_date"]),
                        normalize_date(entry["end_date"]),
                    )
                ),
                None,
            )
            if required == 0:
                background = "#d9d9d9"
                foreground = "#777777"
            elif difference < 0:
                background = "#ff0000"
                foreground = "#ffffff"
            elif own_entry:
                background = get_palette(own_entry["priority"])
                foreground = "#ffffff" if own_entry["priority"] in {"JOKER", "HIGH"} else "#12304a"
            else:
                background = _day_background(target)
                foreground = "#222222"
            row_values.append(required if display_mode == "Bedarf" else available)
            row_styles.append(
                f"background-color: {background}; color: {foreground}; text-align: center;"
            )
        values[month_name] = row_values
        styles[month_name] = row_styles
    columns = [str(day) for day in range(1, 32)]
    value_frame = pd.DataFrame.from_dict(
        values, orient="index", columns=columns
    ).astype("Int64")
    style_frame = pd.DataFrame.from_dict(
        styles, orient="index", columns=columns
    )
    return value_frame, style_frame


def render_availability_table(
    title: str,
    entries: list[dict[str, Any]],
    employee_ids: list[str],
    min_employees: list[int],
    holiday_adjust: list[int],
    display_mode: str,
    own_employee_id: str | None = None,
) -> None:
    st.markdown(f"#### {title}")
    if not employee_ids:
        st.info("Für diese Auswahl sind keine Mitarbeitenden vorhanden.")
        return
    values, styles = availability_matrix(
        entries,
        employee_ids,
        min_employees,
        holiday_adjust,
        display_mode,
        own_employee_id,
    )
    st.dataframe(
        values.style.apply(lambda _: styles, axis=None),
        width='stretch',
        height=470,
    )
    st.markdown(
        """
        <div style="display:flex; flex-wrap:wrap; gap:8px 16px; margin:6px 0 12px;">
          <strong>Legende</strong>
          <span style="background:#ffffff;border:1px solid #999;padding:2px 8px;border-radius:4px;">Ansicht: Bedarf oder Verfügbarkeit</span>
          <span style="background:#ff0000;color:white;padding:2px 8px;border-radius:4px;">Bedarf unterschritten</span>
          <span style="background:#d9d9d9;color:#777;padding:2px 8px;border-radius:4px;">Kein Bedarf</span>
          <span style="background:#1a237e;color:white;padding:2px 8px;border-radius:4px;">Eigener Joker</span>
          <span style="background:#1565c0;color:white;padding:2px 8px;border-radius:4px;">Eigener Urlaub: Hoch</span>
          <span style="background:#64b5f6;color:#12304a;padding:2px 8px;border-radius:4px;">Eigener Urlaub: Niedrig</span>
          <span style="background:#bbdefb;color:#12304a;padding:2px 8px;border-radius:4px;">Eigener Urlaub: Platzhalter</span>
          <span style="background:#f4cccc;padding:2px 8px;border-radius:4px;">Urlaubssperre</span>
          <span style="background:#fff2cc;padding:2px 8px;border-radius:4px;">Schulferien</span>
          <span style="background:#fce4d6;padding:2px 8px;border-radius:4px;">Feiertag</span>
          <span style="background:#e7e6e6;padding:2px 8px;border-radius:4px;">Wochenende</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_employee_view(employee_id: str, all_entries: list[dict[str, Any]]) -> None:
    st.header(f"Willkommen Mitarbeiter {employee_id}")
    summary = compute_employee_statistics(employee_id, all_entries)
    col1, col2, col3 = st.columns(3)
    col1.metric("Erholungsurlaub", f"{summary['recreation_days']} / 30 Tage")
    col2.metric("Joker + Hoch", f"{summary['high_priority_days']} / 20 Tage")
    col3.metric("Joker-Status", "Zugewiesen" if summary["joker_entry"] else "Verfügbar")

    if summary["recreation_days"] > 30:
        st.warning(f"Sie haben {summary['recreation_days']} Tage Erholungsurlaub geplant und überschreiten damit den üblichen Anspruch von 30 Tagen.")
    elif summary["recreation_days"] < 30 and not summary["has_parental_leave"]:
        st.error("Der Erholungsurlaub wurde nicht vollständig verplant, geben Sie ggf. Tage gegen Jahresende mit der Priorität 'Platzhalter' an.")
    elif summary["recreation_days"] < 30 and summary["has_parental_leave"]:
        st.warning(
            f"Es wurden nur {summary['recreation_days']} Tage statt des üblichen Anspruchs von 30 Urlaubstagen geplant. Bei Angabe von Elternzeiten kann der Urlaubsanspruch reduziert sein. Bitte prüfen Sie, ob der Jahresurlaub vollständig verplant wurde."
        )

    entries = get_employee_entries(employee_id, all_entries)
    if not entries:
        st.info("Noch keine Abwesenheiten erfasst.")

    df = pd.DataFrame(entries)
    if not df.empty:
        df["Startdatum"] = pd.to_datetime(df["start_date"]).dt.strftime("%d.%m.%Y")
        df["Enddatum"] = pd.to_datetime(df["end_date"]).dt.strftime("%d.%m.%Y")
        df["Abwesenheit"] = df["absence_code"].map(get_absence_label)
        df["Priorität"] = df["priority"].map(lambda value: PRIORITY_LABELS.get(value, value))
        df = df[["entry_id", "Startdatum", "Enddatum", "Abwesenheit", "Priorität", "workdays_count"]]
        df.columns = ["ID", "Startdatum", "Enddatum", "Abwesenheit", "Priorität", "Arbeitstage"]
        st.dataframe(df, width='stretch')

    st.subheader("Abwesenheit hinzufügen")
    with st.form("absence_form"):
        date_columns = st.columns(2)
        start_date = date_columns[0].date_input(
            "Startdatum", value=date(PLANNING_YEAR, 1, 1),
            min_value=date(PLANNING_YEAR, 1, 1), max_value=date(PLANNING_YEAR, 12, 31),
            format="DD.MM.YYYY",
        )
        end_date = date_columns[1].date_input(
            "Enddatum", value=date(PLANNING_YEAR, 1, 5),
            min_value=date(PLANNING_YEAR, 1, 1), max_value=date(PLANNING_YEAR, 12, 31),
            format="DD.MM.YYYY",
        )
        absence_options = get_absence_options()
        absence_code = date_columns[0].selectbox(
            "Abwesenheitstyp",
            [code for code, _ in absence_options],
            format_func=lambda code: get_absence_label(code),
        )
        priority_label = date_columns[1].radio(
            "Priorität", list(PRIORITY_FROM_LABEL), index=2, horizontal=True
        )
        priority = PRIORITY_FROM_LABEL[priority_label]
        add_absence = st.form_submit_button("Abwesenheit hinzufügen", width="stretch")

    if add_absence:
        workday_count = workdays_in_range(start_date, end_date, get_holiday_dates())
        form = {
            "entry_id": str(uuid.uuid4()),
            "employee_id": str(employee_id),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "absence_code": absence_code,
            "priority": priority,
            "workdays_count": workday_count,
            "is_submitted": False,
            "last_modified_utc": "",
            "is_deleted": False,
        }
        errors = validate_entry(employee_id, form, all_entries)
        if errors:
            for msg in errors:
                st.error(msg)
        else:
            get_pending_entries().append(form)
            st.success("Abwesenheit wurde für diese Sitzung vorgemerkt. Bitte speichern Sie den Entwurf.")
            st.rerun()

    st.subheader("Abwesenheiten löschen")
    if not entries:
        st.info("Keine Abwesenheiten zum Löschen vorhanden.")
    else:
        delete_columns = st.columns(2)
        with delete_columns[0]:
            entry_id_to_delete = st.text_input("ID der Abwesenheit", placeholder="UUID eingeben", width="stretch")   
        with delete_columns[1]:
            if st.button("Eine Abwesenheit löschen", width="stretch"):
                if entry_id_to_delete in {entry["entry_id"] for entry in entries}:
                    delete_single_entry_dialog(entry_id_to_delete)
                else:
                    st.error("Keine Abwesenheit mit dieser ID gefunden.")
        if st.button("🔥 Alle meine Abwesenheiten löschen", width="stretch"):
                delete_all_entries_dialog(employee_id, entries)
    
    st.subheader("Abwesenheiten speichern")
    save_columns = st.columns(2)
    save_draft = save_columns[0].button("Entwurf speichern", width="stretch")
    submit_plan = save_columns[1].button("Jahresurlaubsplan einreichen", width="stretch")

    if submit_plan:
        ok, blocker, warning = validate_submission(employee_id, all_entries)
        if not ok:
            if blocker:
                st.error(blocker)
            if warning:
                st.warning(warning)
        else:
            if warning:
                st.warning(warning)
            persist_pending_entries()
            all_entries = fetch_all_entries()
            for e in all_entries:
                if str(e["employee_id"]) == str(employee_id):
                    e["is_submitted"] = True
                    save_entry({**e, "is_submitted": True})
            st.success("Plan eingereicht.")
            st.rerun()

    if save_draft:
        if get_pending_entries():
            persist_pending_entries()
            st.success("Entwurf wurde in Google Sheets gespeichert.")
            st.rerun()
        st.info("Es gibt keine ungespeicherten Änderungen.")

    st.subheader("Verfügbarkeit")
    display_mode = "Bedarf" if st.toggle(
        "Bedarf statt Verfügbarkeit anzeigen",
        value=False,
        key="availability_display_mode",
    ) else "Verfügbarkeit"
    employee_map = get_employee_map()
    employee_team = employee_map[str(employee_id)]["team_id"]
    team_employee_ids = [
        other_id for other_id, employee in employee_map.items()
        if employee.get("team_id") == employee_team
    ]
    render_availability_table(
        f"Team: {get_team_label(employee_team)}",
        all_entries,
        team_employee_ids,
        get_min_employees("team", employee_team),
        get_holiday_adjust("team", employee_team),
        display_mode,
        own_employee_id=employee_id,
    )
    sub_skill = st.selectbox(
        "Kompetenz auswählen",
        employee_map[str(employee_id)].get("sub_skills", []),
        format_func=lambda skill: skill,
    )
    skill_employee_ids = [
        other_id for other_id, employee in employee_map.items()
        if sub_skill in employee.get("sub_skills", [])
    ]
    render_availability_table(
        f"Kompetenz: {get_skill_label(sub_skill)}",
        all_entries,
        skill_employee_ids,
        get_min_employees("skill", sub_skill),
        get_holiday_adjust("skill", sub_skill),
        display_mode,
        own_employee_id=employee_id,
    )


def render_admin_view(all_entries: list[dict[str, Any]]) -> None:
    st.header("Administration")
    employee_map = get_employee_map()
    employee_selector = st.selectbox("Mitarbeiter", ["ALLE"] + sorted({str(e["employee_id"]) for e in all_entries}))
    team_ids = [str(team["team_id"]) for team in load_master_data()["teams"]["teams"]]
    team_options = {"Alle Teams": "ALLE", **{get_team_label(team_id): team_id for team_id in team_ids}}
    team_label = st.selectbox("Team", list(team_options))
    team_filter = team_options[team_label]
    skills = sorted({skill for employee in employee_map.values() for skill in employee.get("sub_skills", [])})
    selected_skills = st.multiselect("Kompetenzen", skills, default=[])
    visible = all_entries
    if employee_selector != "ALLE":
        visible = [e for e in visible if str(e["employee_id"]) == employee_selector]
    if team_filter != "ALLE":
        visible = [
            e for e in visible if str(employee_map.get(str(e["employee_id"]), {}).get("team_id")) == team_filter
        ]
    if selected_skills:
        visible = [
            e for e in visible if any(skill in employee_map.get(str(e["employee_id"]), {}).get("sub_skills", []) for skill in selected_skills)
        ]

    st.dataframe(pd.DataFrame(visible), width='stretch')
    display_mode = "Bedarf" if st.toggle(
        "Bedarf statt Verfügbarkeit anzeigen",
        value=False,
        key="admin_availability_display_mode",
    ) else "Verfügbarkeit"
    selected_employee_ids = set(employee_map)
    if employee_selector != "ALLE":
        selected_employee_ids = {employee_selector}
    if team_filter != "ALLE":
        selected_employee_ids = {
            employee_id for employee_id in selected_employee_ids
            if employee_map.get(employee_id, {}).get("team_id") == team_filter
        }
    if selected_skills:
        selected_employee_ids = {
            employee_id
            for employee_id in selected_employee_ids
            if any(
                skill in employee_map.get(employee_id, {}).get("sub_skills", [])
                for skill in selected_skills
            )
        }
    if selected_employee_ids:
        selected_team = team_filter
        if selected_team == "ALLE":
            selected_team = employee_map[next(iter(selected_employee_ids))]["team_id"]
        render_availability_table(
            "Verfügbarkeit der Auswahl",
            all_entries,
            sorted(selected_employee_ids),
            get_min_employees("team", selected_team),
            get_holiday_adjust("team", selected_team),
            display_mode,
        )


def login_screen() -> str | None:
    st.title("Jahresurlaubsplanung")
    employees = get_employee_map()
    pin = st.text_input("PIN", type="password", max_chars=4)
    if st.button("Anmelden"):
        matching_employees = [
            employee
            for employee in employees.values()
            if str(employee.get("pin", "")) == str(pin).strip()
        ]
        if len(matching_employees) == 1:
            employee = matching_employees[0]
            st.session_state["employee_id"] = str(employee["employee_id"])
            st.session_state["is_admin"] = bool(employee.get("is_admin"))
            st.rerun()
        else:
            st.error("Ungültiger PIN.")
    return st.session_state.get("employee_id")


def main() -> None:
    st.set_page_config(layout="wide")
    all_entries = entries_with_pending_persistence(fetch_all_entries())
    if "employee_id" not in st.session_state:
        login_screen()
        return

    employee_id = str(st.session_state["employee_id"])
    is_admin = bool(st.session_state.get("is_admin", False))

    if is_admin and st.sidebar.checkbox("Admin mode"):
        render_admin_view(all_entries)
        return

    render_employee_view(employee_id, all_entries)


if __name__ == "__main__":
    main()
