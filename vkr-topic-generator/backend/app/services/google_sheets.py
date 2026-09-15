import json
from typing import Iterable

from google.oauth2 import service_account
from googleapiclient.discovery import build

from ..config import get_settings
from ..models import GeneratedTopic
from ..utils import teacher_table_name

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class GoogleSheetsService:
    def __init__(self, runtime_settings=None) -> None:
        self.settings = runtime_settings or get_settings()

    def _credentials(self):
        raw = self.settings.google_service_account_json
        if not raw:
            raise RuntimeError("Google Sheets не настроен: добавьте JSON сервисного аккаунта в настройках")
        raw = raw.strip()
        if raw.startswith("{"):
            info = json.loads(raw)
            return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        return service_account.Credentials.from_service_account_file(raw, scopes=SCOPES)

    def _prepare_spreadsheet(self, sheets):
        configured_id = getattr(self.settings, "google_spreadsheet_id", None)
        if configured_id:
            spreadsheet = sheets.spreadsheets().get(
                spreadsheetId=configured_id,
                fields="spreadsheetId,spreadsheetUrl,sheets(properties(sheetId,title))",
            ).execute()
            spreadsheet_id = configured_id
            spreadsheet_url = spreadsheet.get("spreadsheetUrl") or f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            first_sheet = spreadsheet["sheets"][0]["properties"]
            return spreadsheet_id, spreadsheet_url, first_sheet["sheetId"], first_sheet["title"], False

        body = {
            "properties": {"title": "Темы ВКР — 09.03.01"},
            "sheets": [{"properties": {"title": "Темы ВКР", "gridProperties": {"frozenRowCount": 1}}}],
        }
        spreadsheet = sheets.spreadsheets().create(
            body=body, fields="spreadsheetId,spreadsheetUrl,sheets(properties(sheetId,title))"
        ).execute()
        spreadsheet_id = spreadsheet["spreadsheetId"]
        spreadsheet_url = spreadsheet.get("spreadsheetUrl") or f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        props = spreadsheet["sheets"][0]["properties"]
        return spreadsheet_id, spreadsheet_url, props["sheetId"], props["title"], True

    def create_topics_sheet(self, topics: Iterable[GeneratedTopic]) -> tuple[str, str, int]:
        topics = list(topics)
        if not topics:
            raise ValueError("Нет утверждённых тем для публикации")

        creds = self._credentials()
        sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
        drive = build("drive", "v3", credentials=creds, cache_discovery=False)

        spreadsheet_id, spreadsheet_url, sheet_id, sheet_title, created_new = self._prepare_spreadsheet(sheets)

        values = [["Преподаватель", "Тема", "ФИО студента", "Контакты преподавателя"]]
        for topic in topics:
            values.append([
                teacher_table_name(topic.teacher.full_name, topic.teacher.position),
                topic.title,
                "",
                topic.teacher.contact_text,
            ])

        # Если пользователь указал существующую таблицу, очищаем старые строки A:D,
        # чтобы новая генерация не смешивалась с предыдущей.
        sheets.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range=f"'{sheet_title}'!A:D",
            body={},
        ).execute()
        sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"'{sheet_title}'!A1:D",
            valueInputOption="RAW",
            body={"values": values},
        ).execute()

        requests = [
            {
                "updateSheetProperties": {
                    "properties": {"sheetId": sheet_id, "gridProperties": {"frozenRowCount": 1}},
                    "fields": "gridProperties.frozenRowCount",
                }
            },
            {
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1},
                    "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
                    "fields": "userEnteredFormat.textFormat.bold",
                }
            },
            {
                "autoResizeDimensions": {
                    "dimensions": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 4}
                }
            },
        ]
        sheets.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": requests}
        ).execute()

        if created_new and self.settings.google_share_with_email:
            drive.permissions().create(
                fileId=spreadsheet_id,
                body={"type": "user", "role": "writer", "emailAddress": self.settings.google_share_with_email},
                sendNotificationEmail=False,
            ).execute()

        return spreadsheet_id, spreadsheet_url, len(topics)
