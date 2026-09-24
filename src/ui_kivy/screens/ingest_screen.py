import asyncio
import os

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.anchorlayout import MDAnchorLayout
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDButton, MDButtonText
from kivymd.uix.label import MDLabel
from kivymd.uix.progressindicator import MDLinearProgressIndicator
from kivymd.uix.screen import MDScreen
from plyer import filechooser

from src.ui_kivy.utils.api_client import CRAGApiClient


class IngestScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ingest"
        self.api = CRAGApiClient()
        self._build_ui()

    def _build_ui(self):
        layout = MDBoxLayout(
            orientation="vertical",
            padding=dp(24),
            spacing=dp(16),
        )

        layout.add_widget(MDLabel(
            text="Загрузка документов",
            font_style="Title",
            role="large",
            size_hint_y=None,
            height=dp(48),
            halign="center",
        ))

        layout.add_widget(MDLabel(
            text="Поддерживаемые форматы: PDF, TXT, MD",
            font_style="Body",
            role="medium",
            size_hint_y=None,
            height=dp(32),
            halign="center",
            theme_text_color="Secondary",
        ))

        # Кнопка выбора файлов
        btn_row = MDAnchorLayout(
            size_hint_y=None,
            height=dp(64),
            anchor_x="center",
            anchor_y="center",
        )

        self.select_btn = MDButton(
            style="elevated",
            theme_width="Custom",
            size_hint=(None, None),
            width=dp(240),
            height=dp(52),
            on_release=self._select_files,

            # Белый фон
            theme_bg_color="Custom",
            md_bg_color=(1, 1, 1, 1),
        )

        btn_text = MDButtonText(
            text="Выбрать файлы",
            theme_text_color="Custom",
            text_color=(0, 0, 0, 1),
        )

        self.select_btn.add_widget(btn_text)
        btn_row.add_widget(self.select_btn)

        layout.add_widget(btn_row)

        # Прогресс
        self.progress_bar = MDLinearProgressIndicator(
            size_hint_y=None,
            height=dp(4),
            value=0,
        )
        self.progress_bar.opacity = 0
        layout.add_widget(self.progress_bar)

        # Статус
        self.status_label = MDLabel(
            text="",
            font_style="Body",
            role="large",
            size_hint_y=None,
            height=dp(48),
            halign="center",
        )
        layout.add_widget(self.status_label)

        layout.add_widget(MDBoxLayout())

        self.add_widget(layout)

    def _select_files(self, *args):
        filechooser.open_file(
            on_selection=self._on_files_selected,
            filters=["*.pdf", "*.txt", "*.md"],
            multiple=True,
        )

    def _on_files_selected(self, selection):
        if not selection:
            return
        self.select_btn.disabled = True
        self.progress_bar.opacity = 1
        asyncio.ensure_future(self._upload_files(selection))

    async def _upload_files(self, file_paths: list[str]):
        total = len(file_paths)

        for i, path in enumerate(file_paths):
            file_name = os.path.basename(path)
            Clock.schedule_once(
                lambda dt, n=file_name, idx=i: self._update_status(
                    f"Загружаю {n}... ({idx + 1}/{total})",
                    (idx / total) * 100,
                )
            )
            try:
                result = await self.api.upload_document(path, file_name)
                chunks = result.get("chunks_indexed", 0)
                Clock.schedule_once(
                    lambda dt, n=file_name, c=chunks: self._update_status(
                        f"{n} — {c} фрагментов",
                        ((i + 1) / total) * 100,
                    )
                )
            except Exception as e:
                Clock.schedule_once(
                    lambda dt, n=file_name, err=str(e): self._update_status(
                        f"{n} — ошибка: {err}",
                        ((i + 1) / total) * 100,
                    )
                )

        Clock.schedule_once(lambda dt: self._finish_upload())

    def _update_status(self, text: str, progress: float):
        self.status_label.text = text
        self.progress_bar.value = progress

    def _finish_upload(self):
        self.select_btn.disabled = False
        self.progress_bar.opacity = 0
        self.status_label.text = "Все файлы загружены"
