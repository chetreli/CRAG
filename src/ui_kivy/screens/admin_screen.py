import asyncio

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDButton, MDButtonText, MDIconButton
from kivymd.uix.divider import MDDivider
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView

from src.ui_kivy.utils.api_client import CRAGApiClient


class DocumentRow(MDBoxLayout):
    """Строка документа в списке."""
    def __init__(self, file_name: str, chunks: int, on_delete, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint_y = None
        self.height = dp(56)
        self.padding = [dp(12), dp(8)]
        self.spacing = dp(8)

        # Имя файла
        name_label = MDLabel(
            text=file_name,
            font_style="Body",
            role="large",
            size_hint_x=0.6,
            valign="center",
        )
        self.add_widget(name_label)

        # Количество чанков
        chunks_label = MDLabel(
            text=f"{chunks} фрагм.",
            font_style="Label",
            role="medium",
            size_hint_x=0.25,
            halign="center",
            valign="center",
            theme_text_color="Secondary",
        )
        self.add_widget(chunks_label)

        # Кнопка удаления
        delete_btn = MDIconButton(
            icon="delete",
            size_hint_x=None,
            width=dp(40),
            theme_icon_color="Custom",
            icon_color=(1, 0.3, 0.3, 1),
            on_release=lambda x: on_delete(file_name),
        )
        self.add_widget(delete_btn)


class AdminScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "admin"
        self.api = CRAGApiClient()
        self._build_ui()

    def _build_ui(self):
        layout = MDBoxLayout(
            orientation="vertical",
            padding=dp(12),
            spacing=dp(8),
        )

        # Заголовок + кнопка обновить
        header_row = MDBoxLayout(
            size_hint_y=None,
            height=dp(48),
            spacing=dp(8),
        )
        header_row.add_widget(MDLabel(
            text="База знаний",
            font_style="Title",
            role="large",
        ))
        refresh_btn = MDIconButton(
            icon="refresh",
            on_release=lambda x: self._load_stats(),
        )
        header_row.add_widget(refresh_btn)
        layout.add_widget(header_row)

        # Сводка
        self.summary_label = MDLabel(
            text="Загрузка...",
            font_style="Label",
            role="medium",
            size_hint_y=None,
            height=dp(28),
            theme_text_color="Secondary",
        )
        layout.add_widget(self.summary_label)

        layout.add_widget(MDDivider())

        # Заголовки колонок
        cols_header = MDBoxLayout(
            size_hint_y=None,
            height=dp(32),
            padding=[dp(12), dp(4)],
            spacing=dp(8),
        )
        cols_header.add_widget(MDLabel(
            text="Документ",
            font_style="Label",
            role="large",
            size_hint_x=0.6,
            theme_text_color="Secondary",
        ))
        cols_header.add_widget(MDLabel(
            text="Фрагменты",
            font_style="Label",
            role="large",
            size_hint_x=0.25,
            halign="center",
            theme_text_color="Secondary",
        ))
        cols_header.add_widget(MDBoxLayout(size_hint_x=None, width=dp(40)))
        layout.add_widget(cols_header)

        layout.add_widget(MDDivider())

        # Список документов
        self.scroll = MDScrollView(size_hint_y=1)
        self.docs_layout = MDBoxLayout(
            orientation="vertical",
            size_hint_y=None,
            spacing=dp(2),
        )
        self.docs_layout.bind(
            minimum_height=self.docs_layout.setter("height")
        )
        self.scroll.add_widget(self.docs_layout)
        layout.add_widget(self.scroll)

        # Статус удаления
        self.status_label = MDLabel(
            text="",
            font_style="Label",
            role="small",
            size_hint_y=None,
            height=dp(24),
            halign="center",
        )
        layout.add_widget(self.status_label)

        self.add_widget(layout)

        layout.add_widget(MDDivider())

        layout.add_widget(MDLabel(
            text="Режим оценки релевантности",
            font_style="Label",
            role="large",
            size_hint_y=None,
            height=dp(32),
            theme_text_color="Secondary",
        ))

        self.mode_row = MDBoxLayout(
    size_hint_y=None,
    height=dp(56),
    spacing=dp(8),
    padding=[dp(12), dp(4)],
)

        self.llm_btn = MDButton(
            style="filled",

            # Не даём KivyMD самостоятельно менять ширину
            theme_width="Custom",

            size_hint_x=0.5,
            height=dp(48),

            theme_bg_color="Custom",
            md_bg_color=(0.30, 0.30, 0.34, 1),

            on_release=lambda x: self._set_grader_mode("llm"),
        )

        self.llm_text = MDButtonText(
            text="Точный (LLM)",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
        )

        self.llm_btn.add_widget(self.llm_text)


        self.emb_btn = MDButton(
            style="filled",

            # Такая же ширина, как у первой
            theme_width="Custom",

            size_hint_x=0.5,
            height=dp(48),

            theme_bg_color="Custom",
            md_bg_color=(0.30, 0.30, 0.34, 1),

            on_release=lambda x: self._set_grader_mode("embedding"),
        )

        self.emb_text = MDButtonText(
            text="Быстрый (Embedding)",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
        )

        self.emb_btn.add_widget(self.emb_text)


        self.mode_row.add_widget(self.llm_btn)
        self.mode_row.add_widget(self.emb_btn)
        layout.add_widget(self.mode_row)

        self.mode_label = MDLabel(
            text="Текущий режим: загрузка...",
            font_style="Label",
            role="small",
            size_hint_y=None,
            height=dp(24),
            halign="center",
            theme_text_color="Secondary",
        )
        layout.add_widget(self.mode_label)
        # Загружаем данные при старте
        Clock.schedule_once(lambda dt: self._load_stats(), 0.5)
        Clock.schedule_once(lambda dt: asyncio.ensure_future(self._load_current_mode()), 0.8)

    def _load_stats(self):
        self.summary_label.text = "Обновление..."
        self.docs_layout.clear_widgets()
        asyncio.ensure_future(self._fetch_stats())

    async def _fetch_stats(self):
        try:
            data = await self.api.get_documents_stats()
            Clock.schedule_once(lambda dt: self._render_stats(data))
        except Exception as e:
            err_msg = str(e)
            Clock.schedule_once(
                lambda dt, err=err_msg: setattr(
                    self.summary_label, "text", f"Ошибка: {err_msg}"
                )
            )

    def _render_stats(self, data: dict):
        total_chunks = data.get("total_chunks", 0)
        total_docs = data.get("total_documents", 0)
        documents = data.get("documents", [])

        self.summary_label.text = (
            f"Всего документов: {total_docs}   |   "
            f"Всего фрагментов: {total_chunks}"
        )

        self.docs_layout.clear_widgets()
        for doc in documents:
            row = DocumentRow(
                file_name=doc["file_name"],
                chunks=doc["chunks"],
                on_delete=self._delete_document,
            )
            self.docs_layout.add_widget(row)
            self.docs_layout.add_widget(MDDivider())

    def _delete_document(self, file_name: str):
        self.status_label.text = f"Удаляю {file_name}..."
        asyncio.ensure_future(self._do_delete(file_name))

    async def _do_delete(self, file_name: str):
        try:
            await self.api.delete_document(file_name)
            Clock.schedule_once(
                lambda dt: setattr(
                    self.status_label, "text",
                    f"Удалено: {file_name}"
                )
            )
            Clock.schedule_once(lambda dt: self._load_stats(), 0.5)
        except Exception as e:
            err_msg = str(e)
            Clock.schedule_once(lambda dt, err = err_msg: setattr(self.status_label, "text", f"Ошибка удаления: {err_msg}"))

    def _set_grader_mode(self, mode: str):
        asyncio.ensure_future(self._do_set_mode(mode))

    async def _do_set_mode(self, mode: str):
        try:
            await self.api.set_grader_mode(mode)
            Clock.schedule_once(lambda dt: self._update_mode_ui(mode))
        except Exception as e:
            err = str(e)
            Clock.schedule_once(
                lambda dt, err=err: setattr(
                    self.mode_label, "text", f"Ошибка: {err}"
                )
            )

    def _update_mode_ui(self, mode: str):
        descriptions = {
            "llm": "Текущий режим: Точный (LLM) — ~20 сек/запрос",
            "embedding": "Текущий режим: Быстрый (Embedding) — ~1 сек/запрос",
        }
        self.mode_label.text = descriptions.get(mode, mode)

        gray = (0.30, 0.30, 0.34, 1)
        white = (1, 1, 1, 1)
        black = (0, 0, 0, 1)

        if mode == "llm":
            self.llm_btn.md_bg_color = white
            self.llm_text.text_color = black
            self.emb_btn.md_bg_color = gray
            self.emb_text.text_color = white

        else:
            self.llm_btn.md_bg_color = gray
            self.llm_text.text_color = white
            self.emb_btn.md_bg_color = white
            self.emb_text.text_color = black

    async def _load_current_mode(self):
            try:
                async with __import__("aiohttp").ClientSession() as session:
                    async with session.get(
                        f"{self.api.base_url}/grader/mode",
                        timeout=__import__("aiohttp").ClientTimeout(total=5),
                    ) as response:
                        data = await response.json()
                        mode = data.get("mode", "llm")
                        Clock.schedule_once(lambda dt: self._update_mode_ui(mode))
            except Exception:
                pass
