import asyncio
import uuid

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.progressindicator import MDLinearProgressIndicator
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.textfield import MDTextField

from src.ui_kivy.components.message_bubble import MessageBubble
from src.ui_kivy.utils.api_client import CRAGApiClient


class ChatScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "chat"
        self.session_id = str(uuid.uuid4())
        self.api = CRAGApiClient()
        self._build_ui()

    def _build_ui(self):
        layout = MDBoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))

        header = MDLabel(
            font_style="Title",
            role="large",
            size_hint_y=None,
            height=dp(48),
            halign="center",
        )
        layout.add_widget(header)

        self.scroll = MDScrollView(
            size_hint=(1, 1),
            do_scroll_x=False,
            do_scroll_y=True,
        )

        self.messages_layout = MDBoxLayout(
            orientation="vertical",
            size_hint_x=1,
            size_hint_y=None,

            spacing=dp(8),
            padding=[dp(4), dp(8)],
        )

        self.messages_layout.bind(
            minimum_height=self.messages_layout.setter("height")
        )
        self.scroll.add_widget(self.messages_layout)
        layout.add_widget(self.scroll)

        self.progress_bar = MDLinearProgressIndicator(
            size_hint_y=None,
            height=dp(4),
            value=0,
        )
        self.progress_bar.opacity = 0
        layout.add_widget(self.progress_bar)

        self.status_label = MDLabel(
            text="",
            font_style="Label",
            role="small",
            size_hint_y=None,
            height=dp(20),
            halign="center",
            theme_text_color="Secondary",
        )
        layout.add_widget(self.status_label)

        input_row = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(56),
            spacing=dp(8),
        )
        self.input_field = MDTextField(
            hint_text="Задай вопрос...",
            mode="outlined",
            size_hint_x=1,
        )
        self.input_field.bind(
            on_text_validate=lambda x: self._send_message()
        )
        send_btn = MDIconButton(
            icon="send",
            on_release=lambda x: self._send_message(),
        )
        input_row.add_widget(self.input_field)
        input_row.add_widget(send_btn)
        layout.add_widget(input_row)

        self.add_widget(layout)

    def _send_message(self):
        text = self.input_field.text.strip()
        if not text:
            return

        self.input_field.text = ""
        self.input_field.disabled = True

        bubble = MessageBubble(text=text, is_user=True)
        self.messages_layout.add_widget(bubble)
        self._scroll_to_bottom()

        self.progress_bar.opacity = 1
        self.progress_bar.value = 5

        asyncio.ensure_future(self._process_query(text))

    async def _process_query(self, query: str):
        stage_progress = {
            "retrieve": 20,
            "grade": 50,
            "rewrite": 40,
            "fallback": 70,
            "generate": 85,
            "done": 100,
        }

        answer = ""
        source = ""

        try:
            async for event in self.api.stream_chat(query, self.session_id):
                stage = event.get("stage", "")
                message = event.get("message", "")
                progress = stage_progress.get(stage, event.get("progress", 0))

                Clock.schedule_once(
                    lambda dt, p=progress, m=message: self._update_progress(p, m)
                )

                if stage == "done":
                    answer = event.get("answer", "")
                    source = event.get("source", "")

        except Exception as e:
            answer = f"Ошибка подключения к серверу: {e}"
            source = "no_context"

        Clock.schedule_once(lambda dt: self._add_response(answer, source))

    def _update_progress(self, value: float, status: str):
        self.progress_bar.value = value
        self.status_label.text = status

    def _add_response(self, answer: str, source: str):
        self.progress_bar.opacity = 0
        self.progress_bar.value = 0
        self.status_label.text = ""
        self.input_field.disabled = False

        bubble = MessageBubble(text=answer, is_user=False, source=source)
        self.messages_layout.add_widget(bubble)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        Clock.schedule_once(
            lambda dt: setattr(self.scroll, "scroll_y", 0), 0.1
        )
