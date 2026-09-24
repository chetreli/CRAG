from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel


class MessageBubble(MDBoxLayout):

    def __init__(
        self,
        text: str,
        is_user: bool,
        source: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.is_user = is_user
        self.source = source

        # Вся строка сообщения занимает ширину чата
        self.orientation = "horizontal"
        self.size_hint_x = 1
        self.size_hint_y = None

        self.padding = [dp(8), dp(4)]
        self.spacing = 0

        # =========================================================
        # BUBBLE
        # =========================================================

        self.card = MDCard(
            size_hint=(None, None),
            radius=[dp(16)],
            md_bg_color=(
                (0.20, 0.60, 1.0, 1)
                if is_user
                else (0.15, 0.15, 0.15, 1)
            ),
        )

        # =========================================================
        # CONTENT
        # =========================================================

        self.content = MDBoxLayout(
            orientation="vertical",
            size_hint=(None, None),
            padding=[
                dp(12),
                dp(10),
                dp(12),
                dp(10),
            ],
            spacing=dp(4),
        )

        # =========================================================
        # TEXT
        # =========================================================

        self.label = MDLabel(
            text=text,
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),

            font_style="Body",
            role="large",

            size_hint=(None, None),

            halign="left",
            valign="top",
        )

        self.content.add_widget(self.label)

        # =========================================================
        # SOURCE
        # =========================================================

        self.source_label = None

        if source and not is_user:

            badge_map = {
                "local": "[Локальная база]",
                "web": "[Веб-поиск]",
                "no_context": "[Без контекста]",
            }

            self.source_label = MDLabel(
                text=badge_map.get(source, source),

                theme_text_color="Custom",
                text_color=(0.7, 0.7, 0.7, 1),

                font_style="Label",
                role="small",

                size_hint=(None, None),
                height=dp(18),
            )

            self.content.add_widget(self.source_label)

        self.card.add_widget(self.content)

        # =========================================================
        # КОНТЕЙНЕРЫ СЛЕВА И СПРАВА
        # =========================================================

        self.left_spacer = MDBoxLayout(
            size_hint=(None, 1),
            width=0,
        )

        self.right_spacer = MDBoxLayout(
            size_hint=(None, 1),
            width=0,
        )

        self.add_widget(self.left_spacer)
        self.add_widget(self.card)
        self.add_widget(self.right_spacer)

        # Пересчёт при изменении ширины окна
        self.bind(width=self._schedule_recalculate)

        # Первый расчёт после размещения виджета
        Clock.schedule_once(
            lambda dt: self._schedule_recalculate(),
            0,
        )

    # =============================================================
    # ПЛАНИРОВАНИЕ ПЕРЕСЧЁТА
    # =============================================================

    def _schedule_recalculate(self, *args):
        Clock.schedule_once(
            lambda dt: self._recalculate(),
            0,
        )

    # =============================================================
    # ОСНОВНОЙ РАСЧЁТ
    # =============================================================

    def _recalculate(self):

        if not self.parent:
            return

        parent_width = self.parent.width

        if parent_width <= 0:
            return

        # ---------------------------------------------------------
        # Ширина доступной области
        # ---------------------------------------------------------

        available_width = (
            parent_width
            - self.padding[0]
            - self.padding[2]
        )

        if available_width <= 0:
            return

        # ---------------------------------------------------------
        # Максимум bubble = 70% ширины чата
        # ---------------------------------------------------------

        max_bubble_width = available_width * 0.70

        # ---------------------------------------------------------
        # Padding внутри bubble
        # ---------------------------------------------------------

        horizontal_padding = dp(24)

        # =========================================================
        # 1. Узнаём естественную ширину текста
        # =========================================================

        self.label.text_size = (None, None)

        # Даем Kivy обновить texture
        Clock.schedule_once(
            lambda dt: self._finish_width_calculation(
                max_bubble_width,
                horizontal_padding,
            ),
            0,
        )

    # =============================================================
    # РАСЧЁТ ШИРИНЫ ПОСЛЕ ОБНОВЛЕНИЯ TEXTURE
    # =============================================================

    def _finish_width_calculation(
        self,
        max_bubble_width,
        horizontal_padding,
    ):

        # ---------------------------------------------------------
        # Естественная ширина текста
        # ---------------------------------------------------------

        natural_text_width = self.label.texture_size[0]

        # ---------------------------------------------------------
        # Максимальная ширина текста внутри bubble
        #
        # Вот здесь max_text_width действительно используется.
        # ---------------------------------------------------------

        max_text_width = (
            max_bubble_width
            - horizontal_padding
        )

        # ---------------------------------------------------------
        # Текст не должен быть меньше 40dp
        # ---------------------------------------------------------

        natural_text_width = max(
            natural_text_width,
            dp(40),
        )

        # ---------------------------------------------------------
        # Выбираем ширину текста:
        #
        # короткий текст -> естественная ширина
        #
        # длинный текст -> максимум 70% bubble
        # ---------------------------------------------------------

        text_width = min(
            natural_text_width,
            max_text_width,
        )

        # ---------------------------------------------------------
        # Bubble = текст + padding
        # ---------------------------------------------------------

        bubble_width = (
            text_width
            + horizontal_padding
        )

        # Минимальный размер bubble
        bubble_width = max(
            bubble_width,
            dp(70),
        )

        # ---------------------------------------------------------
        # Теперь задаём реальную ширину текста.
        #
        # Если текст длинный — MDLabel переносит его.
        # ---------------------------------------------------------

        self.label.width = text_width

        self.label.text_size = (
            text_width,
            None,
        )

        # =========================================================
        # ВАЖНО:
        #
        # После изменения text_size высота texture меняется.
        # Поэтому окончательный размер вычисляем следующим кадром.
        # =========================================================

        Clock.schedule_once(
            lambda dt: self._finish_height_calculation(
                bubble_width,
                text_width,
            ),
            0,
        )

    # =============================================================
    # РАСЧЁТ ВЫСОТЫ
    # =============================================================

    def _finish_height_calculation(
        self,
        bubble_width,
        text_width,
    ):

        # ---------------------------------------------------------
        # Высота текста после переноса строк
        # ---------------------------------------------------------

        self.label.height = self.label.texture_size[1]

        # ---------------------------------------------------------
        # Source
        # ---------------------------------------------------------

        source_height = 0

        if self.source_label:

            self.source_label.width = text_width
            source_height = self.source_label.height

        # ---------------------------------------------------------
        # Высота content
        # ---------------------------------------------------------

        content_height = (
            self.label.height
            + dp(20)
        )

        if self.source_label:
            content_height += (
                dp(4)
                + source_height
            )

        # ---------------------------------------------------------
        # Размер content
        # ---------------------------------------------------------

        self.content.width = bubble_width
        self.content.height = content_height

        # ---------------------------------------------------------
        # Размер bubble
        # ---------------------------------------------------------

        self.card.width = bubble_width
        self.card.height = content_height

        # ---------------------------------------------------------
        # Высота всей строки
        # ---------------------------------------------------------

        self.height = content_height + dp(8)

        # =========================================================
        # ПОЗИЦИЯ
        # =========================================================

        if not self.parent:
            return

        available_width = (
            self.parent.width
            - self.padding[0]
            - self.padding[2]
        )

        free_space = max(
            0,
            available_width - bubble_width,
        )

        if self.is_user:

            # Пользователь → справа
            self.left_spacer.width = free_space
            self.right_spacer.width = 0

        else:

            # Бот → слева
            self.left_spacer.width = 0
            self.right_spacer.width = free_space
