import asyncio

from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.navigationbar import (
    MDNavigationBar,
    MDNavigationItem,
    MDNavigationItemIcon,
    MDNavigationItemLabel,
)
from kivymd.uix.screenmanager import MDScreenManager

from src.ui_kivy.screens.admin_screen import AdminScreen
from src.ui_kivy.screens.chat_screen import ChatScreen
from src.ui_kivy.screens.ingest_screen import IngestScreen
from src.ui_kivy.utils.platform import is_mobile


class CRAGApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

        layout = MDBoxLayout(orientation="vertical")

        self.sm = MDScreenManager()
        self.sm.add_widget(ChatScreen())

        if not is_mobile():
            self.sm.add_widget(IngestScreen())
            self.sm.add_widget(AdminScreen())

        layout.add_widget(self.sm)

        if not is_mobile():
            nav_bar = MDNavigationBar(on_switch_tabs=self._switch_tab)

            chat_item = MDNavigationItem()
            chat_item.add_widget(MDNavigationItemIcon(icon="chat"))
            chat_item.add_widget(MDNavigationItemLabel(text="Чат"))
            nav_bar.add_widget(chat_item)

            ingest_item = MDNavigationItem()
            ingest_item.add_widget(MDNavigationItemIcon(icon="file-upload"))
            ingest_item.add_widget(MDNavigationItemLabel(text="Документы"))
            nav_bar.add_widget(ingest_item)

            admin_item = MDNavigationItem()
            admin_item.add_widget(MDNavigationItemIcon(icon="chart-bar"))
            admin_item.add_widget(MDNavigationItemLabel(text="Статистика"))
            nav_bar.add_widget(admin_item)

            layout.add_widget(nav_bar)

        return layout

    def _switch_tab(self, bar, item, item_icon, item_text):
        screen_map = {
            "Чат": "chat",
            "Документы": "ingest",
            "Статистика": "admin",
        }
        self.sm.current = screen_map.get(item_text, "chat")

    async def async_run_wrapper(self):
        await self.async_run(async_lib="asyncio")

def run():
    app = CRAGApp()
    asyncio.run(app.async_run_wrapper())
