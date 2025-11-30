#!/usr/bin/env python3
"""Simple NiceGUI test"""

from nicegui import ui

ui.label("Hello NiceGUI!")

ui.run(port=8090, title="Simple Test")
