"""Fenêtre « Notice des paramètres » (Aide, Maj+F1) et rattachement des champs à leur fiche."""

from __future__ import annotations

import html

from PySide6 import QtCore, QtWidgets

from ...i18n import t
from ...notice import BY_KEY, ENTRIES, SECTIONS, Entry

KEY_ROLE = QtCore.Qt.ItemDataRole.UserRole


def attach(widget: QtWidgets.QWidget, key: str) -> None:
    """Infobulle tirée de la notice ; Maj+F1 sur ce champ ouvre sa fiche."""
    entry = BY_KEY[key]
    widget.setProperty("notice_key", key)
    widget.setToolTip(entry.text("summary") + "\n" + t("Shift+F1: parameter guide"))


def key_for(widget: QtWidgets.QWidget | None) -> str | None:
    """Fiche du champ qui a le focus (ou de son parent le plus proche qui en a une)."""
    while widget is not None:
        finder = getattr(widget, "notice_key", None)
        key = finder() if callable(finder) else widget.property("notice_key")
        if key:
            return key
        widget = widget.parentWidget()
    return None


def entry_html(entry: Entry) -> str:
    label = html.escape(entry.title())
    rows = []
    if entry.unit:
        rows.append((t("Unit"), t(entry.unit)))
    if entry.default is not None:
        rows.append((t("Default"), entry.default()))
    if entry.limits is not None:
        rows.append((t("Allowed values"), entry.limits()))
    if entry.twin_key:
        rows.append((t("Key in twin.yaml"), entry.twin_key))
    table = "".join(f"<tr><td style='padding-right:12px'><b>{html.escape(name)}</b></td><td>{html.escape(value)}</td></tr>"
                    for name, value in rows)
    details = f"<p>{html.escape(entry.text('details'))}</p>" if entry.details else ""
    return (f"<h3>{label}</h3><p style='color:gray'>{html.escape(t(entry.section))}</p>"
            f"<p>{html.escape(entry.text('summary'))}</p>{details}<table style='margin-top:10px'>{table}</table>")


class NoticeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Parameter guide"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText(t("Search a parameter"))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._show_item)
        self._items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        sections = {}
        for section in SECTIONS:
            sections[section] = QtWidgets.QTreeWidgetItem(self.tree, [t(section)])
            sections[section].setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        for entry in ENTRIES:
            item = QtWidgets.QTreeWidgetItem(sections[entry.section], [entry.title()])
            item.setToolTip(0, entry.text("summary"))
            item.setData(0, KEY_ROLE, entry.key)
            self._items[entry.key] = item
        self.tree.expandAll()
        self.page = QtWidgets.QTextBrowser()
        self.page.setOpenExternalLinks(False)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.search)
        left_layout.addWidget(self.tree, 1)
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.page)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 600])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.resize(960, 600)
        self.show_entry(ENTRIES[0].key)

    def show_entry(self, key: str) -> None:
        self.search.clear()
        self.tree.setCurrentItem(self._items[key])

    def current_key(self) -> str | None:
        item = self.tree.currentItem()
        return item.data(0, KEY_ROLE) if item is not None else None

    def _show_item(self, item: QtWidgets.QTreeWidgetItem | None, _previous=None) -> None:
        key = item.data(0, KEY_ROLE) if item is not None else None
        if key:
            self.page.setHtml(entry_html(BY_KEY[key]))

    def _filter(self, text: str) -> None:
        needle = text.strip().casefold()
        for section_index in range(self.tree.topLevelItemCount()):
            section = self.tree.topLevelItem(section_index)
            visible = 0
            for i in range(section.childCount()):
                item = section.child(i)
                entry = BY_KEY[item.data(0, KEY_ROLE)]
                haystack = " ".join((entry.title(), entry.text("summary"), entry.text("details"), entry.twin_key)).casefold()
                item.setHidden(bool(needle) and needle not in haystack)
                visible += not item.isHidden()
            section.setHidden(visible == 0)

    def visible_keys(self) -> list[str]:
        return [key for key, item in self._items.items() if not item.isHidden() and not item.parent().isHidden()]
