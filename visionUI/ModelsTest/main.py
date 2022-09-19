import PyQt5
from PyQt5 import uic
import os
from pathlib import Path

from PyQt5.QtSql import QSqlDatabase, QSqlTableModel, QSqlQueryModel, QSqlDriver
from PyQt5.QtWidgets import QApplication, QDataWidgetMapper

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
)


class ModelFactory:
    # Yep, a factory. We need that because Qt does not automatically notify SQL models when DB changes occur, so I am
    # keeping a list of these to refresh them in a kind of convoluted way. Makes you wonder why Qt has these models at
    # all
    def __init__(self, db):
        self.models = list()
        self.mappers = list() # just prevents mappers from being garbage collected. Connections don't seem to be enough
        if not db.isOpen():
            db.open()
        self.db = db
        if db.driver().hasFeature(QSqlDriver.EventNotifications):
            db.driver().subscribeToNotification("stuff")
            db.driver().notification.connect(self.db_changed)
        else:
            print("Driver does NOT support database event notifications")
            return

    # Links a widget to a SQL table in an editable fashion
    def table_editor(self, widget, table_name):
        model = QSqlTableModel()
        model.setTable(table_name)
        widget.setModel(model)
        model.select()
        self.models.append(model)

    # Links a widget to a SQL query and creates a mapper if necessary
    def readonly_query(self, widget, query, property_name=b"text"):
        model = QSqlQueryModel()
        model.setQuery(query)
        try:
            widget.setModel(model)
        except AttributeError:
            mapper = QDataWidgetMapper()
            mapper.setModel(model)
            mapper.addMapping(widget, 0, property_name);
            mapper.toFirst()
            model.modelReset.connect(mapper.toFirst)
            model.dataChanged.connect(mapper.toFirst)
            self.mappers.append(mapper)
        self.models.append(model)

    def db_changed(self):
        for m in self.models:
            # Convoluted way to invalidate cached data. A dedicated function is weirdly absent from the Qt API
            if isinstance(m, QSqlTableModel):
                m.select()
            elif isinstance(m, QSqlQueryModel):
                q = m.query().executedQuery()
                m.setQuery(q)

class App(QApplication):
    def __init__(self):
        super().__init__([])
        db = QSqlDatabase.addDatabase("QSQLITE")
        db.setDatabaseName("modelsTest.db")
        self.mf = ModelFactory(db)

        form, window = uic.loadUiType("modelsTest.ui")
        self.window = window()
        self.form = form()
        self.form.setupUi(self.window)
        self.models = list()

        self.mf.table_editor(self.form.tableView, "stuff")
        self.mf.readonly_query(self.form.listView, "SELECT name FROM stuff")
        self.mf.readonly_query(self.form.sumLabel, "SELECT sum(value) FROM stuff")

        self.form.createButton.pressed.connect(self.create_row)
        self.form.deleteButton.pressed.connect(self.remove_row)
        self.window.show()

    def remove_row(self):
        ind = self.form.tableView.selectedIndexes()
        if len(ind) == 0:
            return
        ind = ind[0].row()
        model = self.form.tableView.model()
        model.deleteRowFromTable(ind)
        model.submitAll()
        model.select()

    def create_row(self):
        cm = self.form.tableView.model()
        row = cm.rowCount()
        cm.insertRows(row, 1)
        cm.submitAll()
        self.form.tableView.reset()


app = App()
app.exec_()


