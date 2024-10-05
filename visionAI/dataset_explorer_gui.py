import os
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QPushButton, QFileDialog, QInputDialog, QMessageBox, QSplitter, QMenuBar, QAction
from PyQt5.QtCore import Qt, QSettings, pyqtSignal
from annotation_gui import AnnotationWidget
from database import DatabaseManager

"""
Dataset Explorer GUI

This module provides a graphical user interface for exploring, creating, and editing datasets used in image annotation projects.
It integrates with existing AnnotationGUI and Database classes for seamless interaction with the dataset.

Design decisions and constraints:
1. Use PyQt5 for the graphical interface to maintain consistency with existing code.
2. Integrate AnnotationWidget for image display and annotation.
3. Use the Database class for all database interactions.
4. Implement a single-window interface with a split view: image list on the left, annotation widget on the right.
5. Provide checkable buttons for toggling between candidate and keyframe images.
6. Allow users to open a dataset folder, create new datasets, and import images through a menu.
7. Remember and automatically open the last used dataset.
8. Include a resizable splitter between the left panel (image list) and the right panel (annotation widget) for flexible layout adjustment.

The main class, DatasetExplorerGUI, inherits from QWidget and encapsulates all GUI-related functionality.
"""

class DatasetExplorerGUI(QWidget):
    dataset_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.db_manager = None
        self.settings = QSettings("YourCompany", "DatasetExplorer")
        
        self.init_ui()
        self.load_last_dataset()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Dataset Explorer')
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()

        # Menu bar
        self.create_menu()

        # Main content area with splitter
        self.splitter = QSplitter(Qt.Horizontal)

        # Left panel: toggle buttons and image list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Toggle buttons
        toggle_layout = QHBoxLayout()
        self.candidates_button = QPushButton('Show Candidates')
        self.candidates_button.setCheckable(True)
        self.candidates_button.setChecked(True)
        self.candidates_button.clicked.connect(self.toggle_candidates)
        toggle_layout.addWidget(self.candidates_button)

        self.keyframes_button = QPushButton('Show Keyframes')
        self.keyframes_button.setCheckable(True)
        self.keyframes_button.clicked.connect(self.toggle_keyframes)
        toggle_layout.addWidget(self.keyframes_button)

        left_layout.addLayout(toggle_layout)

        # Image list
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.image_list.currentItemChanged.connect(self.on_image_selected)  # Add this line
        left_layout.addWidget(self.image_list)

        self.splitter.addWidget(left_widget)

        # Annotation widget
        self.annotation_widget = AnnotationWidget()
        self.annotation_widget.annotation_changed.connect(self.save_annotations)
        self.splitter.addWidget(self.annotation_widget)

        # Set initial sizes for splitter (adjust as needed)
        self.splitter.setSizes([400, 800])

        main_layout.addWidget(self.splitter)

        self.setLayout(main_layout)

        # Connect the dataset_changed signal to update the UI
        self.dataset_changed.connect(self.update_ui)

    def create_menu(self):
        menubar = QMenuBar(self)
        
        # Menu
        main_menu = menubar.addMenu('Menu')
        
        load_action = QAction('Load Dataset', self)
        load_action.triggered.connect(self.load_dataset)
        main_menu.addAction(load_action)
        
        create_action = QAction('Create New Dataset', self)
        create_action.triggered.connect(self.create_new_dataset)
        main_menu.addAction(create_action)
        
        import_action = QAction('Import Images', self)
        import_action.triggered.connect(self.import_images)
        main_menu.addAction(import_action)
        
        import_old_db_action = QAction('Import Old DB', self)
        import_old_db_action.triggered.connect(self.import_old_db)
        main_menu.addAction(import_old_db_action)

    def toggle_candidates(self):
        if self.candidates_button.isChecked():
            self.keyframes_button.setChecked(False)
            self.load_candidate_images()

    def toggle_keyframes(self):
        if self.keyframes_button.isChecked():
            self.candidates_button.setChecked(False)
            self.load_keyframe_images()

    def load_candidate_images(self):
        self.image_list.clear()
        if self.db_manager:
            candidates = self.db_manager.get_candidate_images()
            sorted_candidates = sorted(candidates, key=lambda img: img.path.lower())
            for img in sorted_candidates:
                self.image_list.addItem(img.path)

    def load_keyframe_images(self):
        self.image_list.clear()
        if self.db_manager:
            keyframes = self.db_manager.get_keyframe_images()
            sorted_keyframes = sorted(keyframes, key=lambda img: img.path.lower())
            for img in sorted_keyframes:
                self.image_list.addItem(img.path)

    def load_last_dataset(self):
        """Load the last opened dataset if available"""
        last_dataset = self.settings.value("last_dataset")
        if last_dataset and os.path.exists(last_dataset):
            self.load_dataset(last_dataset)

    def load_dataset(self, path=None):
        """Open a file dialog to select and load a dataset"""
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self.dataset_path = path
            self.db_manager = DatabaseManager(os.path.join(self.dataset_path, 'dataset.db'))
            self.dataset_changed.emit(self.dataset_path)
            self.settings.setValue("last_dataset", self.dataset_path)

    def update_ui(self):
        """Update the UI with dataset information"""
        self.load_candidate_images()

    def on_image_selected(self, item):
        if item is None:
            return
        
        image_path = item.text()
        self.current_image = image_path
        full_path = os.path.join(self.dataset_path, image_path)
        self.annotation_widget.load_image(full_path)
        
        # Load existing annotations
        annotations = self.db_manager.get_image_annotations(image_path)
        self.annotation_widget.load_annotations(annotations)
        
        # Print the loaded annotations
        print(f"Loaded annotations for {image_path}:")
        for ann in annotations:
            print(f"  {ann}")

    def create_new_dataset(self):
        """Create a new dataset with a given name"""
        name, ok = QInputDialog.getText(self, 'Create New Dataset', 'Enter dataset name:')
        if ok and name:
            root_dir = QFileDialog.getExistingDirectory(self, "Select Root Directory for New Dataset")
            if root_dir:
                dataset_path = os.path.join(root_dir, name)
                try:
                    os.makedirs(dataset_path)
                    os.makedirs(os.path.join(dataset_path, 'candidates'))
                    os.makedirs(os.path.join(dataset_path, 'keyframes'))
                    
                    db_path = os.path.join(dataset_path, 'dataset.db')
                    DatabaseManager.create_database(db_path)
                    
                    QMessageBox.information(self, "Success", f"Dataset '{name}' created successfully!")
                    self.load_dataset(dataset_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to create dataset: {str(e)}")

    def import_images(self):
        """Import images from another directory into the candidates folder"""
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        import_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Import Images From")
        if import_dir:
            candidates_dir = os.path.join(self.dataset_path, 'candidates')
            imported_count = self.db_manager.import_images(import_dir, candidates_dir)
            QMessageBox.information(self, "Import Complete", f"Imported {imported_count} images to candidates folder.")
            self.load_candidate_images()

    def save_annotations(self, annotations):
        if self.current_image:
            # Convert coordinates to float before saving
            float_annotations = [(ann_type, float(x1), float(y1), float(x2) if x2 is not None else None, float(y2) if y2 is not None else None) for ann_type, x1, y1, x2, y2 in annotations]
            self.db_manager.save_annotations(self.current_image, float_annotations)
            print(f"Saved {len(annotations)} annotations for {self.current_image}")  # Debug print

    def closeEvent(self, event):
        """Clean up resources when the window is closed"""
        if self.db_manager:
            self.db_manager.close()
        event.accept()

    def import_old_db(self):
        # Cette fonction sera implémentée plus tard
        pass

# Example usage
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    explorer = DatasetExplorerGUI()
    explorer.show()
    sys.exit(app.exec_())