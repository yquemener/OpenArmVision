import os
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QPushButton, QFileDialog, QInputDialog, QMessageBox
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
5. Display dataset info at the top of the window.
6. Provide separate tabs or sections for candidate and keyframe images.
7. Allow users to open a dataset folder through a file dialog.
8. Remember and automatically open the last used dataset.
9. Allow creating new datasets and importing images.

The main class, DatasetExplorerGUI, inherits from QWidget and encapsulates all GUI-related functionality.
"""

class DatasetExplorerGUI(QWidget):
    dataset_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.db_manager = None
        self.dataset_info = None
        self.settings = QSettings("YourCompany", "DatasetExplorer")
        
        self.init_ui()
        self.load_last_dataset()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Dataset Explorer')
        self.setGeometry(100, 100, 1200, 800)  # Increased window size

        main_layout = QVBoxLayout()

        # Dataset info section
        self.info_label = QLabel('No dataset loaded')
        main_layout.addWidget(self.info_label)

        # Main content area
        content_layout = QHBoxLayout()

        # Image list
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        content_layout.addWidget(self.image_list, 1)

        # Annotation widget
        self.annotation_widget = AnnotationWidget()
        self.annotation_widget.annotation_changed.connect(self.save_annotations)
        content_layout.addWidget(self.annotation_widget, 1)  # Use stretch factor to give it more space

        main_layout.addLayout(content_layout)

        # Controls
        controls_layout = QHBoxLayout()
        load_button = QPushButton('Load Dataset')
        load_button.clicked.connect(self.load_dataset)
        controls_layout.addWidget(load_button)
        
        self.toggle_button = QPushButton('Show Candidates')
        self.toggle_button.clicked.connect(self.toggle_image_list)
        controls_layout.addWidget(self.toggle_button)

        create_button = QPushButton('Create New Dataset')
        create_button.clicked.connect(self.create_new_dataset)
        controls_layout.addWidget(create_button)

        import_button = QPushButton('Import Images')
        import_button.clicked.connect(self.import_images)
        controls_layout.addWidget(import_button)

        save_button = QPushButton('Save Annotations')
        save_button.clicked.connect(self.save_annotations)
        controls_layout.addWidget(save_button)

        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)

        # Connect the dataset_changed signal to update the UI
        self.dataset_changed.connect(self.update_ui)

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
            self.point_file_path = os.path.join(self.dataset_path, 'point.json')
            
            self._load_dataset_info()
            self.dataset_changed.emit(self.dataset_path)
            self.settings.setValue("last_dataset", self.dataset_path)

    def _load_dataset_info(self):
        """Load dataset information from the point.json file"""
        with open(self.point_file_path, 'r') as f:
            self.dataset_info = json.load(f)

    def update_ui(self):
        """Update the UI with dataset information"""
        self._update_info_label()
        self.load_candidate_images()

    def _update_info_label(self):
        """Update the info label with dataset information"""
        if self.dataset_info:
            info_text = f"Dataset: {os.path.basename(self.dataset_path)}\n"
            info_text += f"Total images: {self.dataset_info.get('total_images', 'N/A')}\n"
            info_text += f"Classes: {', '.join(self.dataset_info.get('classes', []))}"
            self.info_label.setText(info_text)

    def load_candidate_images(self):
        """Load and display the list of candidate images"""
        self.image_list.clear()
        if self.db_manager:
            candidates = self.db_manager.get_candidate_images()
            print(f"Found {len(candidates)} candidate images")  # Débogage
            for img in candidates:
                print(f"Adding image: {img.path}")  # Débogage
                self.image_list.addItem(img.path)
        else:
            print("Database manager is not initialized")  # Débogage
        self.toggle_button.setText('Show Keyframes')

    def load_keyframe_images(self):
        """Load and display the list of keyframe images"""
        self.image_list.clear()
        keyframes = self.db_manager.get_keyframe_images()
        for img in keyframes:
            self.image_list.addItem(img.path)
        self.toggle_button.setText('Show Candidates')

    def toggle_image_list(self):
        """Toggle between displaying candidate and keyframe images"""
        if self.toggle_button.text() == 'Show Keyframes':
            self.load_keyframe_images()
        else:
            self.load_candidate_images()

    def on_image_selected(self, item):
        image_path = item.text()
        self.current_image = image_path
        full_path = os.path.join(self.dataset_path, image_path)
        self.annotation_widget.load_image(full_path)
        
        # Charger les annotations existantes
        annotations = self.db_manager.get_image_annotations(image_path)
        self.annotation_widget.load_annotations(annotations)

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
                    
                    point_data = {
                        "name": name,
                        "total_images": 0,
                        "classes": []
                    }
                    with open(os.path.join(dataset_path, 'point.json'), 'w') as f:
                        json.dump(point_data, f)
                    
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

# Example usage
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    explorer = DatasetExplorerGUI()
    explorer.show()
    sys.exit(app.exec_())