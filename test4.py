import sys
import os
import shutil
import tempfile
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QFileDialog, QComboBox,
                               QTreeView, QAbstractItemView, QHeaderView, QLabel, QMessageBox)
from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex
from PySide6.QtGui import QStandardItemModel, QStandardItem

import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOImage import vtkDICOMImageReader
from vtkmodules.vtkInteractionImage import vtkImageViewer2
from vtkmodules.vtkRenderingCore import (vtkActor2D, vtkRenderWindowInteractor,
                                         vtkTextMapper, vtkTextProperty)
from vtkmodules.vtkImagingCore import vtkImageReslice
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from pymongo import MongoClient
from bson import ObjectId
import pydicom
import numpy as np
from dotenv import load_dotenv

from upload import upload_folder


load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")


class CustomInteractorStyle(vtk.vtkInteractorStyleImage):
    def __init__(self, image_viewer, status_actor):
        super().__init__()
        self.AddObserver('MouseWheelForwardEvent', self.move_slice_forward)
        self.AddObserver('MouseWheelBackwardEvent', self.move_slice_backward)
        self.AddObserver('KeyPressEvent', self.key_press_event)
        self.image_viewer = image_viewer
        self.status_actor = status_actor
        self.slice = image_viewer.GetSliceMin()
        self.min_slice = image_viewer.GetSliceMin()
        self.max_slice = image_viewer.GetSliceMax()
        self.update_status_message()

    def update_status_message(self):
        message = f'Slice Number {self.slice + 1}/{self.max_slice + 1}'
        self.status_actor.GetMapper().SetInput(message)

    def move_slice_forward(self, obj, event):
        if self.slice < self.max_slice:
            self.slice += 1
            self.image_viewer.SetSlice(self.slice)
            self.update_status_message()
            self.image_viewer.Render()

    def move_slice_backward(self, obj, event):
        if self.slice > self.min_slice:
            self.slice -= 1
            self.image_viewer.SetSlice(self.slice)
            self.update_status_message()
            self.image_viewer.Render()

    def key_press_event(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == 'Up':
            self.move_slice_forward(obj, event)
        elif key == 'Down':
            self.move_slice_backward(obj, event)


class DICOMViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors = vtkNamedColors()
        self.dicom_folder = ""
        self.view_orientation = "axial"
        self.dicom_files = []

        # Use appdata/local directory instead of temp directory
        self.app_data_dir = os.path.join(
            os.getenv('LOCALAPPDATA') if os.name == 'nt' else os.path.expanduser('~/.local/share'),
            'DICOMViewer',
            'cache'
        )

        # Create directory if it doesn't exist
        os.makedirs(self.app_data_dir, exist_ok=True)
        print(f"created directory{self.app_data_dir }")

        # Setup UI
        self.setup_ui()

        # Initialize VTK components
        self.setup_vtk()

        self.window_width = 400  # Default values (soft tissue window)
        self.window_level = 50

    def clear_cache(self):
        """Clear the cache directory of DICOM files"""
        for f in os.listdir(self.app_data_dir):
            if f.endswith('.dcm'):
                try:
                    os.remove(os.path.join(self.app_data_dir, f))
                except Exception as e:
                    print(f"Warning: Could not remove {f}: {e}")

    def setup_ui(self):
        """Set up the user interface"""
        self.main_layout = QVBoxLayout(self)

        # Create toolbar
        self.toolbar = QWidget()
        self.toolbar_layout = QHBoxLayout(self.toolbar)

        # Add open buttons to toolbar
        self.open_button = QPushButton("Open DICOM Folder")
        self.open_button.clicked.connect(self.open_dicom_folder)

        # Add Upload Button
        self.upload_button = QPushButton("Upload DICOM")
        self.upload_button.clicked.connect(self.upload_dicom_folder)

        self.view_archive_button = QPushButton("View Archive")
        self.view_archive_button.clicked.connect(self.show_archive_viewer)

        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Axial", "Coronal", "Sagittal"])
        self.orientation_combo.currentTextChanged.connect(self.change_orientation)

        self.toolbar_layout.addWidget(self.open_button)
        self.toolbar_layout.addWidget(self.upload_button)
        self.toolbar_layout.addWidget(self.view_archive_button)
        self.toolbar_layout.addWidget(self.orientation_combo)
        self.toolbar_layout.addStretch()

        # Add toolbar to main layout
        self.main_layout.addWidget(self.toolbar)

        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.main_layout.addWidget(self.vtk_widget)

    def setup_vtk(self):
        """Initialize VTK components"""
        self.image_viewer = vtkImageViewer2()

        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Create text actors
        self.slice_text_actor = self.create_text_actor("No DICOM loaded", 15, 10, 20, align_bottom=True)
        self.usage_text_actor = self.create_text_actor(
            "- Slice with mouse wheel or Up/Down-Key\n- Zoom with pressed right mouse button while dragging",
            0.05, 0.95, 14, normalized=True)

        # Add text actors to renderer
        renderer = self.image_viewer.GetRenderer()
        renderer.AddActor2D(self.slice_text_actor)
        renderer.AddActor2D(self.usage_text_actor)
        renderer.SetBackground(self.colors.GetColor3d('Black'))

    def create_text_actor(self, text, x, y, font_size, align_bottom=False, normalized=False):
        """Helper function to create text actors"""
        text_prop = vtkTextProperty()
        text_prop.SetFontFamilyToCourier()
        text_prop.SetFontSize(font_size)
        text_prop.SetVerticalJustificationToBottom() if align_bottom else text_prop.SetVerticalJustificationToTop()
        text_prop.SetJustificationToLeft()

        text_mapper = vtkTextMapper()
        text_mapper.SetInput(text)
        text_mapper.SetTextProperty(text_prop)

        text_actor = vtkActor2D()
        text_actor.SetMapper(text_mapper)
        if normalized:
            text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        text_actor.SetPosition(x, y)

        return text_actor

    def upload_dicom_folder(self):
        """Handle DICOM upload to MongoDB"""
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder to Upload")
        if folder:
            try:
                upload_folder(folder)
                QMessageBox.information(self, "Success", "DICOM files uploaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Upload failed: {str(e)}")

    def open_dicom_folder(self):
        """Open a dialog to select DICOM folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.load_dicom_series(folder)

    def set_ww_wl(self):
        """ this set the correct window level and width"""
        try:
            # Read first DICOM file for metadata
            first_file = self.dicom_files[0]
            ds = pydicom.dcmread(first_file, stop_before_pixels=False)
            window_width_element = ds.get((0x0028, 0x1051), None)
            window_level_element = ds.get((0x0028, 0x1050), None)

            if window_width_element and window_level_element:
                # Convert to string and split multi-values (e.g., "400\2000")
                ww_str = str(window_width_element.value).split('\\')[0]
                wl_str = str(window_level_element.value).split('\\')[0]

                try:
                    self.window_width = float(ww_str)
                    self.window_level = float(wl_str)
                    print(f"Using DICOM metadata WW/WL: WW={self.window_width}, WL={self.window_level}")
                except (ValueError, TypeError):
                    # Fallback to auto-calculation if conversion fails
                    self.calculate_window_from_pixels(ds)
            else:
                self.calculate_window_from_pixels(ds)

        except Exception as e :
            print(f"Error setting window width and level: {str(e)}")

    def calculate_window_from_pixels(self, ds):
        """Calculate WW/WL from pixel data"""
        pixel_data = ds.pixel_array
        min_val = pixel_data.min()
        max_val = pixel_data.max()
        self.window_level = (max_val + min_val) / 2.0
        self.window_width = max_val - min_val
        print(f"Auto-calculated WW/WL: WW={self.window_width}, WL={self.window_level}")

    def load_dicom_series(self, folder_path):
        """Load DICOM series from the specified folder"""
        try:
            self.dicom_folder = folder_path
            self.dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                if f.lower().endswith('.dcm')]

            if not self.dicom_files:
                raise ValueError("No DICOM files found in the selected folder")

            # set window width and level
            self.set_ww_wl()

            # Create reader
            self.reader = vtkDICOMImageReader()
            self.reader.SetDirectoryName(self.dicom_folder)
            self.reader.Update()

            # Verify data
            dimensions = self.reader.GetOutput().GetDimensions()
            if dimensions[0] == 0 or dimensions[1] == 0 or dimensions[2] == 0:
                raise ValueError("No valid DICOM data found")

            print(f"DICOM data dimensions: {dimensions}")
            print(f"Number of slices: {dimensions[2]}")

            # Update viewer
            self.update_viewer()

        except Exception as e:
            print(f"Error loading DICOM series: {str(e)}")
            self.slice_text_actor.GetMapper().SetInput(f"Error: {str(e)}")
            self.image_viewer.Render()

    def load_dicom_archive(self, file_paths):
        """Load DICOM files from a list of paths"""
        # Verify all files exist first
        valid_files = [f for f in file_paths if os.path.exists(f)]

        if not valid_files:
            error_msg = "No valid DICOM files found"
            print(error_msg)
            self.slice_text_actor.GetMapper().SetInput(error_msg)
            self.image_viewer.Render()
            return

        try:

            # self.dicom_files = file_paths
            self.dicom_folder = self.app_data_dir

            # Create reader
            self.reader = vtkDICOMImageReader()
            self.reader.SetDirectoryName(self.dicom_folder)
            self.reader.Update()

            # Verify data
            dimensions = self.reader.GetOutput().GetDimensions()
            if dimensions[0] == 0 or dimensions[1] == 0 or dimensions[2] == 0:
                raise ValueError("No valid DICOM data found")

            print(f"Loaded {len(file_paths)} DICOM files")
            print(f"DICOM data dimensions: {dimensions}")

            self.update_viewer()
            self.slice_text_actor.GetMapper().SetInput(f"Loaded {len(file_paths)} DICOM images")
            self.image_viewer.Render()

        except Exception as e:
            print(f"Error loading DICOM files: {str(e)}")
            self.slice_text_actor.GetMapper().SetInput(f"Error: {str(e)}")
            self.image_viewer.Render()

    def change_orientation(self, orientation):
        """Change the viewing orientation"""
        self.view_orientation = orientation.lower()
        if hasattr(self, 'reader') and self.reader:
            self.update_viewer()

    def update_viewer(self):
        """Update the viewer with current orientation"""
        if not hasattr(self, 'reader') or not self.reader:
            return

        # Clear previous reslice if it exists
        if hasattr(self, 'reslice'):
            del self.reslice

        # Handle orientation
        if self.view_orientation == 'axial':
            self.image_viewer.SetInputConnection(self.reader.GetOutputPort())
            self.image_viewer.SetSliceOrientationToXY()
        else:
            self.reslice = vtkImageReslice()
            self.reslice.SetInputConnection(self.reader.GetOutputPort())
            self.reslice.SetOutputSpacing(1, 1, 1)
            self.reslice.SetInterpolationModeToLinear()

            if self.view_orientation == 'coronal':
                self.reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 0, 1, 0, -1, 0)
                self.image_viewer.SetSliceOrientationToXZ()
            elif self.view_orientation == 'sagittal':
                self.reslice.SetResliceAxesDirectionCosines(0, 1, 0, 0, 0, 1, 1, 0, 0)
                self.image_viewer.SetSliceOrientationToYZ()

            self.image_viewer.SetInputConnection(self.reslice.GetOutputPort())

        # Apply window/level settings to the mapper
        self.image_viewer.SetColorWindow(self.window_width)
        self.image_viewer.SetColorLevel(self.window_level)

        # Configure interactor
        self.image_viewer.SetRenderWindow(self.vtk_widget.GetRenderWindow())
        self.image_viewer.SetupInteractor(self.interactor)

        # Set custom interactor style
        self.interactor_style = CustomInteractorStyle(self.image_viewer, self.slice_text_actor)
        self.interactor.SetInteractorStyle(self.interactor_style)

        # Set initial slice
        self.image_viewer.SetSlice(self.image_viewer.GetSliceMin())
        self.interactor_style.slice = self.image_viewer.GetSliceMin()
        self.interactor_style.min_slice = self.image_viewer.GetSliceMin()
        self.interactor_style.max_slice = self.image_viewer.GetSliceMax()
        self.interactor_style.update_status_message()

        # Render
        self.image_viewer.Render()
        self.image_viewer.GetRenderer().ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def show_archive_viewer(self):
        """Show the DICOM archive viewer dialog"""
        self.archive_viewer = DICOMArchiveViewer(self)
        self.archive_viewer.show()


class DICOMArchiveViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.setWindowTitle("DICOM Study Archive")
        self.setGeometry(100, 100, 800, 600)

        # MongoDB connection
        self.client = MongoClient(MONGODB_URL)
        self.db = self.client['radiology-db']
        self.patients = self.db['patients']
        self.studies = self.db['studies']
        self.series = self.db['series']
        self.instances = self.db['instances']

        self.setup_ui()
        self.refresh_studies()

    def setup_ui(self):
        """Set up the user interface"""
        self.main_layout = QVBoxLayout(self)

        # Create tree view
        self.tree_view = QTreeView()
        self.tree_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.doubleClicked.connect(self.on_study_double_click)

        # Create model
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels([
            "Patient ID", "Patient Name", "Study Type", "Study Date"
        ])
        self.tree_view.setModel(self.model)

        # Configure header
        header = self.tree_view.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        # Add refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_studies)

        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)

        # Add status label
        self.status_label = QLabel("")

        # Add widgets to layout
        self.main_layout.addWidget(self.tree_view)
        self.main_layout.addWidget(self.status_label)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.close_button)
        button_layout.addStretch()

        self.main_layout.addLayout(button_layout)

    def refresh_studies(self):
        """Refresh the list of studies from MongoDB"""
        self.model.removeRows(0, self.model.rowCount())

        try:
            studies = self.studies.find().sort('created_at', -1)

            for study in studies:
                patient = self.patients.find_one({"_id": study["patient_id"]})
                study_date = study.get('study_date', '')
                if study_date:
                    study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"

                row = [
                    QStandardItem(patient.get('patient_id', '')),
                    QStandardItem(patient.get('name', '')),
                    QStandardItem(study.get('study_description', '')),
                    QStandardItem(study_date)
                ]

                # Store the study ID as data in the first column
                row[0].setData(str(study['_id']), Qt.UserRole)

                self.model.appendRow(row)

            self.status_label.setText(f"Loaded {self.model.rowCount()} studies")

        except Exception as e:
            self.status_label.setText(f"Error loading studies: {str(e)}")

    def on_study_double_click(self, index):
        """Handle double-click on a study to load its images"""
        study_id = self.model.itemFromIndex(index.siblingAtColumn(0)).data(Qt.UserRole)

        try:
            study = self.studies.find_one({"_id": ObjectId(study_id)})

            # Get all series for this study, sorted by series number
            series_list = list(self.series.find(
                {"study_uid": study["study_uid"]}
            ).sort("series_number", 1))

            if not series_list:
                self.status_label.setText("No series found for this study")
                return

            # Get first series (lowest series number)
            first_series = series_list[0]

            # Get all instances for this series, sorted by instance number
            instances_list = list(self.instances.find(
                {"series_id": first_series["_id"]}
            ).sort("instance_number", 1))

            if not instances_list:
                self.status_label.setText("No DICOM instances found for this series")
                return

            # Prepare the DICOM files for display
            self.prepare_dicom_viewer(instances_list)

        except Exception as e:
            self.status_label.setText(f"Error loading study: {str(e)}")

    def prepare_dicom_viewer(self, instances):
        temp_files = []
        try:
            for i, instance in enumerate(instances):
                try:
                    filename = os.path.join(self.parent_viewer.app_data_dir, f"instance_{i+1:04d}.dcm")
                    print(f"Writing to: {filename}")

                    # Verify DICOM data before writing
                    if not instance.get('dicom_file'):
                        print(f"Skipping empty DICOM data at index {i}")
                        continue

                    print(f"Writing {len(instance['dicom_file'])} bytes to {filename}")

                    # Write and immediately flush/close
                    with open(filename, 'wb') as f:
                        f.write(instance['dicom_file'])
                        f.flush()
                        os.fsync(f.fileno())

                    # Verify file exists and has content
                    if os.path.exists(filename):
                        size = os.path.getsize(filename)
                        print(f"Successfully wrote {size} bytes to {filename}")
                        temp_files.append(filename)
                    else:
                        print(f"ERROR: File not found after writing!")

                except Exception as e:
                    print(f"Error creating file {i}: {str(e)}")
                    continue

            # Additional verification
            print(f"\nDirectory listing after writes:")
            for f in os.listdir(self.parent_viewer.app_data_dir):
                print(f"Found: {f}")

                if not temp_files:
                    raise ValueError("No valid DICOM files were created")

                # Verify at least one DICOM file is valid
                try:
                    test_file = temp_files[0]
                    ds = pydicom.dcmread(test_file)
                    print(f"First file validation: Modality={ds.Modality}, SOPClassUID={ds.SOPClassUID}")
                except Exception as e:
                    print(f"DICOM validation failed: {str(e)}")
                    shutil.rmtree(temp_dir)
                    raise ValueError("Invalid DICOM data in first file")

                # Sort files by instance number
                temp_files.sort(key=lambda x: self.get_instance_number(x))
                print(temp_files)

                # Load the DICOM files into the viewer
                self.parent_viewer.load_dicom_archive(temp_files)


            if temp_files:
                self.parent_viewer.set_ww_wl()

        except Exception as e:
            print(f"Error preparing DICOM viewer: {str(e)}")

    def get_instance_number(self, filepath):
        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            return int(getattr(ds, 'InstanceNumber', 0))
        except Exception as e:
            print(f"Error reading instance number from {filepath}: {e}")
            return 0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Viewer")
        self.setGeometry(100, 100, 800, 800)

        # Create central widget
        self.dicom_viewer = DICOMViewerWidget()
        self.setCentralWidget(self.dicom_viewer)



if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec())