"""
h5_explorer.py: Interactive HDF5 File Hierarchy Explorer & Dataset Selector for arVIPAspectro GUI.
"""

import os
from typing import Optional, Dict, Any, List
import numpy as np
import h5py
from PySide6 import QtWidgets, QtCore, QtGui

try:
    from .. import backend as bck
    from .theme import get_system_theme
except ImportError:
    import backend as bck
    from theme import get_system_theme


class H5ExplorerWidget(QtWidgets.QWidget):
    """
    HDF5 Tree Explorer and Dataset Selector.
    Allows browsing groups and datasets, viewing shapes/attributes,
    and selecting datasets for visualization or pipeline assignment.
    """

    # Signals
    sig_dataset_preview = QtCore.Signal(str, object, dict)  # (path, numpy_array, attrs_dict)
    sig_sample_selected = QtCore.Signal(str)               # sample group path (e.g. '01_glycerol_25')
    sig_laser_selected = QtCore.Signal(str)                # laser ref path
    sig_rb_selected = QtCore.Signal(str)                   # Rb ref path
    sig_file_loaded = QtCore.Signal(str)                   # path to loaded h5 file

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_h5_path: Optional[str] = None
        self.theme_name: str = get_system_theme()
        self._setup_ui()

    def set_theme(self, theme_name: str):
        """Updates the tree item colors and details label for dark/light mode."""
        self.theme_name = theme_name
        self._update_details_styling()
        self.reload_tree()

    def _update_details_styling(self):
        if self.theme_name == "dark":
            self.lbl_details.setStyleSheet(
                "font-family: monospace; font-size: 11px; color: #a5a5a5; background: #161616; padding: 4px; border: 1px solid #333333;"
            )
        else:
            self.lbl_details.setStyleSheet(
                "font-family: monospace; font-size: 11px; color: #374151; background: #f8fafc; padding: 4px; border: 1px solid #cbd5e1;"
            )

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Top File Selection Toolbar
        file_box = QtWidgets.QGroupBox("HDF5 Container")
        file_layout = QtWidgets.QVBoxLayout(file_box)
        file_layout.setContentsMargins(6, 8, 6, 6)
        file_layout.setSpacing(4)

        h_path = QtWidgets.QHBoxLayout()
        self.txt_file_path = QtWidgets.QLineEdit()
        self.txt_file_path.setPlaceholderText("Select or enter path to Measures.h5...")
        self.txt_file_path.returnPressed.connect(self._on_manual_path_entered)

        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_h5_file)

        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.clicked.connect(self.reload_tree)

        h_path.addWidget(self.txt_file_path)
        h_path.addWidget(btn_browse)
        h_path.addWidget(btn_refresh)
        file_layout.addLayout(h_path)

        # Helper buttons: convert .dat directory
        h_tools = QtWidgets.QHBoxLayout()
        btn_create_dat = QtWidgets.QPushButton("Convert Folder of .dat to H5...")
        btn_create_dat.setStyleSheet("font-size: 11px;")
        btn_create_dat.clicked.connect(self._on_convert_dat_folder)
        h_tools.addWidget(btn_create_dat)
        h_tools.addStretch()
        file_layout.addLayout(h_tools)

        layout.addWidget(file_box)

        # 2. HDF5 Tree Widget
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Name / Path", "Shape", "Type"])
        self.tree.setColumnWidth(0, 240)
        self.tree.setColumnWidth(1, 90)
        self.tree.setColumnWidth(2, 60)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemSelectionChanged.connect(self._on_item_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        layout.addWidget(self.tree, stretch=1)

        # 3. Attributes / Details Panel
        self.lbl_details = QtWidgets.QLabel("Select a group or dataset to inspect attributes.")
        self.lbl_details.setWordWrap(True)
        self.lbl_details.setMaximumHeight(80)
        self._update_details_styling()
        layout.addWidget(self.lbl_details)

        # 4. Action Assignment Buttons
        btn_bar = QtWidgets.QHBoxLayout()
        self.btn_preview = QtWidgets.QPushButton("👁 Preview Data")
        self.btn_preview.setStyleSheet("background-color: #0288d1; color: white; font-weight: bold; padding: 5px 10px;")
        self.btn_preview.clicked.connect(self._preview_selected_item)

        self.btn_set_sample = QtWidgets.QPushButton("Set as Active Sample")
        self.btn_set_sample.clicked.connect(self._set_as_active_sample)

        self.btn_set_laser = QtWidgets.QPushButton("Set as Laser Ref")
        self.btn_set_laser.clicked.connect(self._set_as_laser_ref)

        self.btn_set_rb = QtWidgets.QPushButton("Set as Rb Ref")
        self.btn_set_rb.clicked.connect(self._set_as_rb_ref)

        btn_bar.addWidget(self.btn_preview)
        btn_bar.addWidget(self.btn_set_sample)
        btn_bar.addWidget(self.btn_set_laser)
        btn_bar.addWidget(self.btn_set_rb)
        layout.addLayout(btn_bar)

    # --------------------------------------------------------------------------
    # File Operations
    # --------------------------------------------------------------------------

    def set_h5_file(self, path: str):
        """Sets the active HDF5 path and loads its hierarchy into the tree."""
        if not path:
            return
        self.current_h5_path = os.path.abspath(path)
        self.txt_file_path.setText(self.current_h5_path)
        self.reload_tree()
        self.sig_file_loaded.emit(self.current_h5_path)

    def browse_h5_file(self):
        """Opens a file dialog to pick an HDF5 file."""
        init_dir = os.path.dirname(self.current_h5_path) if self.current_h5_path else os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select HDF5 Measures File", init_dir, "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if path:
            self.set_h5_file(path)

    def _on_manual_path_entered(self):
        path = self.txt_file_path.text().strip()
        if os.path.exists(path):
            self.set_h5_file(path)
        else:
            QtWidgets.QMessageBox.warning(self, "File Not Found", f"Specified HDF5 file does not exist:\n{path}")

    def _on_convert_dat_folder(self):
        """Selects a directory with .dat files and wraps them into an HDF5 file."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory with .dat files to Convert into Measures.h5", os.getcwd()
        )
        if not folder:
            return
        try:
            wrapper = bck.create_wrapper(folder, name="Measures.h5")
            h5_target = os.path.join(folder, "Measures.h5")
            self.set_h5_file(h5_target)
            QtWidgets.QMessageBox.information(
                self, "HDF5 Creation Complete",
                f"Successfully indexed and converted datasets into:\n{h5_target}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Creating HDF5", f"Failed to create HDF5 wrapper:\n{e}")

    # --------------------------------------------------------------------------
    # Tree Population & Traversals
    # --------------------------------------------------------------------------

    def reload_tree(self):
        """Reloads the HDF5 structure into the tree widget."""
        self.tree.clear()
        if not self.current_h5_path or not os.path.exists(self.current_h5_path):
            self.lbl_details.setText("HDF5 file does not exist or has not been loaded.")
            return

        try:
            with h5py.File(self.current_h5_path, "r") as f:
                root_item = QtWidgets.QTreeWidgetItem(self.tree, ["/", "Root", "File"])
                root_item.setData(0, QtCore.Qt.UserRole, "/")
                root_item.setData(1, QtCore.Qt.UserRole, "group")

                self._populate_group(f, root_item)
                root_item.setExpanded(True)

                # Expand 'Brillouin' group by default if present
                for i in range(root_item.childCount()):
                    child = root_item.child(i)
                    if child.text(0) == "Brillouin":
                        child.setExpanded(True)
                        break

            self.lbl_details.setText(f"Loaded: {os.path.basename(self.current_h5_path)}")
        except Exception as e:
            self.lbl_details.setText(f"Error reading H5 file: {e}")

    def _populate_group(self, h5_group: h5py.Group, parent_item: QtWidgets.QTreeWidgetItem):
        """Recursively populates groups and datasets into tree items."""
        grp_color = QtGui.QColor("#ffb74d" if self.theme_name == "dark" else "#b45309")
        ds_color = QtGui.QColor("#4fc3f7" if self.theme_name == "dark" else "#0284c7")

        for name, item in h5_group.items():
            full_path = item.name
            if isinstance(item, h5py.Group):
                n_children = len(item)
                tree_item = QtWidgets.QTreeWidgetItem(
                    parent_item, [name, f"{n_children} items", "Group"]
                )
                tree_item.setData(0, QtCore.Qt.UserRole, full_path)
                tree_item.setData(1, QtCore.Qt.UserRole, "group")
                tree_item.setForeground(0, grp_color)
                self._populate_group(item, tree_item)
            elif isinstance(item, h5py.Dataset):
                shape_str = "x".join(str(s) for s in item.shape) if item.shape else "scalar"
                dtype_str = str(item.dtype)
                tree_item = QtWidgets.QTreeWidgetItem(
                    parent_item, [name, shape_str, dtype_str]
                )
                tree_item.setData(0, QtCore.Qt.UserRole, full_path)
                tree_item.setData(1, QtCore.Qt.UserRole, "dataset")
                tree_item.setForeground(0, ds_color)

    # --------------------------------------------------------------------------
    # Selection & Preview
    # --------------------------------------------------------------------------

    def _on_item_selection_changed(self):
        selected = self.tree.selectedItems()
        if not selected:
            self.lbl_details.setText("No item selected.")
            return

        item = selected[0]
        full_path = item.data(0, QtCore.Qt.UserRole)
        kind = item.data(1, QtCore.Qt.UserRole)

        if not self.current_h5_path or not os.path.exists(self.current_h5_path):
            return

        try:
            with h5py.File(self.current_h5_path, "r") as f:
                if full_path in f:
                    node = f[full_path]
                    attrs_txt = []
                    for k, v in node.attrs.items():
                        val_str = f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
                        attrs_txt.append(f"{k}: {val_str}")
                    attrs_summary = " | ".join(attrs_txt) if attrs_txt else "None"
                    self.lbl_details.setText(f"Path: {full_path}\nKind: {kind.capitalize()} | Attributes: {attrs_summary}")
        except Exception as e:
            self.lbl_details.setText(f"Path: {full_path} (Error reading attributes: {e})")

    def _on_item_double_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        self._preview_selected_item()

    def _preview_selected_item(self):
        """Reads data from the selected H5 dataset and emits preview signal."""
        selected = self.tree.selectedItems()
        if not selected:
            return

        item = selected[0]
        full_path = item.data(0, QtCore.Qt.UserRole)
        kind = item.data(1, QtCore.Qt.UserRole)

        if not self.current_h5_path or not os.path.exists(self.current_h5_path):
            return

        try:
            with h5py.File(self.current_h5_path, "r") as f:
                if kind == "dataset":
                    ds = f[full_path]
                    data = ds[()]
                    attrs = dict(ds.attrs)
                    self.sig_dataset_preview.emit(full_path, data, attrs)
                elif kind == "group":
                    # If user clicks a group that has an 'image' or raw dataset inside, preview that!
                    grp = f[full_path]
                    attrs = dict(grp.attrs)
                    preview_data = None
                    preview_path = full_path
                    for candidate in ["image", "image_corrected", "shift", "linewidth", "q_vector"]:
                        if candidate in grp and isinstance(grp[candidate], h5py.Dataset):
                            preview_data = grp[candidate][()]
                            preview_path = f"{full_path}/{candidate}"
                            break

                    if preview_data is not None:
                        self.sig_dataset_preview.emit(preview_path, preview_data, attrs)
                    else:
                        QtWidgets.QMessageBox.information(
                            self, "Group Selected",
                            f"Group: {full_path}\nContains {len(grp)} items:\n{list(grp.keys())}"
                        )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Loading Dataset", f"Failed to load dataset '{full_path}':\n{e}")

    # --------------------------------------------------------------------------
    # Context Menu & Role Assignment
    # --------------------------------------------------------------------------

    def _on_tree_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item:
            return
        menu = QtWidgets.QMenu(self)
        act_preview = menu.addAction("👁 Preview on Visualizer")
        act_sample = menu.addAction("🎯 Set as Active Sample")
        act_laser = menu.addAction("⚡ Set as Laser Reference")
        act_rb = menu.addAction("🔬 Set as Rubidium Reference")

        action = menu.exec_(self.tree.mapToGlobal(pos))
        if action == act_preview:
            self._preview_selected_item()
        elif action == act_sample:
            self._set_as_active_sample()
        elif action == act_laser:
            self._set_as_laser_ref()
        elif action == act_rb:
            self._set_as_rb_ref()

    def _get_selected_clean_path(self) -> Optional[str]:
        selected = self.tree.selectedItems()
        if not selected:
            return None
        full_path = selected[0].data(0, QtCore.Qt.UserRole)
        # Strip leading '/' if present
        return full_path.lstrip("/")

    def _set_as_active_sample(self):
        path = self._get_selected_clean_path()
        if not path:
            return
        # If user picked a dataset inside a sample group (e.g. 'Brillouin/water_5s/image'), get the sample group
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] == "Brillouin":
            sample_name = parts[1]
        else:
            sample_name = parts[0]
        self.sig_sample_selected.emit(sample_name)

    def _set_as_laser_ref(self):
        path = self._get_selected_clean_path()
        if path:
            self.sig_laser_selected.emit(path)

    def _set_as_rb_ref(self):
        path = self._get_selected_clean_path()
        if path:
            self.sig_rb_selected.emit(path)

    def list_available_samples(self) -> List[str]:
        """Scans the HDF5 file and returns a list of sample group names under Brillouin/."""
        if not self.current_h5_path or not os.path.exists(self.current_h5_path):
            return []
        samples = []
        try:
            with h5py.File(self.current_h5_path, "r") as f:
                if "Brillouin" in f:
                    for k in f["Brillouin"].keys():
                        if k not in ["laser_reference", "Rb_absorption"]:
                            samples.append(k)
        except Exception:
            pass
        return sorted(samples)
