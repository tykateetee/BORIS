# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'boris/project.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dlgProject(object):
    def setupUi(self, dlgProject):
        dlgProject.setObjectName("dlgProject")
        dlgProject.resize(1202, 697)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(dlgProject)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.tabProject = QtWidgets.QTabWidget(dlgProject)
        self.tabProject.setObjectName("tabProject")
        self.tabInformation = QtWidgets.QWidget()
        self.tabInformation.setObjectName("tabInformation")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tabInformation)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label = QtWidgets.QLabel(self.tabInformation)
        self.label.setObjectName("label")
        self.horizontalLayout_15.addWidget(self.label)
        self.leProjectName = QtWidgets.QLineEdit(self.tabInformation)
        self.leProjectName.setObjectName("leProjectName")
        self.horizontalLayout_15.addWidget(self.leProjectName)
        self.verticalLayout.addLayout(self.horizontalLayout_15)
        self.lbProjectFilePath = QtWidgets.QLabel(self.tabInformation)
        self.lbProjectFilePath.setObjectName("lbProjectFilePath")
        self.verticalLayout.addWidget(self.lbProjectFilePath)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.label_7 = QtWidgets.QLabel(self.tabInformation)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_18.addWidget(self.label_7)
        self.dteDate = QtWidgets.QDateTimeEdit(self.tabInformation)
        self.dteDate.setCalendarPopup(True)
        self.dteDate.setObjectName("dteDate")
        self.horizontalLayout_18.addWidget(self.dteDate)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_18)
        self.label_6 = QtWidgets.QLabel(self.tabInformation)
        self.label_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.teDescription = QtWidgets.QPlainTextEdit(self.tabInformation)
        self.teDescription.setObjectName("teDescription")
        self.verticalLayout.addWidget(self.teDescription)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.lbTimeFormat = QtWidgets.QLabel(self.tabInformation)
        self.lbTimeFormat.setObjectName("lbTimeFormat")
        self.horizontalLayout_19.addWidget(self.lbTimeFormat)
        self.rbSeconds = QtWidgets.QRadioButton(self.tabInformation)
        self.rbSeconds.setChecked(True)
        self.rbSeconds.setObjectName("rbSeconds")
        self.horizontalLayout_19.addWidget(self.rbSeconds)
        self.rbHMS = QtWidgets.QRadioButton(self.tabInformation)
        self.rbHMS.setObjectName("rbHMS")
        self.horizontalLayout_19.addWidget(self.rbHMS)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_19.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_19)
        self.lb_project_format_version = QtWidgets.QLabel(self.tabInformation)
        self.lb_project_format_version.setObjectName("lb_project_format_version")
        self.verticalLayout.addWidget(self.lb_project_format_version)
        self.tabProject.addTab(self.tabInformation, "")
        self.tabEthogram = QtWidgets.QWidget()
        self.tabEthogram.setObjectName("tabEthogram")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.tabEthogram)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.twBehaviors = QtWidgets.QTableWidget(self.tabEthogram)
        self.twBehaviors.setAutoFillBackground(False)
        self.twBehaviors.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.twBehaviors.setMidLineWidth(0)
        self.twBehaviors.setAlternatingRowColors(True)
        self.twBehaviors.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.twBehaviors.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.twBehaviors.setObjectName("twBehaviors")
        self.twBehaviors.setColumnCount(9)
        self.twBehaviors.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.twBehaviors.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.twBehaviors.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.twBehaviors.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.twBehaviors.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.twBehaviors.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.twBehaviors.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.twBehaviors.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.twBehaviors.setHorizontalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.twBehaviors.setHorizontalHeaderItem(8, item)
        self.twBehaviors.horizontalHeader().setSortIndicatorShown(False)
        self.twBehaviors.verticalHeader().setSortIndicatorShown(False)
        self.horizontalLayout_11.addWidget(self.twBehaviors)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.pb_behavior = QtWidgets.QPushButton(self.tabEthogram)
        self.pb_behavior.setObjectName("pb_behavior")
        self.verticalLayout_11.addWidget(self.pb_behavior)
        self.pb_import = QtWidgets.QPushButton(self.tabEthogram)
        self.pb_import.setObjectName("pb_import")
        self.verticalLayout_11.addWidget(self.pb_import)
        self.pbBehaviorsCategories = QtWidgets.QPushButton(self.tabEthogram)
        self.pbBehaviorsCategories.setObjectName("pbBehaviorsCategories")
        self.verticalLayout_11.addWidget(self.pbBehaviorsCategories)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_11.addItem(spacerItem2)
        self.pb_exclusion_matrix = QtWidgets.QPushButton(self.tabEthogram)
        self.pb_exclusion_matrix.setObjectName("pb_exclusion_matrix")
        self.verticalLayout_11.addWidget(self.pb_exclusion_matrix)
        self.pbExportEthogram = QtWidgets.QPushButton(self.tabEthogram)
        self.pbExportEthogram.setObjectName("pbExportEthogram")
        self.verticalLayout_11.addWidget(self.pbExportEthogram)
        self.horizontalLayout_11.addLayout(self.verticalLayout_11)
        self.verticalLayout_5.addLayout(self.horizontalLayout_11)
        self.lbObservationsState = QtWidgets.QLabel(self.tabEthogram)
        self.lbObservationsState.setObjectName("lbObservationsState")
        self.verticalLayout_5.addWidget(self.lbObservationsState)
        self.verticalLayout_10.addLayout(self.verticalLayout_5)
        self.tabProject.addTab(self.tabEthogram, "")
        self.tabSubjects = QtWidgets.QWidget()
        self.tabSubjects.setObjectName("tabSubjects")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.tabSubjects)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.twSubjects = QtWidgets.QTableWidget(self.tabSubjects)
        self.twSubjects.setAutoFillBackground(False)
        self.twSubjects.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.twSubjects.setMidLineWidth(0)
        self.twSubjects.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.twSubjects.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.twSubjects.setObjectName("twSubjects")
        self.twSubjects.setColumnCount(3)
        self.twSubjects.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.twSubjects.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.twSubjects.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.twSubjects.setHorizontalHeaderItem(2, item)
        self.horizontalLayout_12.addWidget(self.twSubjects)
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.pb_subjects = QtWidgets.QPushButton(self.tabSubjects)
        self.pb_subjects.setObjectName("pb_subjects")
        self.verticalLayout_15.addWidget(self.pb_subjects)
        self.pbImportSubjectsFromProject = QtWidgets.QPushButton(self.tabSubjects)
        self.pbImportSubjectsFromProject.setObjectName("pbImportSubjectsFromProject")
        self.verticalLayout_15.addWidget(self.pbImportSubjectsFromProject)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_15.addItem(spacerItem3)
        self.pb_export_subjects = QtWidgets.QPushButton(self.tabSubjects)
        self.pb_export_subjects.setObjectName("pb_export_subjects")
        self.verticalLayout_15.addWidget(self.pb_export_subjects)
        self.horizontalLayout_12.addLayout(self.verticalLayout_15)
        self.verticalLayout_14.addLayout(self.horizontalLayout_12)
        self.lbSubjectsState = QtWidgets.QLabel(self.tabSubjects)
        self.lbSubjectsState.setObjectName("lbSubjectsState")
        self.verticalLayout_14.addWidget(self.lbSubjectsState)
        self.verticalLayout_16.addLayout(self.verticalLayout_14)
        self.tabProject.addTab(self.tabSubjects, "")
        self.tabIndependentVariables = QtWidgets.QWidget()
        self.tabIndependentVariables.setObjectName("tabIndependentVariables")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.tabIndependentVariables)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.twVariables = QtWidgets.QTableWidget(self.tabIndependentVariables)
        self.twVariables.setAutoFillBackground(False)
        self.twVariables.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.twVariables.setMidLineWidth(0)
        self.twVariables.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.twVariables.setDragDropOverwriteMode(False)
        self.twVariables.setAlternatingRowColors(True)
        self.twVariables.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.twVariables.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.twVariables.setObjectName("twVariables")
        self.twVariables.setColumnCount(5)
        self.twVariables.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.twVariables.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.twVariables.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.twVariables.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.twVariables.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.twVariables.setHorizontalHeaderItem(4, item)
        self.twVariables.horizontalHeader().setSortIndicatorShown(False)
        self.verticalLayout_2.addWidget(self.twVariables)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.tabIndependentVariables)
        self.label_2.setMinimumSize(QtCore.QSize(120, 0))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.leLabel = QtWidgets.QLineEdit(self.tabIndependentVariables)
        self.leLabel.setObjectName("leLabel")
        self.horizontalLayout_3.addWidget(self.leLabel)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.tabIndependentVariables)
        self.label_3.setMinimumSize(QtCore.QSize(120, 0))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.leDescription = QtWidgets.QLineEdit(self.tabIndependentVariables)
        self.leDescription.setObjectName("leDescription")
        self.horizontalLayout_5.addWidget(self.leDescription)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem5)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_8 = QtWidgets.QLabel(self.tabIndependentVariables)
        self.label_8.setMinimumSize(QtCore.QSize(120, 0))
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_6.addWidget(self.label_8)
        self.cbType = QtWidgets.QComboBox(self.tabIndependentVariables)
        self.cbType.setMinimumSize(QtCore.QSize(120, 0))
        self.cbType.setObjectName("cbType")
        self.horizontalLayout_6.addWidget(self.cbType)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem6)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_4 = QtWidgets.QLabel(self.tabIndependentVariables)
        self.label_4.setMinimumSize(QtCore.QSize(120, 0))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_7.addWidget(self.label_4)
        self.lePredefined = QtWidgets.QLineEdit(self.tabIndependentVariables)
        self.lePredefined.setObjectName("lePredefined")
        self.horizontalLayout_7.addWidget(self.lePredefined)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem7)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_9 = QtWidgets.QLabel(self.tabIndependentVariables)
        self.label_9.setMinimumSize(QtCore.QSize(120, 0))
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_8.addWidget(self.label_9)
        self.dte_default_date = QtWidgets.QDateTimeEdit(self.tabIndependentVariables)
        self.dte_default_date.setObjectName("dte_default_date")
        self.horizontalLayout_8.addWidget(self.dte_default_date)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem8)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_5 = QtWidgets.QLabel(self.tabIndependentVariables)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_9.addWidget(self.label_5)
        self.leSetValues = QtWidgets.QLineEdit(self.tabIndependentVariables)
        self.leSetValues.setObjectName("leSetValues")
        self.horizontalLayout_9.addWidget(self.leSetValues)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem9)
        self.verticalLayout_2.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_13.addLayout(self.verticalLayout_2)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.pbAddVariable = QtWidgets.QPushButton(self.tabIndependentVariables)
        self.pbAddVariable.setObjectName("pbAddVariable")
        self.verticalLayout_12.addWidget(self.pbAddVariable)
        self.pbRemoveVariable = QtWidgets.QPushButton(self.tabIndependentVariables)
        self.pbRemoveVariable.setObjectName("pbRemoveVariable")
        self.verticalLayout_12.addWidget(self.pbRemoveVariable)
        self.pbImportVarFromProject = QtWidgets.QPushButton(self.tabIndependentVariables)
        self.pbImportVarFromProject.setObjectName("pbImportVarFromProject")
        self.verticalLayout_12.addWidget(self.pbImportVarFromProject)
        spacerItem10 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_12.addItem(spacerItem10)
        self.horizontalLayout_13.addLayout(self.verticalLayout_12)
        self.horizontalLayout_14.addLayout(self.horizontalLayout_13)
        self.tabProject.addTab(self.tabIndependentVariables, "")
        self.tabBehavCodingMap = QtWidgets.QWidget()
        self.tabBehavCodingMap.setObjectName("tabBehavCodingMap")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.tabBehavCodingMap)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.twBehavCodingMap = QtWidgets.QTableWidget(self.tabBehavCodingMap)
        self.twBehavCodingMap.setAutoFillBackground(False)
        self.twBehavCodingMap.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.twBehavCodingMap.setMidLineWidth(0)
        self.twBehavCodingMap.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.twBehavCodingMap.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.twBehavCodingMap.setObjectName("twBehavCodingMap")
        self.twBehavCodingMap.setColumnCount(2)
        self.twBehavCodingMap.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.twBehavCodingMap.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.twBehavCodingMap.setHorizontalHeaderItem(1, item)
        self.horizontalLayout.addWidget(self.twBehavCodingMap)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pbAddBehaviorsCodingMap = QtWidgets.QPushButton(self.tabBehavCodingMap)
        self.pbAddBehaviorsCodingMap.setObjectName("pbAddBehaviorsCodingMap")
        self.verticalLayout_4.addWidget(self.pbAddBehaviorsCodingMap)
        self.pbRemoveBehaviorsCodingMap = QtWidgets.QPushButton(self.tabBehavCodingMap)
        self.pbRemoveBehaviorsCodingMap.setObjectName("pbRemoveBehaviorsCodingMap")
        self.verticalLayout_4.addWidget(self.pbRemoveBehaviorsCodingMap)
        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem11)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.verticalLayout_8.addLayout(self.horizontalLayout)
        self.tabProject.addTab(self.tabBehavCodingMap, "")
        self.tab_time_converters = QtWidgets.QWidget()
        self.tab_time_converters.setObjectName("tab_time_converters")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout(self.tab_time_converters)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.label_11 = QtWidgets.QLabel(self.tab_time_converters)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_18.addWidget(self.label_11)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.tw_converters = QtWidgets.QTableWidget(self.tab_time_converters)
        self.tw_converters.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tw_converters.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tw_converters.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tw_converters.setObjectName("tw_converters")
        self.tw_converters.setColumnCount(3)
        self.tw_converters.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tw_converters.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_converters.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_converters.setHorizontalHeaderItem(2, item)
        self.horizontalLayout_16.addWidget(self.tw_converters)
        self.verticalLayout_17 = QtWidgets.QVBoxLayout()
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.pb_add_converter = QtWidgets.QPushButton(self.tab_time_converters)
        self.pb_add_converter.setObjectName("pb_add_converter")
        self.verticalLayout_17.addWidget(self.pb_add_converter)
        self.pb_modify_converter = QtWidgets.QPushButton(self.tab_time_converters)
        self.pb_modify_converter.setObjectName("pb_modify_converter")
        self.verticalLayout_17.addWidget(self.pb_modify_converter)
        self.pb_delete_converter = QtWidgets.QPushButton(self.tab_time_converters)
        self.pb_delete_converter.setObjectName("pb_delete_converter")
        self.verticalLayout_17.addWidget(self.pb_delete_converter)
        self.pb_load_from_file = QtWidgets.QPushButton(self.tab_time_converters)
        self.pb_load_from_file.setObjectName("pb_load_from_file")
        self.verticalLayout_17.addWidget(self.pb_load_from_file)
        self.pb_load_from_repo = QtWidgets.QPushButton(self.tab_time_converters)
        self.pb_load_from_repo.setObjectName("pb_load_from_repo")
        self.verticalLayout_17.addWidget(self.pb_load_from_repo)
        spacerItem12 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_17.addItem(spacerItem12)
        self.horizontalLayout_16.addLayout(self.verticalLayout_17)
        self.verticalLayout_18.addLayout(self.horizontalLayout_16)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.label_13 = QtWidgets.QLabel(self.tab_time_converters)
        self.label_13.setMinimumSize(QtCore.QSize(120, 0))
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_17.addWidget(self.label_13)
        self.le_converter_name = QtWidgets.QLineEdit(self.tab_time_converters)
        self.le_converter_name.setObjectName("le_converter_name")
        self.horizontalLayout_17.addWidget(self.le_converter_name)
        spacerItem13 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_17.addItem(spacerItem13)
        self.verticalLayout_18.addLayout(self.horizontalLayout_17)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_10 = QtWidgets.QLabel(self.tab_time_converters)
        self.label_10.setMinimumSize(QtCore.QSize(120, 0))
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_10.addWidget(self.label_10)
        self.le_converter_description = QtWidgets.QLineEdit(self.tab_time_converters)
        self.le_converter_description.setObjectName("le_converter_description")
        self.horizontalLayout_10.addWidget(self.le_converter_description)
        spacerItem14 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem14)
        self.verticalLayout_18.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_12 = QtWidgets.QLabel(self.tab_time_converters)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_9.addWidget(self.label_12)
        self.pb_code_help = QtWidgets.QPushButton(self.tab_time_converters)
        self.pb_code_help.setObjectName("pb_code_help")
        self.verticalLayout_9.addWidget(self.pb_code_help)
        spacerItem15 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_9.addItem(spacerItem15)
        self.horizontalLayout_2.addLayout(self.verticalLayout_9)
        self.pteCode = QtWidgets.QPlainTextEdit(self.tab_time_converters)
        font = QtGui.QFont()
        font.setFamily("Monospace")
        self.pteCode.setFont(font)
        self.pteCode.setObjectName("pteCode")
        self.horizontalLayout_2.addWidget(self.pteCode)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.pb_save_converter = QtWidgets.QPushButton(self.tab_time_converters)
        self.pb_save_converter.setObjectName("pb_save_converter")
        self.verticalLayout_13.addWidget(self.pb_save_converter)
        self.pb_cancel_converter = QtWidgets.QPushButton(self.tab_time_converters)
        self.pb_cancel_converter.setObjectName("pb_cancel_converter")
        self.verticalLayout_13.addWidget(self.pb_cancel_converter)
        spacerItem16 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_13.addItem(spacerItem16)
        self.horizontalLayout_2.addLayout(self.verticalLayout_13)
        self.verticalLayout_18.addLayout(self.horizontalLayout_2)
        self.tabProject.addTab(self.tab_time_converters, "")
        self.verticalLayout_6.addWidget(self.tabProject)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem17 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem17)
        self.pbCancel = QtWidgets.QPushButton(dlgProject)
        self.pbCancel.setObjectName("pbCancel")
        self.horizontalLayout_4.addWidget(self.pbCancel)
        self.pbOK = QtWidgets.QPushButton(dlgProject)
        self.pbOK.setObjectName("pbOK")
        self.horizontalLayout_4.addWidget(self.pbOK)
        self.verticalLayout_6.addLayout(self.horizontalLayout_4)
        self.verticalLayout_7.addLayout(self.verticalLayout_6)

        self.retranslateUi(dlgProject)
        self.tabProject.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(dlgProject)

    def retranslateUi(self, dlgProject):
        _translate = QtCore.QCoreApplication.translate
        dlgProject.setWindowTitle(_translate("dlgProject", "Project"))
        self.label.setText(_translate("dlgProject", "Project name"))
        self.lbProjectFilePath.setText(_translate("dlgProject", "Project file path:"))
        self.label_7.setText(_translate("dlgProject", "Project date and time"))
        self.dteDate.setDisplayFormat(_translate("dlgProject", "yyyy-MM-dd hh:mm"))
        self.label_6.setText(_translate("dlgProject", "Project description"))
        self.lbTimeFormat.setText(_translate("dlgProject", "Project time format"))
        self.rbSeconds.setText(_translate("dlgProject", "seconds"))
        self.rbHMS.setText(_translate("dlgProject", "hh:mm:ss.mss"))
        self.lb_project_format_version.setText(_translate("dlgProject", "Project format version:"))
        self.tabProject.setTabText(self.tabProject.indexOf(self.tabInformation), _translate("dlgProject", "Information"))
        self.twBehaviors.setSortingEnabled(False)
        item = self.twBehaviors.horizontalHeaderItem(0)
        item.setText(_translate("dlgProject", "Behavior type"))
        item = self.twBehaviors.horizontalHeaderItem(1)
        item.setText(_translate("dlgProject", "Key"))
        item = self.twBehaviors.horizontalHeaderItem(2)
        item.setText(_translate("dlgProject", "Code"))
        item = self.twBehaviors.horizontalHeaderItem(3)
        item.setText(_translate("dlgProject", "Description"))
        item = self.twBehaviors.horizontalHeaderItem(4)
        item.setText(_translate("dlgProject", "Color"))
        item = self.twBehaviors.horizontalHeaderItem(5)
        item.setText(_translate("dlgProject", "Category"))
        item = self.twBehaviors.horizontalHeaderItem(6)
        item.setText(_translate("dlgProject", "Modifiers"))
        item = self.twBehaviors.horizontalHeaderItem(7)
        item.setText(_translate("dlgProject", "Exclusion"))
        item = self.twBehaviors.horizontalHeaderItem(8)
        item.setText(_translate("dlgProject", "Modifiers coding map"))
        self.pb_behavior.setText(_translate("dlgProject", "Behavior"))
        self.pb_import.setText(_translate("dlgProject", "Import ethogram"))
        self.pbBehaviorsCategories.setText(_translate("dlgProject", "Behavioral categories"))
        self.pb_exclusion_matrix.setText(_translate("dlgProject", "Exclusion matrix"))
        self.pbExportEthogram.setText(_translate("dlgProject", "Export ethogram"))
        self.lbObservationsState.setText(_translate("dlgProject", "TextLabel"))
        self.tabProject.setTabText(self.tabProject.indexOf(self.tabEthogram), _translate("dlgProject", "Ethogram"))
        self.twSubjects.setSortingEnabled(False)
        item = self.twSubjects.horizontalHeaderItem(0)
        item.setText(_translate("dlgProject", "Key"))
        item = self.twSubjects.horizontalHeaderItem(1)
        item.setText(_translate("dlgProject", "Subject name"))
        item = self.twSubjects.horizontalHeaderItem(2)
        item.setText(_translate("dlgProject", "Description"))
        self.pb_subjects.setText(_translate("dlgProject", "Subjects"))
        self.pbImportSubjectsFromProject.setText(_translate("dlgProject", "Import subjects"))
        self.pb_export_subjects.setText(_translate("dlgProject", "Export subjects"))
        self.lbSubjectsState.setText(_translate("dlgProject", "TextLabel"))
        self.tabProject.setTabText(self.tabProject.indexOf(self.tabSubjects), _translate("dlgProject", "Subjects"))
        self.twVariables.setSortingEnabled(False)
        item = self.twVariables.horizontalHeaderItem(0)
        item.setText(_translate("dlgProject", "Label"))
        item = self.twVariables.horizontalHeaderItem(1)
        item.setText(_translate("dlgProject", "Description"))
        item = self.twVariables.horizontalHeaderItem(2)
        item.setText(_translate("dlgProject", "Type"))
        item = self.twVariables.horizontalHeaderItem(3)
        item.setText(_translate("dlgProject", "Predefined value"))
        item = self.twVariables.horizontalHeaderItem(4)
        item.setText(_translate("dlgProject", "Set of values"))
        self.label_2.setText(_translate("dlgProject", "Label"))
        self.label_3.setText(_translate("dlgProject", "Description"))
        self.label_8.setText(_translate("dlgProject", "Type"))
        self.label_4.setText(_translate("dlgProject", "Predefined value"))
        self.label_9.setText(_translate("dlgProject", "Predefined timestamp"))
        self.dte_default_date.setDisplayFormat(_translate("dlgProject", "yyyy-MM-dd hh:mm:ss.zzz"))
        self.label_5.setText(_translate("dlgProject", "Set of values (separated by comma)"))
        self.pbAddVariable.setText(_translate("dlgProject", "Add variable"))
        self.pbRemoveVariable.setText(_translate("dlgProject", "Remove variable"))
        self.pbImportVarFromProject.setText(_translate("dlgProject", "Import variables\n"
"from a BORIS project"))
        self.tabProject.setTabText(self.tabProject.indexOf(self.tabIndependentVariables), _translate("dlgProject", "Independent variables"))
        self.twBehavCodingMap.setSortingEnabled(False)
        item = self.twBehavCodingMap.horizontalHeaderItem(0)
        item.setText(_translate("dlgProject", "Name"))
        item = self.twBehavCodingMap.horizontalHeaderItem(1)
        item.setText(_translate("dlgProject", "Behavior codes"))
        self.pbAddBehaviorsCodingMap.setText(_translate("dlgProject", "Add a behaviors coding map"))
        self.pbRemoveBehaviorsCodingMap.setText(_translate("dlgProject", "Remove behaviors coding map"))
        self.tabProject.setTabText(self.tabProject.indexOf(self.tabBehavCodingMap), _translate("dlgProject", "Behaviors coding map"))
        self.label_11.setText(_translate("dlgProject", "Time converters for external data"))
        self.tw_converters.setSortingEnabled(False)
        item = self.tw_converters.horizontalHeaderItem(0)
        item.setText(_translate("dlgProject", "Name"))
        item = self.tw_converters.horizontalHeaderItem(1)
        item.setText(_translate("dlgProject", "Description"))
        item = self.tw_converters.horizontalHeaderItem(2)
        item.setText(_translate("dlgProject", "Code"))
        self.pb_add_converter.setText(_translate("dlgProject", "Add new converter"))
        self.pb_modify_converter.setText(_translate("dlgProject", "Modify converter"))
        self.pb_delete_converter.setText(_translate("dlgProject", "Delete converter"))
        self.pb_load_from_file.setText(_translate("dlgProject", "Load converters from file"))
        self.pb_load_from_repo.setText(_translate("dlgProject", "Load converters from BORIS repository"))
        self.label_13.setText(_translate("dlgProject", "Name"))
        self.label_10.setText(_translate("dlgProject", "Description"))
        self.label_12.setText(_translate("dlgProject", "Python code"))
        self.pb_code_help.setText(_translate("dlgProject", "Help"))
        self.pb_save_converter.setText(_translate("dlgProject", "Save converter"))
        self.pb_cancel_converter.setText(_translate("dlgProject", "Cancel"))
        self.tabProject.setTabText(self.tabProject.indexOf(self.tab_time_converters), _translate("dlgProject", "Converters"))
        self.pbCancel.setText(_translate("dlgProject", "Cancel"))
        self.pbOK.setText(_translate("dlgProject", "OK"))
