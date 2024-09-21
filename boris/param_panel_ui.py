# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'boris/param_panel.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1037, 890)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lbSubjects = QtWidgets.QLabel(Dialog)
        self.lbSubjects.setObjectName("lbSubjects")
        self.verticalLayout_2.addWidget(self.lbSubjects)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pbSelectAllSubjects = QtWidgets.QPushButton(Dialog)
        self.pbSelectAllSubjects.setObjectName("pbSelectAllSubjects")
        self.horizontalLayout_3.addWidget(self.pbSelectAllSubjects)
        self.pbUnselectAllSubjects = QtWidgets.QPushButton(Dialog)
        self.pbUnselectAllSubjects.setObjectName("pbUnselectAllSubjects")
        self.horizontalLayout_3.addWidget(self.pbUnselectAllSubjects)
        self.pbReverseSubjectsSelection = QtWidgets.QPushButton(Dialog)
        self.pbReverseSubjectsSelection.setObjectName("pbReverseSubjectsSelection")
        self.horizontalLayout_3.addWidget(self.pbReverseSubjectsSelection)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.lwSubjects = QtWidgets.QListWidget(Dialog)
        self.lwSubjects.setObjectName("lwSubjects")
        self.verticalLayout_2.addWidget(self.lwSubjects)
        self.lbBehaviors = QtWidgets.QLabel(Dialog)
        self.lbBehaviors.setObjectName("lbBehaviors")
        self.verticalLayout_2.addWidget(self.lbBehaviors)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pbSelectAllBehaviors = QtWidgets.QPushButton(Dialog)
        self.pbSelectAllBehaviors.setObjectName("pbSelectAllBehaviors")
        self.horizontalLayout_4.addWidget(self.pbSelectAllBehaviors)
        self.pbUnselectAllBehaviors = QtWidgets.QPushButton(Dialog)
        self.pbUnselectAllBehaviors.setObjectName("pbUnselectAllBehaviors")
        self.horizontalLayout_4.addWidget(self.pbUnselectAllBehaviors)
        self.pbReverseBehaviorsSelection = QtWidgets.QPushButton(Dialog)
        self.pbReverseBehaviorsSelection.setObjectName("pbReverseBehaviorsSelection")
        self.horizontalLayout_4.addWidget(self.pbReverseBehaviorsSelection)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.lwBehaviors = QtWidgets.QListWidget(Dialog)
        self.lwBehaviors.setObjectName("lwBehaviors")
        self.verticalLayout_2.addWidget(self.lwBehaviors)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.cbIncludeModifiers = QtWidgets.QCheckBox(Dialog)
        self.cbIncludeModifiers.setObjectName("cbIncludeModifiers")
        self.horizontalLayout_9.addWidget(self.cbIncludeModifiers)
        self.cb_exclude_non_coded_modifiers = QtWidgets.QCheckBox(Dialog)
        self.cb_exclude_non_coded_modifiers.setChecked(True)
        self.cb_exclude_non_coded_modifiers.setObjectName("cb_exclude_non_coded_modifiers")
        self.horizontalLayout_9.addWidget(self.cb_exclude_non_coded_modifiers)
        self.cbExcludeBehaviors = QtWidgets.QCheckBox(Dialog)
        self.cbExcludeBehaviors.setObjectName("cbExcludeBehaviors")
        self.horizontalLayout_9.addWidget(self.cbExcludeBehaviors)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_9)
        self.frm_time_bin_size = QtWidgets.QFrame(Dialog)
        self.frm_time_bin_size.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frm_time_bin_size.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frm_time_bin_size.setObjectName("frm_time_bin_size")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frm_time_bin_size)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.lb_time_bin_size = QtWidgets.QLabel(self.frm_time_bin_size)
        self.lb_time_bin_size.setObjectName("lb_time_bin_size")
        self.horizontalLayout_7.addWidget(self.lb_time_bin_size)
        self.sb_time_bin_size = QtWidgets.QSpinBox(self.frm_time_bin_size)
        self.sb_time_bin_size.setMaximum(86400)
        self.sb_time_bin_size.setSingleStep(10)
        self.sb_time_bin_size.setObjectName("sb_time_bin_size")
        self.horizontalLayout_7.addWidget(self.sb_time_bin_size)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem3)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_7)
        self.verticalLayout_2.addWidget(self.frm_time_bin_size)
        self.frm_time = QtWidgets.QFrame(Dialog)
        self.frm_time.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frm_time.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frm_time.setObjectName("frm_time")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frm_time)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.lb_time_interval = QtWidgets.QLabel(self.frm_time)
        self.lb_time_interval.setObjectName("lb_time_interval")
        self.verticalLayout_3.addWidget(self.lb_time_interval)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.rb_observed_events = QtWidgets.QRadioButton(self.frm_time)
        self.rb_observed_events.setObjectName("rb_observed_events")
        self.horizontalLayout_5.addWidget(self.rb_observed_events)
        self.rb_user_defined = QtWidgets.QRadioButton(self.frm_time)
        self.rb_user_defined.setObjectName("rb_user_defined")
        self.horizontalLayout_5.addWidget(self.rb_user_defined)
        self.rb_media_duration = QtWidgets.QRadioButton(self.frm_time)
        self.rb_media_duration.setCheckable(True)
        self.rb_media_duration.setChecked(False)
        self.rb_media_duration.setObjectName("rb_media_duration")
        self.horizontalLayout_5.addWidget(self.rb_media_duration)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.frm_time_interval = QtWidgets.QFrame(self.frm_time)
        self.frm_time_interval.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frm_time_interval.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frm_time_interval.setObjectName("frm_time_interval")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frm_time_interval)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lbStartTime = QtWidgets.QLabel(self.frm_time_interval)
        self.lbStartTime.setObjectName("lbStartTime")
        self.horizontalLayout.addWidget(self.lbStartTime)
        self.label_2 = QtWidgets.QLabel(self.frm_time_interval)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.lbEndTime = QtWidgets.QLabel(self.frm_time_interval)
        self.lbEndTime.setObjectName("lbEndTime")
        self.horizontalLayout_6.addWidget(self.lbEndTime)
        self.label_3 = QtWidgets.QLabel(self.frm_time_interval)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem6)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem7)
        self.verticalLayout_3.addWidget(self.frm_time_interval)
        self.verticalLayout_2.addWidget(self.frm_time)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem8)
        self.pbCancel = QtWidgets.QPushButton(Dialog)
        self.pbCancel.setObjectName("pbCancel")
        self.horizontalLayout_2.addWidget(self.pbCancel)
        self.pbOK = QtWidgets.QPushButton(Dialog)
        self.pbOK.setObjectName("pbOK")
        self.horizontalLayout_2.addWidget(self.pbOK)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Parameters"))
        self.lbSubjects.setText(_translate("Dialog", "Subjects"))
        self.pbSelectAllSubjects.setText(_translate("Dialog", "Select all"))
        self.pbUnselectAllSubjects.setText(_translate("Dialog", "Unselect all"))
        self.pbReverseSubjectsSelection.setText(_translate("Dialog", "Reverse selection"))
        self.lbBehaviors.setText(_translate("Dialog", "Behaviors"))
        self.pbSelectAllBehaviors.setText(_translate("Dialog", "Select all"))
        self.pbUnselectAllBehaviors.setText(_translate("Dialog", "Unselect all"))
        self.pbReverseBehaviorsSelection.setText(_translate("Dialog", "Reverse selection"))
        self.cbIncludeModifiers.setText(_translate("Dialog", "Include modifiers"))
        self.cb_exclude_non_coded_modifiers.setText(_translate("Dialog", "Exclude non coded modifiers"))
        self.cbExcludeBehaviors.setText(_translate("Dialog", "Exclude behaviors without events"))
        self.lb_time_bin_size.setText(_translate("Dialog", "Time bin size (s)"))
        self.lb_time_interval.setText(_translate("Dialog", "Time interval"))
        self.rb_observed_events.setText(_translate("Dialog", "Observed events"))
        self.rb_user_defined.setText(_translate("Dialog", "User defined"))
        self.rb_media_duration.setText(_translate("Dialog", "Media file(s) duration"))
        self.lbStartTime.setText(_translate("Dialog", "Start time"))
        self.lbEndTime.setText(_translate("Dialog", "End time"))
        self.pbCancel.setText(_translate("Dialog", "Cancel"))
        self.pbOK.setText(_translate("Dialog", "OK"))
