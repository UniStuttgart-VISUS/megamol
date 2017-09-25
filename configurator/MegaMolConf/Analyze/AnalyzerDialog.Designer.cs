namespace MegaMolConf.Analyze {
    partial class AnalyzerDialog {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing) {
            if (disposing && (components != null)) {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent() {
            this.label1 = new System.Windows.Forms.Label();
            this.megaMolTextBox = new System.Windows.Forms.TextBox();
            this.browseMegaMolButton = new System.Windows.Forms.Button();
            this.browseWorkDirButton = new System.Windows.Forms.Button();
            this.workDirTextBox = new System.Windows.Forms.TextBox();
            this.useWorkDirCheckBox = new System.Windows.Forms.CheckBox();
            this.paramsTextBox = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.buttonExecute = new System.Windows.Forms.Button();
            this.buttonOk = new System.Windows.Forms.Button();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.saveCheckBox = new System.Windows.Forms.CheckBox();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.reportRichTextBox = new System.Windows.Forms.RichTextBox();
            this.buttonClearLog = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(9, 15);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(99, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "MegaMol Frontend:";
            // 
            // megaMolTextBox
            // 
            this.megaMolTextBox.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.megaMolTextBox.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.megaMolTextBox.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.FileSystem;
            this.megaMolTextBox.Location = new System.Drawing.Point(122, 12);
            this.megaMolTextBox.Name = "megaMolTextBox";
            this.megaMolTextBox.Size = new System.Drawing.Size(1099, 20);
            this.megaMolTextBox.TabIndex = 1;
            // 
            // browseMegaMolButton
            // 
            this.browseMegaMolButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.browseMegaMolButton.Location = new System.Drawing.Point(1227, 10);
            this.browseMegaMolButton.Name = "browseMegaMolButton";
            this.browseMegaMolButton.Size = new System.Drawing.Size(25, 23);
            this.browseMegaMolButton.TabIndex = 2;
            this.browseMegaMolButton.Text = "...";
            this.browseMegaMolButton.UseVisualStyleBackColor = true;
            this.browseMegaMolButton.Click += new System.EventHandler(this.browseMegaMolButton_Click);
            // 
            // browseWorkDirButton
            // 
            this.browseWorkDirButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.browseWorkDirButton.Location = new System.Drawing.Point(1227, 36);
            this.browseWorkDirButton.Name = "browseWorkDirButton";
            this.browseWorkDirButton.Size = new System.Drawing.Size(25, 23);
            this.browseWorkDirButton.TabIndex = 4;
            this.browseWorkDirButton.Text = "...";
            this.browseWorkDirButton.UseVisualStyleBackColor = true;
            this.browseWorkDirButton.Click += new System.EventHandler(this.browseWorkDirButton_Click);
            // 
            // workDirTextBox
            // 
            this.workDirTextBox.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.workDirTextBox.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
            this.workDirTextBox.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.FileSystemDirectories;
            this.workDirTextBox.Location = new System.Drawing.Point(122, 38);
            this.workDirTextBox.Name = "workDirTextBox";
            this.workDirTextBox.Size = new System.Drawing.Size(1099, 20);
            this.workDirTextBox.TabIndex = 3;
            this.workDirTextBox.TextChanged += new System.EventHandler(this.workDirTextBox_TextChanged);
            // 
            // useWorkDirCheckBox
            // 
            this.useWorkDirCheckBox.AutoSize = true;
            this.useWorkDirCheckBox.Location = new System.Drawing.Point(12, 40);
            this.useWorkDirCheckBox.Name = "useWorkDirCheckBox";
            this.useWorkDirCheckBox.Size = new System.Drawing.Size(88, 17);
            this.useWorkDirCheckBox.TabIndex = 5;
            this.useWorkDirCheckBox.Text = "Working Dir.:";
            this.useWorkDirCheckBox.UseVisualStyleBackColor = true;
            // 
            // paramsTextBox
            // 
            this.paramsTextBox.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.paramsTextBox.Location = new System.Drawing.Point(122, 64);
            this.paramsTextBox.Name = "paramsTextBox";
            this.paramsTextBox.ReadOnly = true;
            this.paramsTextBox.Size = new System.Drawing.Size(1130, 20);
            this.paramsTextBox.TabIndex = 10;
            this.paramsTextBox.Text = "-i GenStateFile genState -v ::genState::gen::filename \"MegaMolConf.state\"";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(9, 67);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(63, 13);
            this.label3.TabIndex = 9;
            this.label3.Text = "Parameters:";
            // 
            // buttonExecute
            // 
            this.buttonExecute.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.buttonExecute.Image = global::MegaMolConf.Properties.Resources.start;
            this.buttonExecute.ImageAlign = System.Drawing.ContentAlignment.MiddleRight;
            this.buttonExecute.Location = new System.Drawing.Point(576, 90);
            this.buttonExecute.Name = "buttonExecute";
            this.buttonExecute.Size = new System.Drawing.Size(112, 23);
            this.buttonExecute.TabIndex = 23;
            this.buttonExecute.Text = "Execute";
            this.buttonExecute.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            this.buttonExecute.TextImageRelation = System.Windows.Forms.TextImageRelation.ImageBeforeText;
            this.buttonExecute.UseVisualStyleBackColor = true;
            this.buttonExecute.Click += new System.EventHandler(this.buttonExecute_Click);
            // 
            // buttonOk
            // 
            this.buttonOk.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.buttonOk.Enabled = false;
            this.buttonOk.Location = new System.Drawing.Point(1046, 646);
            this.buttonOk.Name = "buttonOk";
            this.buttonOk.Size = new System.Drawing.Size(100, 23);
            this.buttonOk.TabIndex = 24;
            this.buttonOk.Text = "Use this State File";
            this.buttonOk.UseVisualStyleBackColor = true;
            this.buttonOk.Click += new System.EventHandler(this.buttonOk_Click);
            // 
            // buttonCancel
            // 
            this.buttonCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.buttonCancel.Location = new System.Drawing.Point(1152, 646);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(100, 23);
            this.buttonCancel.TabIndex = 25;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            // 
            // saveCheckBox
            // 
            this.saveCheckBox.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.saveCheckBox.AutoSize = true;
            this.saveCheckBox.Checked = true;
            this.saveCheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.saveCheckBox.Location = new System.Drawing.Point(968, 650);
            this.saveCheckBox.Name = "saveCheckBox";
            this.saveCheckBox.Size = new System.Drawing.Size(72, 17);
            this.saveCheckBox.TabIndex = 28;
            this.saveCheckBox.Text = "Save and";
            this.saveCheckBox.UseVisualStyleBackColor = true;
            // 
            // folderBrowserDialog1
            // 
            this.folderBrowserDialog1.Description = "Select Working Directory...";
            // 
            // reportRichTextBox
            // 
            this.reportRichTextBox.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.reportRichTextBox.Location = new System.Drawing.Point(12, 119);
            this.reportRichTextBox.Name = "reportRichTextBox";
            this.reportRichTextBox.Size = new System.Drawing.Size(1240, 521);
            this.reportRichTextBox.TabIndex = 29;
            this.reportRichTextBox.Text = "";
            // 
            // buttonClearLog
            // 
            this.buttonClearLog.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.buttonClearLog.Location = new System.Drawing.Point(12, 646);
            this.buttonClearLog.Name = "buttonClearLog";
            this.buttonClearLog.Size = new System.Drawing.Size(45, 23);
            this.buttonClearLog.TabIndex = 30;
            this.buttonClearLog.Text = "clear";
            this.buttonClearLog.UseVisualStyleBackColor = true;
            this.buttonClearLog.Click += new System.EventHandler(this.button1_Click);
            // 
            // AnalyzerDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.buttonCancel;
            this.ClientSize = new System.Drawing.Size(1264, 681);
            this.Controls.Add(this.buttonClearLog);
            this.Controls.Add(this.reportRichTextBox);
            this.Controls.Add(this.saveCheckBox);
            this.Controls.Add(this.buttonOk);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonExecute);
            this.Controls.Add(this.paramsTextBox);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.useWorkDirCheckBox);
            this.Controls.Add(this.browseWorkDirButton);
            this.Controls.Add(this.workDirTextBox);
            this.Controls.Add(this.browseMegaMolButton);
            this.Controls.Add(this.megaMolTextBox);
            this.Controls.Add(this.label1);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Pixel);
            this.Name = "AnalyzerDialog";
            this.ShowIcon = false;
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.WindowsDefaultBounds;
            this.Text = "MegaMol™ Configurator - Analyze MegaMol™ ...";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox megaMolTextBox;
        private System.Windows.Forms.Button browseMegaMolButton;
        private System.Windows.Forms.Button browseWorkDirButton;
        private System.Windows.Forms.TextBox workDirTextBox;
        private System.Windows.Forms.CheckBox useWorkDirCheckBox;
        private System.Windows.Forms.TextBox paramsTextBox;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button buttonExecute;
        private System.Windows.Forms.Button buttonOk;
        private System.Windows.Forms.Button buttonCancel;
        private System.Windows.Forms.CheckBox saveCheckBox;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private System.Windows.Forms.RichTextBox reportRichTextBox;
        private System.Windows.Forms.Button buttonClearLog;
    }
}