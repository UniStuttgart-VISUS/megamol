namespace MegaMolConf {
    partial class TransferFunc1DDialog {
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
            this.lbl_Res = new System.Windows.Forms.Label();
            this.nUD_Res = new System.Windows.Forms.NumericUpDown();
            this.cb_R = new System.Windows.Forms.CheckBox();
            this.cb_G = new System.Windows.Forms.CheckBox();
            this.cb_B = new System.Windows.Forms.CheckBox();
            this.cb_A = new System.Windows.Forms.CheckBox();
            this.cb_All = new System.Windows.Forms.CheckBox();
            this.panel_Canvas = new MegaMolConf.NoflickerPanel();
            this.panel1 = new System.Windows.Forms.Panel();
            this.pb_TransferFunc = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.nUD_Res)).BeginInit();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pb_TransferFunc)).BeginInit();
            this.SuspendLayout();
            // 
            // lbl_Res
            // 
            this.lbl_Res.AutoSize = true;
            this.lbl_Res.Location = new System.Drawing.Point(186, 13);
            this.lbl_Res.Name = "lbl_Res";
            this.lbl_Res.Size = new System.Drawing.Size(29, 13);
            this.lbl_Res.TabIndex = 6;
            this.lbl_Res.Text = "Res:";
            // 
            // nUD_Res
            // 
            this.nUD_Res.Location = new System.Drawing.Point(221, 11);
            this.nUD_Res.Maximum = new decimal(new int[] {
            1024,
            0,
            0,
            0});
            this.nUD_Res.Name = "nUD_Res";
            this.nUD_Res.Size = new System.Drawing.Size(106, 20);
            this.nUD_Res.TabIndex = 7;
            this.nUD_Res.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.nUD_Res.Value = new decimal(new int[] {
            256,
            0,
            0,
            0});
            this.nUD_Res.ValueChanged += new System.EventHandler(this.NUDRes_ValChanged);
            // 
            // cb_R
            // 
            this.cb_R.Appearance = System.Windows.Forms.Appearance.Button;
            this.cb_R.AutoSize = true;
            this.cb_R.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.cb_R.Location = new System.Drawing.Point(12, 11);
            this.cb_R.Name = "cb_R";
            this.cb_R.Size = new System.Drawing.Size(25, 23);
            this.cb_R.TabIndex = 8;
            this.cb_R.Text = "R";
            this.cb_R.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.cb_R.UseVisualStyleBackColor = true;
            this.cb_R.CheckedChanged += new System.EventHandler(this.CBs_CheckedChanged);
            // 
            // cb_G
            // 
            this.cb_G.Appearance = System.Windows.Forms.Appearance.Button;
            this.cb_G.AutoSize = true;
            this.cb_G.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.cb_G.Location = new System.Drawing.Point(43, 11);
            this.cb_G.Name = "cb_G";
            this.cb_G.Size = new System.Drawing.Size(25, 23);
            this.cb_G.TabIndex = 8;
            this.cb_G.Text = "G";
            this.cb_G.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.cb_G.UseVisualStyleBackColor = true;
            this.cb_G.CheckedChanged += new System.EventHandler(this.CBs_CheckedChanged);
            // 
            // cb_B
            // 
            this.cb_B.Appearance = System.Windows.Forms.Appearance.Button;
            this.cb_B.AutoSize = true;
            this.cb_B.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.cb_B.Location = new System.Drawing.Point(74, 11);
            this.cb_B.Name = "cb_B";
            this.cb_B.Size = new System.Drawing.Size(24, 23);
            this.cb_B.TabIndex = 8;
            this.cb_B.Text = "B";
            this.cb_B.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.cb_B.UseVisualStyleBackColor = true;
            this.cb_B.CheckedChanged += new System.EventHandler(this.CBs_CheckedChanged);
            // 
            // cb_A
            // 
            this.cb_A.Appearance = System.Windows.Forms.Appearance.Button;
            this.cb_A.AutoSize = true;
            this.cb_A.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.cb_A.Location = new System.Drawing.Point(105, 11);
            this.cb_A.Name = "cb_A";
            this.cb_A.Size = new System.Drawing.Size(24, 23);
            this.cb_A.TabIndex = 8;
            this.cb_A.Text = "A";
            this.cb_A.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.cb_A.UseVisualStyleBackColor = true;
            this.cb_A.CheckedChanged += new System.EventHandler(this.CBs_CheckedChanged);
            // 
            // cb_All
            // 
            this.cb_All.Appearance = System.Windows.Forms.Appearance.Button;
            this.cb_All.AutoSize = true;
            this.cb_All.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.cb_All.Location = new System.Drawing.Point(135, 11);
            this.cb_All.Name = "cb_All";
            this.cb_All.Size = new System.Drawing.Size(28, 23);
            this.cb_All.TabIndex = 8;
            this.cb_All.Text = "All";
            this.cb_All.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.cb_All.UseVisualStyleBackColor = true;
            this.cb_All.CheckedChanged += new System.EventHandler(this.CBAll_CheckedChanged);
            // 
            // panel_Canvas
            // 
            this.panel_Canvas.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel_Canvas.BackColor = System.Drawing.Color.White;
            this.panel_Canvas.Location = new System.Drawing.Point(12, 42);
            this.panel_Canvas.Name = "panel_Canvas";
            this.panel_Canvas.Size = new System.Drawing.Size(536, 327);
            this.panel_Canvas.TabIndex = 5;
            this.panel_Canvas.Click += new System.EventHandler(this.PanelCanvas_Click);
            this.panel_Canvas.Paint += new System.Windows.Forms.PaintEventHandler(this.PanelCanvas_Paint);
            this.panel_Canvas.MouseDown += new System.Windows.Forms.MouseEventHandler(this.PanelCanvas_MouseDown);
            this.panel_Canvas.MouseMove += new System.Windows.Forms.MouseEventHandler(this.PanelCanvas_MouseMove);
            this.panel_Canvas.MouseUp += new System.Windows.Forms.MouseEventHandler(this.PanelCanvas_MouseUp);
            this.panel_Canvas.Resize += new System.EventHandler(this.PanelCanvas_Resize);
            // 
            // panel1
            // 
            this.panel1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel1.BackColor = System.Drawing.Color.Black;
            this.panel1.Controls.Add(this.pb_TransferFunc);
            this.panel1.Location = new System.Drawing.Point(12, 375);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(536, 63);
            this.panel1.TabIndex = 10;
            // 
            // pb_TransferFunc
            // 
            this.pb_TransferFunc.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.pb_TransferFunc.Location = new System.Drawing.Point(0, 1);
            this.pb_TransferFunc.Name = "pb_TransferFunc";
            this.pb_TransferFunc.Size = new System.Drawing.Size(536, 125);
            this.pb_TransferFunc.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pb_TransferFunc.TabIndex = 10;
            this.pb_TransferFunc.TabStop = false;
            // 
            // TransferFunc1DDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(560, 450);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.cb_All);
            this.Controls.Add(this.cb_A);
            this.Controls.Add(this.cb_B);
            this.Controls.Add(this.cb_G);
            this.Controls.Add(this.cb_R);
            this.Controls.Add(this.nUD_Res);
            this.Controls.Add(this.lbl_Res);
            this.Controls.Add(this.panel_Canvas);
            this.Name = "TransferFunc1DDialog";
            this.Text = "TransferFunc1DDialog";
            ((System.ComponentModel.ISupportInitialize)(this.nUD_Res)).EndInit();
            this.panel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pb_TransferFunc)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private NoflickerPanel panel_Canvas;
        private System.Windows.Forms.Label lbl_Res;
        private System.Windows.Forms.NumericUpDown nUD_Res;
        private System.Windows.Forms.CheckBox cb_R;
        private System.Windows.Forms.CheckBox cb_G;
        private System.Windows.Forms.CheckBox cb_B;
        private System.Windows.Forms.CheckBox cb_A;
        private System.Windows.Forms.CheckBox cb_All;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.PictureBox pb_TransferFunc;
    }
}