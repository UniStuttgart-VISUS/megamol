namespace MegaMolConf {
    partial class TransferFunctionDialog {
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
            this.components = new System.ComponentModel.Container();
            this.lbl_Res = new System.Windows.Forms.Label();
            this.nUD_Res = new System.Windows.Forms.NumericUpDown();
            this.panel1 = new System.Windows.Forms.Panel();
            this.pb_TransferFunc = new System.Windows.Forms.PictureBox();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.b_R = new System.Windows.Forms.ToolStripButton();
            this.b_G = new System.Windows.Forms.ToolStripButton();
            this.b_B = new System.Windows.Forms.ToolStripButton();
            this.b_A = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.b_All = new System.Windows.Forms.ToolStripButton();
            this.b_None = new System.Windows.Forms.ToolStripButton();
            this.b_Ramp = new System.Windows.Forms.ToolStripButton();
            this.b_Zero = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.toolStripLabel2 = new System.Windows.Forms.ToolStripLabel();
            this.panel_Canvas = new MegaMolConf.NoflickerPanel();
            this.throttleTimer = new System.Windows.Forms.Timer(this.components);
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            ((System.ComponentModel.ISupportInitialize)(this.nUD_Res)).BeginInit();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pb_TransferFunc)).BeginInit();
            this.toolStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // lbl_Res
            // 
            this.lbl_Res.AutoSize = true;
            this.lbl_Res.Location = new System.Drawing.Point(186, 13);
            this.lbl_Res.Name = "lbl_Res";
            this.lbl_Res.Size = new System.Drawing.Size(29, 13);
            this.lbl_Res.TabIndex = 3;
            this.lbl_Res.Text = "Res:";
            // 
            // nUD_Res
            // 
            this.nUD_Res.Location = new System.Drawing.Point(319, 3);
            this.nUD_Res.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.nUD_Res.Name = "nUD_Res";
            this.nUD_Res.Size = new System.Drawing.Size(53, 20);
            this.nUD_Res.TabIndex = 0;
            this.nUD_Res.TabStop = false;
            this.nUD_Res.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.nUD_Res.Value = new decimal(new int[] {
            32,
            0,
            0,
            0});
            this.nUD_Res.ValueChanged += new System.EventHandler(this.NUDRes_ValChanged);
            // 
            // panel1
            // 
            this.panel1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel1.BackColor = System.Drawing.Color.Black;
            this.panel1.Controls.Add(this.pb_TransferFunc);
            this.panel1.Location = new System.Drawing.Point(12, 426);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(920, 63);
            this.panel1.TabIndex = 2;
            // 
            // pb_TransferFunc
            // 
            this.pb_TransferFunc.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.pb_TransferFunc.Location = new System.Drawing.Point(0, 1);
            this.pb_TransferFunc.Name = "pb_TransferFunc";
            this.pb_TransferFunc.Size = new System.Drawing.Size(923, 125);
            this.pb_TransferFunc.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pb_TransferFunc.TabIndex = 0;
            this.pb_TransferFunc.TabStop = false;
            // 
            // toolStrip1
            // 
            this.toolStrip1.GripStyle = System.Windows.Forms.ToolStripGripStyle.Hidden;
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.b_R,
            this.b_G,
            this.b_B,
            this.b_A,
            this.toolStripSeparator1,
            this.b_All,
            this.b_None,
            this.toolStripSeparator3,
            this.b_Ramp,
            this.b_Zero,
            this.toolStripSeparator2,
            this.toolStripLabel2});
            this.toolStrip1.Location = new System.Drawing.Point(0, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(944, 25);
            this.toolStrip1.TabIndex = 1;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // b_R
            // 
            this.b_R.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.b_R.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.b_R.Name = "b_R";
            this.b_R.Size = new System.Drawing.Size(23, 22);
            this.b_R.Text = "R";
            this.b_R.Click += new System.EventHandler(this.color_Clicked);
            // 
            // b_G
            // 
            this.b_G.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.b_G.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.b_G.Name = "b_G";
            this.b_G.Size = new System.Drawing.Size(23, 22);
            this.b_G.Text = "G";
            this.b_G.Click += new System.EventHandler(this.color_Clicked);
            // 
            // b_B
            // 
            this.b_B.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.b_B.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.b_B.Name = "b_B";
            this.b_B.Size = new System.Drawing.Size(23, 22);
            this.b_B.Text = "B";
            this.b_B.Click += new System.EventHandler(this.color_Clicked);
            // 
            // b_A
            // 
            this.b_A.Checked = true;
            this.b_A.CheckState = System.Windows.Forms.CheckState.Checked;
            this.b_A.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.b_A.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.b_A.Name = "b_A";
            this.b_A.Size = new System.Drawing.Size(23, 22);
            this.b_A.Text = "A";
            this.b_A.Click += new System.EventHandler(this.color_Clicked);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 25);
            // 
            // b_All
            // 
            this.b_All.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.b_All.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.b_All.Name = "b_All";
            this.b_All.Size = new System.Drawing.Size(25, 22);
            this.b_All.Text = "All";
            this.b_All.Click += new System.EventHandler(this.all_Clicked);
            // 
            // b_None
            // 
            this.b_None.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.b_None.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.b_None.Name = "b_None";
            this.b_None.Size = new System.Drawing.Size(40, 22);
            this.b_None.Text = "None";
            this.b_None.Click += new System.EventHandler(this.zero_Clicked);
            // 
            // b_Ramp
            // 
            this.b_Ramp.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.b_Ramp.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.b_Ramp.Name = "b_Ramp";
            this.b_Ramp.Size = new System.Drawing.Size(42, 22);
            this.b_Ramp.Text = "Ramp";
            this.b_Ramp.Click += new System.EventHandler(this.btn_Ramp_Click);
            // 
            // b_Zero
            // 
            this.b_Zero.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.b_Zero.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.b_Zero.Name = "b_Zero";
            this.b_Zero.Size = new System.Drawing.Size(35, 22);
            this.b_Zero.Text = "Zero";
            this.b_Zero.Click += new System.EventHandler(this.btn_Zero_Click);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(6, 25);
            // 
            // toolStripLabel2
            // 
            this.toolStripLabel2.Name = "toolStripLabel2";
            this.toolStripLabel2.Size = new System.Drawing.Size(63, 22);
            this.toolStripLabel2.Text = "Resolution";
            // 
            // panel_Canvas
            // 
            this.panel_Canvas.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel_Canvas.BackColor = System.Drawing.Color.White;
            this.panel_Canvas.Location = new System.Drawing.Point(12, 29);
            this.panel_Canvas.Name = "panel_Canvas";
            this.panel_Canvas.Size = new System.Drawing.Size(920, 391);
            this.panel_Canvas.TabIndex = 4;
            this.panel_Canvas.Click += new System.EventHandler(this.PanelCanvas_Click);
            this.panel_Canvas.Paint += new System.Windows.Forms.PaintEventHandler(this.PanelCanvas_Paint);
            this.panel_Canvas.MouseDown += new System.Windows.Forms.MouseEventHandler(this.PanelCanvas_MouseDown);
            this.panel_Canvas.MouseMove += new System.Windows.Forms.MouseEventHandler(this.PanelCanvas_MouseMove);
            this.panel_Canvas.MouseUp += new System.Windows.Forms.MouseEventHandler(this.PanelCanvas_MouseUp);
            this.panel_Canvas.Resize += new System.EventHandler(this.PanelCanvas_Resize);
            // 
            // throttleTimer
            // 
            this.throttleTimer.Tick += new System.EventHandler(this.throttleTimer_Tick);
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            this.toolStripSeparator3.Size = new System.Drawing.Size(6, 25);
            // 
            // TransferFunctionDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(944, 501);
            this.Controls.Add(this.nUD_Res);
            this.Controls.Add(this.toolStrip1);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.lbl_Res);
            this.Controls.Add(this.panel_Canvas);
            this.KeyPreview = true;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "TransferFunctionDialog";
            this.ShowIcon = false;
            this.Text = "Transfer function";
            this.Load += new System.EventHandler(this.TransferFunctionDialog_Load);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.TransferFunctionDialog_KeyDown);
            ((System.ComponentModel.ISupportInitialize)(this.nUD_Res)).EndInit();
            this.panel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pb_TransferFunc)).EndInit();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private NoflickerPanel panel_Canvas;
        private System.Windows.Forms.Label lbl_Res;
        private System.Windows.Forms.NumericUpDown nUD_Res;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.PictureBox pb_TransferFunc;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton b_R;
        private System.Windows.Forms.ToolStripButton b_G;
        private System.Windows.Forms.ToolStripButton b_B;
        private System.Windows.Forms.ToolStripButton b_A;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton b_All;
        private System.Windows.Forms.ToolStripButton b_None;
        private System.Windows.Forms.ToolStripButton b_Ramp;
        private System.Windows.Forms.ToolStripButton b_Zero;
        private System.Windows.Forms.ToolStripLabel toolStripLabel2;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        private System.Windows.Forms.Timer throttleTimer;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
    }
}