using System;
using System.Drawing;
using System.Windows.Forms;

namespace MegaMolConf {

    /// <summary>
    /// Dialog to edit the MegaMol command line arguments for start
    /// </summary>
    public partial class StartParamDialog : Form {

        /// <summary>
        /// Possible values for StartShellType
        /// </summary>
        public enum StartShellType : int {
            Direct = 0,
            Cmd = 1,
            Powershell = 2
        };

        /// <summary>
        /// Gets or sets the start shell type
        /// </summary>
        public StartShellType ShellType {
            get {
                if (radioButton1.Checked) return StartShellType.Cmd;
                if (radioButton2.Checked) return StartShellType.Powershell;
                return StartShellType.Direct;
            }
            set {
                switch (value) {
                    case StartShellType.Direct:
                        radioButton3.Checked = true;
                        break;
                    case StartShellType.Cmd:
                        radioButton1.Checked = true;
                        break;
                    case StartShellType.Powershell:
                        radioButton2.Checked = true;
                        break;
                    default:
                        radioButton3.Checked = true;
                        break;
                }
            }
        }

        /// <summary>
        /// Gets or sets the flag if the shell is to be kept open
        /// </summary>
        public bool KeepShellOpen {
            get { return checkBox1.Checked; }
            set { checkBox1.Checked = value; }
        }

        /// <summary>
        /// Gets or sets the flag if we want to manipulate parameters of the live MegaMol child process
        /// </summary>
        public bool LiveConnection {
            get { return chkLive.Checked; }
            set { chkLive.Checked = value; }
        }

        /// <summary>
        /// Gets or sets the standard cmd arguments
        /// </summary>
        public string StdCmdArgs { get; set; }

        /// <summary>
        /// Gets or sets the standart powershell arguments
        /// </summary>
        public string StdPSArgs { get; set; }

        /// <summary>
        /// Gets or sets the argument history
        /// </summary>
        public System.Collections.Specialized.StringCollection ArgsHistory {
            get {
                System.Collections.Specialized.StringCollection sc = new System.Collections.Specialized.StringCollection();
                if ((comboBox1.Items != null) && (comboBox1.Items.Count > 0)) {
                    foreach (object o in comboBox1.Items) {
                        sc.Add(o.ToString());
                    }
                }
                return sc;
            }
            set {
                comboBox1.Items.Clear();
                if ((value != null) && (value.Count > 0)) {
                    foreach (string s in value) {
                        comboBox1.Items.Add(s);
                    }
                }
            }
        }

        /// <summary>
        /// Gets or sets the edited start arguments
        /// </summary>
        public string StartArgs {
            get { return comboBox1.Text; }
            set { comboBox1.Text = value; }
        }

        /// <summary>
        /// Gets or sets the MegaMol application to start
        /// </summary>
        public string Application {
            get { return textBoxApp.Text; }
            set { textBoxApp.Text = value; }
        }

        /// <summary>
        /// Gets or sets the working directory to start MegaMol in
        /// </summary>
        public string WorkingDir {
            get { return textBoxWorkingDir.Text; }
            set { textBoxWorkingDir.Text = value; }
        }

        /// <summary>
        /// Gets or sets the flag to use the application directory as working directory
        /// </summary>
        public bool UseApplicationWorkingDir {
            get { return checkBox2.Checked; }
            set { checkBox2.Checked = value; }
        }

        /// <summary>
        /// Ctor
        /// </summary>
        public StartParamDialog() {
            InitializeComponent();
            Font = SystemFonts.DefaultFont;
        }

        /// <summary>
        /// Show the "template" context menu strip
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void button1_Click(object sender, EventArgs e) {
            contextMenuStrip1.Show(button1, new Point(0, button1.Height));
        }

        /// <summary>
        /// Sets the arguments to the standard CMD arguments
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void standardCmdCommandToolStripMenuItem_Click(object sender, EventArgs e) {
            comboBox1.Text = StdCmdArgs;
        }

        /// <summary>
        /// Sets the arguments to the standard Powershell arguments
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void standardPowershellCommandToolStripMenuItem_Click(object sender, EventArgs e) {
            comboBox1.Text = StdPSArgs;
        }

        /// <summary>
        /// Enable input of working directory only if not using the application directory
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void checkBox2_CheckedChanged(object sender, EventArgs e) {
            textBoxWorkingDir.Enabled = !checkBox2.Checked;
        }

        private void browseMegaMolButton_Click(object sender, EventArgs e) {
            Util.ApplicationSearchDialog asd = new Util.ApplicationSearchDialog();
            asd.FileName = textBoxApp.Text;
            if (asd.ShowDialog(this) == DialogResult.OK) {
                textBoxApp.Text = asd.FileName;
            }
        }
    }

}
