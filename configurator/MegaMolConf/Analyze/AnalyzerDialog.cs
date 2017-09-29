using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace MegaMolConf.Analyze {

    /// <summary>
    /// Dialog to control the new and fancy MegaMol™ analysis process
    /// </summary>
    public partial class AnalyzerDialog : Form {

        public string MegaMolPath {
            get { return this.megaMolTextBox.Text; }
            set { this.megaMolTextBox.Text = value; }
        }

        public string WorkingDirectory {
            get { return this.workDirTextBox.Text; }
            set {
                this.workDirTextBox.Text = value;
                if (!string.IsNullOrWhiteSpace(this.workDirTextBox.Text)) {
                    useWorkDirCheckBox.Checked = true;
                }
            }
        }

        public List<Data.PluginFile> Plugins { get; set; }

        public bool SaveAfterOk {
            get { return saveCheckBox.Checked; }
            set { saveCheckBox.Checked = value; }
        }

        /// <summary>
        /// Ctor
        /// </summary>
        public AnalyzerDialog() {
            InitializeComponent();
            Font = SystemFonts.DefaultFont;

            reportRichTextBox.Text = @"
To create a StateFile:

• Select the MegaMol™ Frontend with StateFile generation capability.
• Optionally specify the working directory.
• Specify the file name to store the new StateFile.

Then you can execute MegaMol™ to generate the StateFile using the button, or you can copy the command line argument and execute MegaMol™ yourself.
";
        }

        private void workDirTextBox_TextChanged(object sender, EventArgs e) {
            useWorkDirCheckBox.Checked = true;
        }

        private void browseMegaMolButton_Click(object sender, EventArgs e) {
            Util.ApplicationSearchDialog asd = new Util.ApplicationSearchDialog();
            asd.FileName = megaMolTextBox.Text;
            if (asd.ShowDialog(this) == DialogResult.OK) {
                megaMolTextBox.Text = asd.FileName;
            }
        }

        private void browseWorkDirButton_Click(object sender, EventArgs e) {
            folderBrowserDialog1.SelectedPath = workDirTextBox.Text;
            if (folderBrowserDialog1.ShowDialog() == DialogResult.OK) {
                workDirTextBox.Text = folderBrowserDialog1.SelectedPath;
                useWorkDirCheckBox.Checked = true;
            }
        }

        private void buttonOk_Click(object sender, EventArgs e) {
            if (Plugins != null) DialogResult = DialogResult.OK;
        }

        private void buttonExecute_Click(object sender, EventArgs e) {
            try {
                buttonExecute.Enabled = false;

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = megaMolTextBox.Text;
                if (string.IsNullOrWhiteSpace(psi.FileName)) throw new ArgumentNullException("MegaMol Frontend");
                if (!File.Exists(psi.FileName)) throw new ArgumentException("MegaMol Frontend not found");
                if (useWorkDirCheckBox.Checked) {
                    if (!string.IsNullOrWhiteSpace(workDirTextBox.Text) && !Directory.Exists(workDirTextBox.Text)) throw new ArgumentException("Working directory not found");
                    psi.WorkingDirectory = workDirTextBox.Text;
                }
                if (string.IsNullOrWhiteSpace(psi.WorkingDirectory)) {
                    psi.WorkingDirectory = Path.GetDirectoryName(psi.FileName);
                }
                psi.RedirectStandardOutput = true;
                psi.RedirectStandardError = true;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;

                string stateFileName = Path.Combine(System.IO.Path.GetTempPath(), Guid.NewGuid().ToString() + ".mmstate");

                psi.Arguments = "-i GenStateFile genState -v ::genState::gen::filename \"" + stateFileName + "\"";

                log(@"
Starting StateFile generation for MegaMol:
" + psi.FileName);

                megaMolConOut = new List<string>();
                megaMolConOutSync = new object();

                Process megaMol = Process.Start(psi);
                Thread stdoutReader = new Thread(readStdOut);
                stdoutReader.Start(megaMol.StandardOutput);
                Thread stderrReader = new Thread(readStdErr);
                stderrReader.Start(megaMol.StandardError);
                megaMol.WaitForExit();
                stdoutReader.Join();
                stderrReader.Join();

                foreach (string l in megaMolConOut) {
                    log(l);
                }

                log("MegaMol process completed");

                if (!File.Exists(stateFileName)) throw new Exception("State file not found");

                Io.PluginsStateFile psf = new Io.PluginsStateFile();
                log("Loading temporary state file...");
                try {
                    psf.Load(stateFileName);
                } catch {
                    File.Delete(stateFileName);
                    throw;
                }
                File.Delete(stateFileName);

                Plugins = new List<Data.PluginFile>(psf.Plugins);

                if (Plugins == null) {
                    log("StateFile did not contain any data");
                } else {
                    log("State contains " + Plugins.Count + " files:");
                    foreach (Data.PluginFile p in Plugins) {
                        log(string.Format("    {0} with {1} Modules and {2} Calls", p.Filename, p.Modules.Length, p.Calls.Length));
                    }
                }

            } catch (Exception ex) {
                log(@"
Failed to anaylse MegaMol™ and create StateFile:
" + ex.ToString());
            }

            buttonExecute.Enabled = true;
            buttonOk.Enabled = (Plugins != null);
        }

        private List<string> megaMolConOut;
        private object megaMolConOutSync;

        private void readStdOut(object stream) {
            StreamReader sr = (StreamReader)stream;
            try {
                while (!sr.EndOfStream) {
                    string line = sr.ReadLine();
                    lock (megaMolConOutSync) {
                        megaMolConOut.Add("  o | " + line);
                    }
                }
            } catch { }
        }

        private void readStdErr(object stream) {
            StreamReader sr = (StreamReader)stream;
            try {
                while (!sr.EndOfStream) {
                    string line = sr.ReadLine();
                    lock (megaMolConOutSync) {
                        megaMolConOut.Add("  e | " + line);
                    }
                }
            } catch { }
        }

        private void log(string line) {
            //if (InvokeRequired) {
            //    Invoke(new Action<string>(log), new object[] { line });
            //    return;
            //}
            if (!line.EndsWith("\n")) line += "\n";
            reportRichTextBox.AppendText(line);
            reportRichTextBox.SelectionStart = reportRichTextBox.TextLength;
            reportRichTextBox.ScrollToCaret();
        }

        private void button1_Click(object sender, EventArgs e) {
            reportRichTextBox.Clear();
        }

    }

}
