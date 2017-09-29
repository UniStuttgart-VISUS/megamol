using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace MegaMolConf.Util {

    internal partial class StartupCheckForm : Form {

        /// <summary>
        /// Creates the startup checks to be performed. Startup checks require
        /// the main form to be created and visible.
        /// </summary>
        /// <returns>Array of startup checks</returns>
        private static StartupCheck[] MakeStartupCheckList() {
            List<StartupCheck> chks = new List<StartupCheck>();
            chks.Add(new StartupCheck() {
                    Title = "MegaMol Console",
                    Description = "Test if the path to the MegaMol Console front end binary is known.",
                    DoEvaluate = new Func<StartupCheck,bool>((StartupCheck sc) => {
                        sc.EvaluateComment = Properties.Settings.Default.MegaMolBin;
                        return !String.IsNullOrWhiteSpace(Properties.Settings.Default.MegaMolBin);
                    }),
                    DoFix = new Action<StartupCheck,IWin32Window>((StartupCheck sc, IWin32Window w) => {
                        Util.ApplicationSearchDialog asd = new Util.ApplicationSearchDialog();
                        asd.FileName = Properties.Settings.Default.MegaMolBin;
                        if (asd.ShowDialog(w) == DialogResult.OK) {
                            Properties.Settings.Default.MegaMolBin = asd.FileName;
                            sc.EvaluateComment = asd.FileName;
                            Properties.Settings.Default.Save();
                        }
                    })
                });
            chks.Add(new StartupCheck() {
                Title = "MegaMol Config File",
                Description = "There is a MegaMol Config file ready to load.",
                DoEvaluate = new Func<StartupCheck, bool>((StartupCheck sc) => {
                    List<string> paths = new List<string>();
                    string path = Environment.GetEnvironmentVariable("MEGAMOLCONFIG");
                    if (!string.IsNullOrWhiteSpace(path)) {
                        if (System.IO.Directory.Exists(path)) paths.Add(path);
                        else if (System.IO.File.Exists(path)) {
                            sc.EvaluateComment = path; // config file
                            return true;
                        }
                    }
                    path = Properties.Settings.Default.MegaMolBin;
                    if (!string.IsNullOrWhiteSpace(path)) {
                        path = System.IO.Path.GetDirectoryName(path);
                        if (System.IO.Directory.Exists(path)) paths.Add(path);
                    }
                    path = Environment.GetFolderPath(Environment.SpecialFolder.Personal);
                    if (System.IO.Directory.Exists(path)) {
                        paths.Add(path);
                        path = System.IO.Path.Combine(path, ".megamol");
                        if (System.IO.Directory.Exists(path)) paths.Add(path);
                    }
                    if (Properties.Settings.Default.UseApplicationDirectoryAsWorkingDirectory) {
                        path = Properties.Settings.Default.WorkingDirectory;
                        if (!string.IsNullOrWhiteSpace(path)) {
                            path = System.IO.Path.GetDirectoryName(path);
                            if (System.IO.Directory.Exists(path)) paths.Add(path);
                        }
                    }
                    string[] filenames = new string[]{ "megamolconfig.lua",
                        "megamolconfig.xml", "megamol.cfg", ".megamolconfig.xml", ".megamol.cfg"};
                    foreach (string p in paths) {
                        foreach (string f in filenames) {
                            string file = System.IO.Path.Combine(p, f);
                            if (System.IO.File.Exists(file)) {
                                sc.EvaluateComment = file;
                                return true;
                            }
                        }
                    }
                    return false;
                }),
                DoFix = new Action<StartupCheck, IWin32Window>((StartupCheck sc, IWin32Window w) => {
                    SaveFileDialog sfd = new SaveFileDialog();
                    sfd.DefaultExt = "cfg";
                    sfd.Filter = "Config Files|*.cfg|Xml Files|*.xml|All Files|*.*";
                    sfd.Title = "Write MegaMol Config File...";

                    string path = Environment.GetEnvironmentVariable("MEGAMOLCONFIG");
                    if (!string.IsNullOrWhiteSpace(path)) {
                        if (System.IO.Directory.Exists(path)) sfd.FileName = System.IO.Path.Combine(path, "megamol.cfg");
                        else if (System.IO.File.Exists(path)) sfd.FileName = path;
                    }
                    path = Properties.Settings.Default.MegaMolBin;
                    if (string.IsNullOrWhiteSpace(sfd.FileName) && !string.IsNullOrWhiteSpace(path)) {
                        sfd.FileName = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(path), "megamol.cfg");
                    }

                    try {
                        path = System.IO.Path.GetDirectoryName(sfd.FileName);
                        if (System.IO.Directory.Exists(path)) sfd.InitialDirectory = path;
                    } catch {}

                    if (sfd.ShowDialog() == DialogResult.OK) {
                        path = sfd.FileName;
                        if (System.IO.File.Exists(path)) {
                            if (MessageBox.Show("File \"" + path + "\" will be overwritten. Continue?", Application.ProductName, MessageBoxButtons.OKCancel, MessageBoxIcon.Exclamation) != DialogResult.OK) return;
                        }
                        path = System.IO.Path.GetFileName(sfd.FileName);
                        if (string.IsNullOrWhiteSpace(path)) {
                            MessageBox.Show("You must specify a file name", Application.ProductName);
                            return;
                        }
                        if (path.Equals("megamol", StringComparison.CurrentCultureIgnoreCase)
                                || path.Equals(".megamol", StringComparison.CurrentCultureIgnoreCase)) {
                            sfd.FileName = System.IO.Path.ChangeExtension(sfd.FileName, "cfg");
                            path = System.IO.Path.GetFileName(sfd.FileName);
                        } else if (path.Equals("megamolconfig", StringComparison.CurrentCultureIgnoreCase)
                                || path.Equals(".megamolconfig", StringComparison.CurrentCultureIgnoreCase)) {
                            sfd.FileName = System.IO.Path.ChangeExtension(sfd.FileName, "xml");
                            path = System.IO.Path.GetFileName(sfd.FileName);
                        }
                        if (!path.Equals("megamol.cfg", StringComparison.CurrentCultureIgnoreCase)
                                && !path.Equals(".megamol.cfg", StringComparison.CurrentCultureIgnoreCase)
                                && !path.Equals("megamolconfig.xml", StringComparison.CurrentCultureIgnoreCase)
                                && !path.Equals(".megamolconfig.xml", StringComparison.CurrentCultureIgnoreCase)
                                && !sfd.FileName.Equals(Environment.GetEnvironmentVariable("MEGAMOLCONFIG"))) {
                            if (MessageBox.Show("Non-standard MegaMol config file \"" + path + "\" might not work. Continue?", Application.ProductName, MessageBoxButtons.OKCancel, MessageBoxIcon.Exclamation) != DialogResult.OK) return;
                        }

                        try {
                            System.IO.File.WriteAllBytes(sfd.FileName, Properties.Resources.megamol);
                            sc.EvaluateComment = sfd.FileName;
                        } catch (Exception ex) {
                            sc.EvaluateComment = ex.ToString();
                        }
                    }

                })
            });
            chks.Add(new StartupCheck() {
                    Title = "Core and Plugins State",
                    Description = "A state file is loaded which contains all Modules and Calls in the Core and the known plugins",
                    DoEvaluate = new Func<StartupCheck,bool>((StartupCheck sc) => {
                        List<Data.PluginFile> plgs = Program.MainForm.Plugins;
                        int pc = 0, mc = 0, cc = 0;

                        if ((plgs!= null) && (plgs.Count > 0)) {
                            foreach (Data.PluginFile p in plgs) {
                                pc++;
                                if (p.Modules != null) mc += p.Modules.Length;
                                if (p.Calls != null) cc += p.Calls.Length;
                            }
                        }
                        sc.EvaluateComment = pc.ToString() + " Plugins with " + mc.ToString() + " Modules and " + cc.ToString() + " Calls";

                        return (pc + mc + cc) > 0;
                    }),
                    DoFix = new Action<StartupCheck,IWin32Window>((StartupCheck sc, IWin32Window w) => {

                        Analyze.AnalyzerDialog ad = new Analyze.AnalyzerDialog();

                        ad.MegaMolPath = Properties.Settings.Default.MegaMolBin;
                        ad.WorkingDirectory = Properties.Settings.Default.WorkingDirectory;

                        if (ad.ShowDialog(w) == System.Windows.Forms.DialogResult.OK) {

                            Properties.Settings.Default.MegaMolBin = ad.MegaMolPath;
                            Properties.Settings.Default.Save();

                            Program.MainForm.SetPlugins(ad.Plugins, ad.SaveAfterOk);
                        }
                    
                    })
                });
            if (Environment.OSVersion.Platform == PlatformID.Win32NT) {
                chks.Add(new StartupCheck() {
                    Title = "Register *.mmprj File Type",
                    Description = "The MMPRJ file type is registered.",
                    DoEvaluate = new Func<StartupCheck, bool>((StartupCheck sc) => {
                        string[] types = Microsoft.Win32.Registry.ClassesRoot.GetSubKeyNames();
                        sc.NeedsElevation = SG.Utilities.Forms.Elevation.IsElevationRequired && !SG.Utilities.Forms.Elevation.IsElevated;
                        foreach (string t in types) {
                            if (t.Equals(".mmprj", StringComparison.InvariantCultureIgnoreCase)) return true;
                        }
                        return false;
                    }),
                    DoFix = new Action<StartupCheck, IWin32Window>((StartupCheck sc, IWin32Window w) => {
                        if (SG.Utilities.Forms.Elevation.IsElevationRequired && !SG.Utilities.Forms.Elevation.IsElevated) {
                            if (SG.Utilities.Forms.Elevation.RestartElevated("?#REG " + Application.ExecutablePath + " " + w.Handle.ToString()) == int.MinValue) {
                                MessageBox.Show("Failed to start elevated Process. Most likely a security conflict.",
                                    Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
                            }
                        } else {
                            Util.FileTypeRegistration.Register(w, Application.ExecutablePath);
                        }
                    })
                });
            }
            return chks.ToArray();
        }

        private StartupCheck[] checks;
        private Action reevaluateAll;

        public StartupCheckForm() {
            InitializeComponent();
            Font = SystemFonts.DefaultFont;
            KeepOpen = false;
            Text = "MegaMol™ Configurator - Startup Checks";
            Icon = Properties.Resources.MegaMol_Ctrl;

            checks = MakeStartupCheckList();
            if ((checks != null) && (checks.Length > 0)) {
                tableLayoutPanel1.Visible = true;
                tableLayoutPanel1.RowCount = checks.Length * 4;

                int cc = checks.Length;
                for (int ci = 0; ci < cc; ++ci) {
                    StartupCheck check = checks[ci];

                    check.Value = check.DoEvaluate(check);

                    PictureBox icon = new PictureBox();
                    tableLayoutPanel1.Controls.Add(icon);
                    tableLayoutPanel1.SetCellPosition(icon, new TableLayoutPanelCellPosition(0, ci * 4 + 0));
                    tableLayoutPanel1.SetRowSpan(icon, 4);
                    icon.Image = check.Value
                        ? Properties.Resources.StatusAnnotations_Complete_and_ok_32xLG_color
                        : Properties.Resources.StatusAnnotations_Required_32xLG_color;
                    icon.SizeMode = PictureBoxSizeMode.CenterImage;
                    icon.Dock = DockStyle.Fill;

                    Label title = new Label();
                    title.Text = check.Title;
                    title.AutoSize = true;
                    title.Font = new Font(title.Font, FontStyle.Bold);
                    title.Padding = new Padding(0, 4, 0, 0);
                    tableLayoutPanel1.Controls.Add(title);
                    tableLayoutPanel1.SetCellPosition(title, new TableLayoutPanelCellPosition(1, ci * 4 + 0));

                    Label description = new Label();
                    description.Text = check.Description;
                    description.AutoSize = true;
                    description.MaximumSize = new Size((int)tableLayoutPanel1.GetColumnWidths()[1] - 8, 1000);
                    description.Padding = new Padding(0, 4, 0, 0);
                    tableLayoutPanel1.Controls.Add(description);
                    tableLayoutPanel1.SetCellPosition(description, new TableLayoutPanelCellPosition(1, ci * 4 + 1));

                    Label comment = new Label();
                    comment.Text = check.EvaluateComment;
                    comment.AutoSize = true;
                    comment.MaximumSize = new Size((int)tableLayoutPanel1.GetColumnWidths()[1] - 8, 1000);
                    comment.Padding = new Padding(0, 4, 0, 4);
                    tableLayoutPanel1.Controls.Add(comment);
                    tableLayoutPanel1.SetCellPosition(comment, new TableLayoutPanelCellPosition(1, ci * 4 + 2));

                    Button fix = new Button();
                    fix.Text = "Fix";
                    fix.UseVisualStyleBackColor = true;
                    tableLayoutPanel1.Controls.Add(fix);
                    tableLayoutPanel1.SetCellPosition(fix, new TableLayoutPanelCellPosition(1, ci * 4 + 3));
                    if (check.NeedsElevation) {
                        SG.Utilities.Forms.Elevation.ShowButtonShield(fix, true);
                    }

                    fix.Click += (object sender, EventArgs args) => {
                        check.DoFix(check, this);
                        reevaluateAll();
                    };
                    reevaluateAll += () => {
                        check.Value = check.DoEvaluate(check);
                        icon.Image = check.Value
                            ? Properties.Resources.StatusAnnotations_Complete_and_ok_32xLG_color
                            : Properties.Resources.StatusAnnotations_Required_32xLG_color;
                        comment.Text = check.EvaluateComment;
                    };

                    tableLayoutPanel1.Resize += (object sender, EventArgs args) => {
                        int w = (int)tableLayoutPanel1.GetColumnWidths()[1] - 8;
                        description.MaximumSize = new Size(w, 1000);
                        comment.MaximumSize = new Size(w, 1000);
                    };
                }

            } else {
                tableLayoutPanel1.Visible = false;
            }
        }

        private void StartupCheckForm_Shown(object sender, EventArgs e) {
            foreach (StartupCheck check in checks) {
                if (!check.Value) KeepOpen = true;
            }
            if (!KeepOpen) {
                // If all checks succeed and KeepOpen == false:
                DialogResult = System.Windows.Forms.DialogResult.Ignore;
                Close();
            }
        }

        public bool KeepOpen { get; set; }
    }

}
