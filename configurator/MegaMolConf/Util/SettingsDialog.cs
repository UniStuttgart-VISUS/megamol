using System;
using System.Drawing;
using System.Windows.Forms;

namespace MegaMolConf.Util {

    /// <summary>
    /// Simple dialog for settings editing
    /// </summary>
    public partial class SettingsDialog : Form {

        /// <summary>
        /// Ctor
        /// </summary>
        public SettingsDialog() {
            InitializeComponent();
            Font = SystemFonts.DefaultFont;
            button3.Enabled = false;
            button4.Enabled = false;
            if (Environment.OSVersion.Platform == PlatformID.Win32NT) {
                // If is windows, then enable
                button3.Enabled = true;
                button4.Enabled = true;
                //  Then test for elevation requirement
                if (SG.Utilities.Forms.Elevation.IsElevationRequired && !SG.Utilities.Forms.Elevation.IsElevated) {
                    SG.Utilities.Forms.Elevation.ShowButtonShield(button3, true);
                    SG.Utilities.Forms.Elevation.ShowButtonShield(button4, true);
                }
            }
        }

        /// <summary>
        /// Gets or sets the settings to be edited
        /// </summary>
        internal Properties.Settings Settings {
            get { return propertyGrid1.SelectedObject as Properties.Settings; }
            set { propertyGrid1.SelectedObject = value; }
        }

        private void button3_Click(object sender, EventArgs e) {
            if (SG.Utilities.Forms.Elevation.IsElevationRequired && !SG.Utilities.Forms.Elevation.IsElevated) {
                if (SG.Utilities.Forms.Elevation.RestartElevated("?#REG " + Application.ExecutablePath + " " + Handle.ToString()) == int.MinValue) {
                    MessageBox.Show("Failed to start elevated Process. Most likely a security conflict.",
                        Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            } else {
                FileTypeRegistration.Register(this, Application.ExecutablePath);
            }
        }

        private void button4_Click(object sender, EventArgs e) {
            if (SG.Utilities.Forms.Elevation.IsElevationRequired && !SG.Utilities.Forms.Elevation.IsElevated) {
                if (SG.Utilities.Forms.Elevation.RestartElevated("?#UNREG " + Application.ExecutablePath + " " + Handle.ToString()) == int.MinValue) {
                    MessageBox.Show("Failed to start elevated Process. Most likely a security conflict.",
                        Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            } else {
                FileTypeRegistration.Unregister(this, Application.ExecutablePath);
            }
        }

        private void button1_Click(object sender, EventArgs e) {
            StartupCheckForm scform = new StartupCheckForm();
            scform.KeepOpen = true;
            scform.ShowDialog(this);
        }

        private void button2_Click(object sender, EventArgs e) {
            Settings.Save();
        }

    }

}
