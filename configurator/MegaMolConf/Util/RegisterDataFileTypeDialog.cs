using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace MegaMolConf.Util {

    public partial class RegisterDataFileTypeDialog : Form {

        public RegisterDataFileTypeDialog() {
            InitializeComponent();
            Font = SystemFonts.DefaultFont;
            Icon = Properties.Resources.MegaMol_Ctrl;

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

        public string FileExtension { get { return textBoxExt.Text; } set { textBoxExt.Text = value; } }
        public string FileDescription { get { return textBoxDescription.Text; } set { textBoxDescription.Text = value; } }
        public string FileIconPath { get { return textBoxIconPath.Text; } set { textBoxIconPath.Text = value; } }
        public string FileOpenCommand { get { return textBoxOpenCommand.Text; } set { textBoxOpenCommand.Text = value; } }

        private void registerFileType_Click(object sender, EventArgs e) {
            if (SG.Utilities.Forms.Elevation.IsElevationRequired && !SG.Utilities.Forms.Elevation.IsElevated) {

                // Superugly hack and potential dangerous because of the command length size limit.
                // Correct would be to use the pipes for communication, but I don't care right now.

                System.IO.MemoryStream mem = new System.IO.MemoryStream();

                byte[] blob = Encoding.UTF8.GetBytes(textBoxExt.Text);
                byte[] len = BitConverter.GetBytes(blob.Length);
                mem.Write(len, 0, len.Length);
                mem.Write(blob, 0, blob.Length);

                blob = Encoding.UTF8.GetBytes(textBoxDescription.Text);
                len = BitConverter.GetBytes(blob.Length);
                mem.Write(len, 0, len.Length);
                mem.Write(blob, 0, blob.Length);

                blob = Encoding.UTF8.GetBytes(textBoxIconPath.Text);
                len = BitConverter.GetBytes(blob.Length);
                mem.Write(len, 0, len.Length);
                mem.Write(blob, 0, blob.Length);

                blob = Encoding.UTF8.GetBytes(textBoxOpenCommand.Text);
                len = BitConverter.GetBytes(blob.Length);
                mem.Write(len, 0, len.Length);
                mem.Write(blob, 0, blob.Length);

                mem.Position = 0;
                string regData = Convert.ToBase64String(mem.ToArray());

                if (SG.Utilities.Forms.Elevation.RestartElevated("?#FILEREG " + regData + " " + this.Handle.ToString()) == int.MinValue) {
                    MessageBox.Show("Failed to start elevated Process. Most likely a security conflict.", Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            } else {
                Util.FileTypeRegistration.RegisterDataFileType(this, textBoxExt.Text, textBoxDescription.Text, textBoxIconPath.Text, textBoxOpenCommand.Text);
            }
        }

        private void unregisterFileType_Click(object sender, EventArgs e) {
            if (SG.Utilities.Forms.Elevation.IsElevationRequired && !SG.Utilities.Forms.Elevation.IsElevated) {
                if (SG.Utilities.Forms.Elevation.RestartElevated("?#FILEUNREG " + textBoxExt.Text + " " + this.Handle.ToString()) == int.MinValue) {
                    MessageBox.Show("Failed to start elevated Process. Most likely a security conflict.", Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            } else {
                Util.FileTypeRegistration.UnregisterDataFileType(this, textBoxExt.Text);
            }
        }

    }

}
