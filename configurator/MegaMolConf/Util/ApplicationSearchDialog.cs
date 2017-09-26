using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace MegaMolConf.Util {
    internal class ApplicationSearchDialog {

        private OpenFileDialog openFileDialog1;

        public string FileName {
            get { return openFileDialog1.FileName; }
            set { openFileDialog1.FileName = value; }
        }

        public ApplicationSearchDialog() {
            this.openFileDialog1 = new OpenFileDialog();
            this.openFileDialog1.DefaultExt = "exe";
            this.openFileDialog1.Filter = "Applications|*.exe;*.sh|All Files|*.*";
            this.openFileDialog1.Title = "Select MegaMol Frontend...";
        }

        public DialogResult ShowDialog() {
            return ShowDialog(null);
        }
        
        public DialogResult ShowDialog(IWin32Window owner) {
            try {
                if (System.IO.File.Exists(openFileDialog1.FileName))
                    openFileDialog1.InitialDirectory = System.IO.Path.GetDirectoryName(openFileDialog1.FileName);
                else
                    openFileDialog1.InitialDirectory = Environment.CurrentDirectory;
            } catch { }
            return openFileDialog1.ShowDialog(owner);

        }

    }
}
