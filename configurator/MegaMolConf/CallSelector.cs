using System;
using System.Drawing;
using System.Windows.Forms;

namespace MegaMolConf {
    public partial class CallSelector : Form {

        public string SelectedItem {
            get {
                return listBox1.SelectedItem as string;
            }
        }

        public int SelectedIndex {
            get {
                return listBox1.SelectedIndex;
            }
        }

        public CallSelector(string[] compatibles) {
            InitializeComponent();
            Font = SystemFonts.DefaultFont;
            listBox1.Items.AddRange(compatibles);
            if (listBox1.Items.Count > 0) {
                listBox1.SelectedIndex = 0;
            }
        }

        private void listBox1_DoubleClick(object sender, EventArgs e) {
            DialogResult = DialogResult.OK;
        }

    }
}
