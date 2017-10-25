using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace MegaMolConf {
    public partial class ImportParamfileForm : Form {
        internal List<GraphicalModule> Modules { get; set; }
        internal Io.Paramfile ParamFile { get; set; }
        internal string ProjectName { get; set; }

        public ImportParamfileForm() {
            InitializeComponent();
            Font = SystemFonts.DefaultFont;
            Icon = Properties.Resources.MegaMol_Ctrl;
        }

        private void ImportParamfileForm_Shown(object sender, EventArgs e) {
            listView1.BeginUpdate();
            listView1.Items.Clear();
            ListViewItem root = listView1.Items.Add(ProjectName);
            List<ListViewItem> rc = new List<ListViewItem>();
            root.Tag = rc;
            foreach (GraphicalModule m in Modules) {
                ListViewItem mod = listView1.Items.Add("    " + m.Name);
                List<ListViewItem> c = new List<ListViewItem>();
                mod.Tag = c;
                mod.SubItems[0].Tag = m;
                foreach (var p in m.ParameterValues) {
                    ListViewItem lvi = listView1.Items.Add("        " + p.Key.Name);
                    lvi.SubItems[0].Tag = p;
                    c.Add(lvi);
                    rc.Add(lvi);
                    lvi.Name = m.Name + "::" + p.Key.Name;
                    lvi.SubItems.Add(p.Value);
                    lvi.Tag = p.Key;
                }
            }

            int cnt = ParamFile.Count;
            for (int i = 0; i < cnt; ++i) {
                var p = ParamFile[i];
                string fn = string.Join("::", p.Key);
                string sn = string.Join("::", p.Key, 1, p.Key.Length - 1);
                ListViewItem lvi = null;
                foreach (ListViewItem l in listView1.Items) {
                    if ((l.Name == fn) || (l.Name == sn)) {
                        lvi = l;
                    }
                }
                if (lvi == null) continue;
                if (lvi.SubItems.Count < 3) {
                    lvi.SubItems.Add(
                        ((Data.ParamSlot)lvi.Tag).Type.ValuesEqual(lvi.SubItems[1].Text, p.Value)
                        ? "" : "<<<");
                    lvi.SubItems.Add(p.Value);
                }
            }
            listView1.EndUpdate();
        }

        private void okButton_Click(object sender, EventArgs e) {
            GraphicalModule gm = null;
            foreach (ListViewItem lvi in listView1.Items) {
                Data.ParamSlot p = lvi.Tag as Data.ParamSlot;
                if (p == null) {
                    GraphicalModule m = lvi.SubItems[0].Tag as GraphicalModule;
                    if (m != null) gm = m;
                    continue;
                }
                if (lvi.SubItems.Count < 3) continue;
                if (lvi.SubItems[2].Text != "<<<") continue;

                System.Diagnostics.Debug.Assert(gm != null);
                System.Diagnostics.Debug.Assert(gm.ParameterValues.ContainsKey(p));

                if (!p.Type.ValuesEqual(gm.ParameterValues[p], lvi.SubItems[3].Text)) {
                    gm.ParameterValues[p] = lvi.SubItems[3].Text;
                }
            }

            DialogResult = System.Windows.Forms.DialogResult.OK;
        }

        private void button3_Click(object sender, EventArgs e) {
            listView1.SelectedItems.Clear();
        }

        private void button2_Click(object sender, EventArgs e) {
            listView1.SelectedItems.Clear();
            foreach (ListViewItem lvi in listView1.Items) {
                lvi.Selected = ((lvi.Tag as Data.ParamSlot) != null)
                    && (lvi.SubItems.Count > 3)
                    && !string.IsNullOrWhiteSpace(lvi.SubItems[3].ToString());
            }
        }

        private void button5_Click(object sender, EventArgs e) {
            foreach (ListViewItem lvi in allSelected()) {
                if (lvi.SubItems.Count > 3) {
                    lvi.SubItems[2].Text = "";
                }
            }
        }

        private void button4_Click(object sender, EventArgs e) {
            foreach (ListViewItem lvi in allSelected()) {
                if (lvi.SubItems.Count > 3) {
                    lvi.SubItems[2].Text = "<<<";
                }
            }
        }

        private HashSet<ListViewItem> allSelected() {
            HashSet<ListViewItem> allItems = new HashSet<ListViewItem>();
            foreach (ListViewItem lvi in listView1.SelectedItems) {
                List<ListViewItem> c = lvi.Tag as List<ListViewItem>;
                if (c != null) {
                    foreach (ListViewItem clvi in c) {
                        if (clvi.Tag is Data.ParamSlot) allItems.Add(clvi);
                    }
                }
                if (lvi.Tag is Data.ParamSlot) allItems.Add(lvi);
            }
            return allItems;
        }

    }
}
