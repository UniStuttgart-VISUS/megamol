using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using System.Xml.Serialization;
using MegaMolConf.Communication;
using System.Threading;
using System.Reflection;
using System.Net;
using System.Net.Sockets;
using System.Net.NetworkInformation;

namespace MegaMolConf {
    public partial class Form1 : Form     {
        private List<Data.PluginFile> plugins = null;
        private Dictionary<TabPage, List<GraphicalModule>> tabModules = new Dictionary<TabPage, List<GraphicalModule>>();
        private Dictionary<TabPage, List<GraphicalConnection>> tabConnections = new Dictionary<TabPage, List<GraphicalConnection>>();
        private Dictionary<TabPage, GraphicalModule> tabMainViews = new Dictionary<TabPage, GraphicalModule>();
        private Dictionary<TabPage, object> tabSelectedObjects = new Dictionary<TabPage, object>();
        private Dictionary<TabPage, string> tabStartParameters = new Dictionary<TabPage, string>();
        private int tabCount = 0;
        private GraphicalModule movedModule = null;
        private Point lastMousePos;
        private Point mouseDownPos;
        private Point connectingTip = new Point();
        private bool drawConnection = false;
        private Rectangle drawArea = new Rectangle();
        private bool saveShortcut = false;

        private static int minPort = 30000;
        private static int maxPort = 31000;
        private bool[] occupiedPorts = new bool[maxPort - minPort + 1];
        bool isEyedropping = false;
        private Cursor eyeDropperCursor;
        internal Util.ListBoxLog listBoxLog;
        

        #region Selection management

        internal static Data.CalleeSlot selectedCallee { get; private set; }
        internal static Data.CallerSlot selectedCaller { get; private set; }
        internal static GraphicalModule selectedModule { get; private set; }
        internal static GraphicalModule copiedModule { get; private set; }
        internal static GraphicalModule eyedropperTarget { get; private set; }
        internal static GraphicalConnection selectedConnection { get; private set; }
        internal static TabPage selectedTab { get; private set; }

        private static void SelectItem(GraphicalModule m, Data.CalleeSlot s) {
            selectedCallee = s;
            selectedCaller = null;
            selectedModule = m;
            selectedConnection = null;
        }

        private static void SelectItem(GraphicalModule m, Data.CallerSlot s) {
            selectedCallee = null;
            selectedCaller = s;
            selectedModule = m;
            selectedConnection = null;
        }

        private static void SelectItem(GraphicalModule m) {
            selectedCallee = null;
            selectedCaller = null;
            selectedModule = m;
            selectedConnection = null;
        }

        private static void SelectItem(GraphicalConnection c) {
            selectedCallee = null;
            selectedCaller = null;
            selectedModule = null;
            selectedConnection = c;
        }

        #endregion

#if false
        // http://stackoverflow.com/questions/516494/create-a-semi-transparent-cursor-from-an-image
        private struct IconInfo {
            public bool fIcon;
            public int xHotspot;
            public int yHotspot;
            public IntPtr hbmMask;
            public IntPtr hbmColor;
        }
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetIconInfo(IntPtr hIcon, ref IconInfo pIconInfo);

        [DllImport("user32.dll")]
        private static extern IntPtr CreateIconIndirect(ref IconInfo icon);
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool DestroyIcon(IntPtr hIcon);
        private List<IntPtr> iconsToDestroy = new List<IntPtr>();

        private Cursor CreateCursor(Bitmap bmp, int xHotSpot, int yHotSpot) {
            IntPtr ptr = Properties.Resources.eyedropper.GetHicon();
            IconInfo tmp = new IconInfo();
            GetIconInfo(ptr, ref tmp);
            tmp.xHotspot = xHotSpot;
            tmp.yHotspot = yHotSpot;
            tmp.fIcon = false;
            ptr = CreateIconIndirect(ref tmp);
            Cursor cur = new Cursor(ptr);
            iconsToDestroy.Add(ptr);
            return cur;
        }
#endif

        public Form1() {
            InitializeComponent();
            Font = SystemFonts.DefaultFont;
            this.toolStripStatusLabel1.Text = string.Empty;
            this.Icon = Properties.Resources.MegaMol_Ctrl;
            this.setStateInfo();
            this.btnLoad_Click(null, null);
            this.btnNewProject_Click(null, null);
            this.saveStateToolStripMenuItem.Enabled = (this.plugins != null);

            //eyeDropperCursor = CreateCursor(new Bitmap(Properties.Resources.eyedropper), 1, 1);

            MemoryStream ms = new MemoryStream(Properties.Resources.eyedroppercur);
            eyeDropperCursor = new Cursor(ms);

            //this.StartPosition = FormStartPosition.Manual;
            //this.Bounds = new Rectangle(Point.Empty, new Size(1908, 635));
            //this.splitContainer1.SplitterDistance = 290;
            listBoxLog = new Util.ListBoxLog(listBox1);

            this.ScanPorts();
        }

        internal void ScanPorts() {
            occupiedPorts.Initialize();

            IPGlobalProperties ipg
                = IPGlobalProperties.GetIPGlobalProperties();
            TcpConnectionInformation[] localConns
                = ipg.GetActiveTcpConnections();
            IPEndPoint[] remoteListeners = ipg.GetActiveTcpListeners();

            foreach (var info in localConns) {
                if (info.LocalEndPoint.Port >= minPort
                    && info.LocalEndPoint.Port <= maxPort) {
                    occupiedPorts[info.LocalEndPoint.Port - minPort] = true;
                }
            }
            foreach (var info in remoteListeners) {
                if (info.Port >= minPort && info.Port <= maxPort) {
                    occupiedPorts[info.Port - minPort] = true;
                }
            }
        }

        internal int ReservePort() {
            for (int i = 0; i < maxPort - minPort + 1; i++) {
                if (!occupiedPorts[i]) {
                    occupiedPorts[i] = true;
                    return i + minPort;
                }
            }
            return 0;
        }

        internal void FreePort(int p) {
            if (p >= minPort && p <= maxPort) {
                int port = p - minPort;
                occupiedPorts[port] = false;
            }
        }

        internal void ParamChangeDetected() {
            if (this.InvokeRequired) {
                this.Invoke(new Action(() => {
                    this.ParamChangeDetected();
                }));
            } else {
                this.propertyGrid1.Refresh();
            }
        }

        internal void SetTabPageIcon(TabPage tp, int iconIdx) {
            if (this.InvokeRequired) {
                this.Invoke(new Action(() => {
                    this.SetTabPageIcon(tp, iconIdx);
                }));
            } else {
                if (tp.ImageIndex != iconIdx) {
                    tp.ImageIndex = iconIdx;
                }
            }
        }

        internal void SetTabPageTag(TabPage tp, object tag) {
            if (this.InvokeRequired) {
                this.Invoke(new Action(() => {
                    this.SetTabPageTag(tp, tag);
                }));
            } else {
                if (tp.Tag != tag) {
                    tp.Tag = tag;
                }
            }
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData) {
            saveShortcut = false;
            GraphicalModule m = null;
            switch (keyData) {
                case (Keys.Control | Keys.S):
                    saveShortcut = true;
                    // falls through!
                    goto case Keys.Control | Keys.Shift | Keys.S;
                case (Keys.Control | Keys.Shift | Keys.S):
                    btnSaveProject.PerformClick();
                    break;
                case (Keys.Control | Keys.O):
                    btnLoadProject.PerformClick();
                    break;
                case (Keys.Control | Keys.D):
                    m = this.duplicateSelectedModule();
                    selectedModule = m;
                    this.propertyGrid1.SelectedObject = new GraphicalModuleDescriptor(m);
                    this.refreshCurrent();
                    resizePanel();
                    break;
                case (Keys.Control | Keys.C):
                    this.btnCopy.PerformClick();
                    break;
                case (Keys.Control | Keys.V):
                    this.btnPaste.PerformClick();
                    break;
                case (Keys.Control | Keys.P):
                    this.btnEyeDrop.PerformClick();
                    break;
                case (Keys.Escape):
                    if (isEyedropping) {
                        isEyedropping = false;
                        tabViews.Cursor = Cursors.Default;
                    }
                    break;
            } 
            return base.ProcessCmdKey(ref msg, keyData);
        }

        private GraphicalModule duplicateSelectedModule() {
            GraphicalModule src = selectedModule;
            GraphicalModule dst = pasteModule(tabViews.SelectedTab, src);
            return dst;
        }

        private void eyedropParameters(GraphicalModule destMod, GraphicalModule srcMod) {
            if (srcMod != null && destMod != null && srcMod != destMod) {
                foreach (var p in srcMod.ParameterValues) {
                    if (destMod.ParameterValues.ContainsKey(p.Key)) {
                        destMod.ParameterValues[p.Key] = p.Value;
                    }
                }
            }
        }

        private GraphicalModule pasteModule(TabPage destPage, GraphicalModule srcMod) {
            if (destPage != null && srcMod != null) {
                tabModules[destPage].Add(new GraphicalModule(srcMod.Module, tabModules[destPage]));
                GraphicalModule dst = tabModules[destPage].Last();
                dst.Position = new Point(srcMod.Position.X + 10, srcMod.Position.Y + 10);
                eyedropParameters(dst, srcMod);
                return dst;
            }
            return null;
        }

        private void refreshCurrent() {
            if (tabViews.SelectedTab != null) {
                tabViews.SelectedTab.Controls[0].Refresh();
            }
        }

        private void updateFiltered() {
            List<Data.Module> mods = new List<Data.Module>();
            if (plugins != null) {
                foreach (Data.PluginFile pf in plugins) {
                    foreach (Data.Module m in pf.Modules) {
                        if (!string.IsNullOrWhiteSpace(moduleFilterBox.Text)) {
                            if (Application.CurrentCulture.CompareInfo.IndexOf(m.Name, moduleFilterBox.Text, CompareOptions.IgnoreCase) == -1) {
                                continue;
                            }
                        }
                        if (chkCompatible.Checked) {
                            if (selectedCallee != null) {
                                // we need a caller slot
                                if (m.CallerSlots != null) {
                                    foreach (Data.CallerSlot c in m.CallerSlots) {
                                        if (c.CompatibleCalls.Intersect(selectedCallee.CompatibleCalls).Count() > 0) {
                                            mods.Add(m);
                                            break;
                                        }
                                    }
                                }
                            } else if (selectedCaller != null) {
                                // we need a callee slot
                                if (m.CalleeSlots != null) {
                                    foreach (Data.CalleeSlot c in m.CalleeSlots) {
                                        if (c.CompatibleCalls.Intersect(selectedCaller.CompatibleCalls).Count() > 0) {
                                            mods.Add(m);
                                            break;
                                        }
                                    }
                                }
                            } else {
                                mods.Add(m);
                            }
                        } else {
                            mods.Add(m);
                        }
                    }
                }
            }
            mods = mods.OrderBy(x => x.ToString()).ToList();
            lbModules.DataSource = mods;
        }

        public List<Data.PluginFile> Plugins {
            get { return this.plugins; }
        }

        public void SetPlugins(List<Data.PluginFile> plugs, bool save) {
            this.plugins = plugs;
            this.setStateInfo();
            if (save) {
                btnSave_Click(null, null);
            }
            this.saveStateToolStripMenuItem.Enabled = (this.plugins != null);
            updateFiltered();
        }

        /// <summary>
        /// Open the dialog to analyze MegaMol
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void btnAnalyze_Click(object sender, EventArgs e) {
            Analyze.AnalyzerDialog ad = new Analyze.AnalyzerDialog();

            ad.MegaMolPath = Properties.Settings.Default.MegaMolBin;
            ad.MegaMolPath = EnsureMegaMolFrontendApplication(ad.MegaMolPath);
            ad.WorkingDirectory = Properties.Settings.Default.WorkingDirectory;

            if (ad.ShowDialog() == System.Windows.Forms.DialogResult.OK) {

                Properties.Settings.Default.MegaMolBin = ad.MegaMolPath;
                Properties.Settings.Default.Save();

                SetPlugins(ad.Plugins, ad.SaveAfterOk);
            }
        }

        /// <summary>
        /// Saves a MegaMolConf state file
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void btnSave_Click(object sender, EventArgs e) {
            SaveFileDialog sfd = new SaveFileDialog();

            string defFileName = "MegaMolConf.state";
            foreach (Data.PluginFile p in this.plugins) {
                if (p.IsCore) {
                    string name = Path.GetFileNameWithoutExtension(p.Filename);
                    if (name.StartsWith("MegaMol")) name = name.Substring(7);

                    string[] seg = Path.GetDirectoryName(p.Filename).Split(new char[] { '\\', '/' });
                    int seg_len = seg.Length;
                    for (int i = seg_len - 1; i >= 0; --i) {
                        if (seg[i].Equals("bin", StringComparison.CurrentCultureIgnoreCase)) {
                            defFileName = "MegaMolConf." + name;
                            for (int j = i + 1; j < seg_len; ++j) defFileName += "." + seg[j];
                            defFileName += ".state";
                        }
                    }

                    //defFileName = string.Format("MegaMolConf.{0}.state", name);
                }
            }

            sfd.FileName = Properties.Settings.Default.MegaMolStateFile;
            try {
                if (String.IsNullOrWhiteSpace(sfd.FileName)) {
                    sfd.FileName = defFileName;
                } else if (!Path.GetFileName(sfd.FileName).Equals(defFileName)) {
                    sfd.FileName = Path.Combine(Path.GetDirectoryName(sfd.FileName), defFileName);
                }
            } catch {
                sfd.FileName = defFileName;
            }

            try {
                sfd.InitialDirectory = Path.GetDirectoryName(sfd.FileName);
            } catch {
            }
            sfd.Title = "Save MegaMolConf State file ...";
            sfd.Filter = "MegaMolConf State Files|*.state|All Files|*.*";
            sfd.CheckPathExists = true;

            if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK) {
                Io.PluginsStateFile stateFile = new Io.PluginsStateFile();
                stateFile.Version = Io.PluginsStateFile.MaxVersion;
                stateFile.Plugins = plugins.ToArray();
                stateFile.Save(sfd.FileName);
                Properties.Settings.Default.MegaMolStateFile = sfd.FileName;
                Properties.Settings.Default.Save();
            }
        }

        /// <summary>
        /// Loads a MegaMolConf state file
        /// </summary>
        /// <param name="sender">If null, the method attemps to load the default state file specified in the application settings</param>
        /// <param name="e">not used</param>
        private void btnLoad_Click(object sender, EventArgs e) {
            string filename = null;
            if (sender == null) {
                filename = Properties.Settings.Default.MegaMolStateFile;
                if (!File.Exists(filename)) {
                    filename = "MegaMolConf.state";
                }
            } else {
                OpenFileDialog ofd = new OpenFileDialog();
                ofd.FileName = Properties.Settings.Default.MegaMolStateFile;
                try {
                    ofd.InitialDirectory = Path.GetDirectoryName(ofd.FileName);
                } catch {
                }
                ofd.Title = "Open MegaMolConf State file ...";
                ofd.Filter = "MegaMolConf State Files|*.state|All Files|*.*";
                ofd.CheckFileExists = true;
                ofd.CheckPathExists = true;
                if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK) {
                    filename = ofd.FileName;
                }
            }

            if (String.IsNullOrWhiteSpace(filename) || !File.Exists(filename)) {
                // all is lost
                return;
            }

            try {
                Io.PluginsStateFile stateFile = new Io.PluginsStateFile();
                stateFile.Load(filename);
                plugins = (stateFile.Plugins != null) ? new List<Data.PluginFile>(stateFile.Plugins) : null;
                Properties.Settings.Default.MegaMolStateFile = filename;
                Properties.Settings.Default.Save();
                this.setStateInfo();
                saveStateToolStripMenuItem.Enabled = (this.plugins != null);
            } catch (System.IO.FileNotFoundException) {
                // intentionally empty
            } finally {
            }

            updateFiltered();
        }

        /// <summary>
        /// Updates GUI with further information about the state
        /// </summary>
        private void setStateInfo() {
            if ((this.plugins == null) || (this.plugins.Count <= 0)) {
                this.MenuItem_StateInfo.Text = "No MegaMolConf State loaded";
                this.MenuItem_StateInfo.Enabled = false;
            } else {
                Data.PluginFile core = null;
                int mc = 0;
                int cc = 0;
                foreach (Data.PluginFile p in this.plugins) {
                    if (p.IsCore) core = p;
                    mc += p.Modules.Length;
                    cc += p.Calls.Length;
                }
                if (core != null) {
                    this.MenuItem_StateInfo.Text = String.Format("{0} and {1} Plugins: {2} Modules; {3} Calls",
                        new object[] { core.Filename, this.plugins.Count - 1, mc, cc });
                } else {
                    this.MenuItem_StateInfo.Text = String.Format("{0} Plugins: {1} Modules; {2} Calls",
                        new object[] { this.plugins.Count, mc, cc });
                }
                this.MenuItem_StateInfo.Enabled = true;
            }
        }

        internal static bool isMainView(GraphicalModule mod) {
            try {
                return ((Form1)Application.OpenForms[0]).tabMainViews.ContainsValue(mod);
            } catch {
            }
            return false;
        }

        internal static void setMainView(GraphicalModule mod) {
            try {
                Form1 that = ((Form1)Application.OpenForms[0]);
                foreach (TabPage tp in that.tabModules.Keys) {
                    if (that.tabModules[tp].Contains(mod)) {
                        that.tabMainViews[tp] = mod;
                        tp.Refresh();
                        return;
                    }
                }
            } catch {
            }
        }

        internal static void removeMainView(GraphicalModule mod) {
            try {
                Form1 that = ((Form1)Application.OpenForms[0]);
                foreach (TabPage tp in that.tabModules.Keys) {
                    if (that.tabMainViews[tp] == mod) {
                        foreach (GraphicalModule m in that.tabModules[tp]) {
                            if (m == mod) continue;
                            if (!m.Module.IsViewModule) continue;
                            that.tabMainViews[tp] = m;
                            tp.Refresh();
                            return;
                        }
                        return;
                    }
                }
            } catch {
            }
        }

        private void lbModules_DoubleClick(object sender, EventArgs e) {
            
            if (lbModules.SelectedItem != null) {
                TabPage tp = tabViews.SelectedTab;
                if (tp != null) {
                    tabModules[tp].Add(new GraphicalModule((Data.Module)lbModules.SelectedItem, tabModules[tp]));
                    if ((tabMainViews[tp] == null) && ((Data.Module)lbModules.SelectedItem).IsViewModule) {
                        tabMainViews[tp] = tabModules[tp].Last();
                    }
                    tabModules[tp].Last().Position = new Point(
                        //-(tp.Controls[0] as Panel).AutoScrollPosition.X + (tp.Controls[0] as Panel).Width / 2,
                        //-(tp.Controls[0] as Panel).AutoScrollPosition.Y + (tp.Controls[0] as Panel).Height / 2
                        -(tp.Controls[0] as Panel).AutoScrollPosition.X + 10,
                        -(tp.Controls[0] as Panel).AutoScrollPosition.Y + 10
                        );


                    if (selectedCaller != null) {
                        tabModules[tp].Last().Position = new Point(
                            selectedModule.Position.X + selectedModule.Bounds.Width * 2,
                            selectedModule.Position.Y
                        );
                        foreach (Data.CalleeSlot ce in tabModules[tp].Last().Module.CalleeSlots) {
                            string[] compatibles = selectedCaller.CompatibleCalls.Intersect(ce.CompatibleCalls).ToArray();
                            if (compatibles.Count() == 0) continue;
                            string theName = compatibles[0];
                            if (compatibles.Count() > 1) {
                                using (CallSelector cs = new CallSelector(compatibles)) {
                                    cs.StartPosition = FormStartPosition.Manual;
                                    Point p = this.PointToScreen(tabModules[tp].Last().Position);
                                    cs.Location = p;
                                    if (cs.ShowDialog(this) == System.Windows.Forms.DialogResult.Cancel) {
                                        SelectItem((GraphicalConnection)null);
                                        refreshCurrent();
                                        return;
                                    }
                                    theName = cs.SelectedItem;
                                }
                            }
                            GraphicalConnection gc = new GraphicalConnection(selectedModule, tabModules[tp].Last(), selectedCaller, ce, FindCallByName(theName));
                            // does the callerslot already have a connection? erase it
                            foreach (GraphicalConnection gtemp in tabConnections[tp]) {
                                if (gtemp.src.Equals(selectedModule) && selectedCaller.Equals(gtemp.srcSlot)) {
                                    tabConnections[tp].Remove(gtemp);
                                    break;
                                }
                            }
                            tabConnections[tp].Add(gc);
                            break;
                        }
                    }
                    if (selectedCallee != null) {
                        tabModules[tp].Last().Position = new Point(
                            selectedModule.Position.X - selectedModule.Bounds.Width * 2,
                            selectedModule.Position.Y
                        );
                        foreach (Data.CallerSlot cr in tabModules[tp].Last().Module.CallerSlots) {
                            string[] compatibles = selectedCallee.CompatibleCalls.Intersect(cr.CompatibleCalls).ToArray();
                            if (compatibles.Count() == 0) continue;
                            string theName = compatibles[0];
                            if (compatibles.Count() > 1) {
                                using (CallSelector cs = new CallSelector(compatibles)) {
                                    cs.StartPosition = FormStartPosition.Manual;
                                    Point p = this.PointToScreen(tabModules[tp].Last().Position);
                                    cs.Location = p;
                                    if (cs.ShowDialog(this) == System.Windows.Forms.DialogResult.Cancel) {
                                        SelectItem((GraphicalConnection)null);
                                        refreshCurrent();
                                        return;
                                    }
                                    theName = cs.SelectedItem;
                                }
                            }
                            GraphicalConnection gc = new GraphicalConnection(tabModules[tp].Last(), selectedModule, cr, selectedCallee, FindCallByName(theName));
                            tabConnections[tp].Add(gc);
                            break;
                        }
                    }

                    SelectItem(tabModules[tp].Last());
                    this.propertyGrid1.SelectedObject = new GraphicalModuleDescriptor(selectedModule);
                }
            }
            resizePanel();
            updateFiltered();
            refreshCurrent();
        }

        void PaintTabpage(object sender, PaintEventArgs e) {
            NoflickerPanel np = sender as NoflickerPanel;
            if (np != null) {
                TabPage tp = np.Tag as TabPage;
                if (tp != null) {
                    e.Graphics.ResetTransform();
                    //e.Graphics.TranslateTransform(-drawArea.Left + tp.HorizontalScroll.Value, -drawArea.Top + tp.VerticalScroll.Value);
                    e.Graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                    foreach (GraphicalModule gm in tabModules[tp]) {
                        gm.Draw(e.Graphics);
                    }
                    foreach (GraphicalConnection gc in tabConnections[tp]) {
                        gc.Draw(e.Graphics);
                    }
                    if (drawConnection && (selectedCallee != null || selectedCaller != null)) {
                        Point p = tp.Controls[0].Controls[0].PointToClient(Cursor.Position);
                        e.Graphics.DrawLine(Pens.Black, connectingTip, p);
                    }
                }
            }
        }

        private void tabViews_MouseDown(object sender, MouseEventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            Data.CallerSlot cr;
            Data.CalleeSlot ce;
            bool somethingSelected = false;
            mouseDownPos = e.Location;
            if (tp != null) {
                if (isEyedropping) {
                    for (int i = tabModules[tp].Count - 1; i >= 0; i--) {
                        GraphicalModule gm = tabModules[tp][i];
                        if (gm.IsHit(e.Location)) {
                            eyedropParameters(eyedropperTarget, gm);
                            this.ParamChangeDetected();
                            break;
                        }
                    }
                    isEyedropping = false;
                    tabViews.Cursor = Cursors.Default;
                }
                else {
                    for (int i = tabModules[tp].Count - 1; i >= 0; i--) {
                        GraphicalModule gm = tabModules[tp][i];
                        if (gm.IsSlotHit(e.Location, out ce, out cr, out connectingTip)) {
                            if (ce != null) {
                                SelectItem(gm, ce);
                            }
                            else {
                                SelectItem(gm, cr);
                            }
                            somethingSelected = true;
                            if (selectedCallee != null) {
                                toolStripStatusLabel1.Text = selectedCallee.Name + ": " + selectedCallee.Description;
                            }
                            else if (selectedCaller != null) {
                                toolStripStatusLabel1.Text = selectedCaller.Name + ": " + selectedCaller.Description;
                            }
                            else {
                                somethingSelected = false;
                            }
                            propertyGrid1.SelectedObject = new GraphicalModuleDescriptor(gm);
                            break;
                        }
                        if (gm.IsHit(e.Location)) {
                            SelectItem(gm);
                            movedModule = gm;
                            toolStripStatusLabel1.Text = gm.Module.Name + ": " + gm.Module.Description;
                            lastMousePos = e.Location;
                            propertyGrid1.SelectedObject = new GraphicalModuleDescriptor(gm);
                            somethingSelected = true;
                            break;
                        }
                    }

                    if (!somethingSelected) {
                        propertyGrid1.SelectedObject = new TabPageDescriptor(tp);
                    }

                    foreach (GraphicalConnection gc in tabConnections[tp]) {
                        if (gc.IsHit(e.Location)) {
                            SelectItem(gc);
                            somethingSelected = true;
                            toolStripStatusLabel1.Text = gc.Call.Name + ": " + gc.Call.Description;
                            break;
                        }
                    }

                    if (!somethingSelected) {
                        SelectItem((GraphicalModule)null);
                    }
                    updateFiltered();
                }
                refreshCurrent();
            }
        }

        private void resizePanel(bool allowShrinking = false) {
            TabPage tp = tabViews.SelectedTab;
            if (tp != null) {
                int minX = int.MaxValue;
                int maxX = int.MinValue;
                int minY = int.MaxValue;
                int maxY = int.MinValue;
                foreach (GraphicalModule gm in tabModules[tp]) {
                    if (gm.Position.X < minX) {
                        minX = gm.Position.X;
                    }
                    if (gm.Position.X + gm.Bounds.Width > maxX) {
                        maxX = gm.Position.X + gm.Bounds.Width;
                    }
                    if (gm.Position.Y < minY) {
                        minY = gm.Position.Y;
                    }
                    if (gm.Position.Y + gm.Bounds.Height > maxY) {
                        maxY = gm.Position.Y + gm.Bounds.Height;
                    }
                }
                foreach (GraphicalModule gm in tabModules[tp]) {
                    gm.Position = new Point(gm.Position.X + (minX < 0 ? -minX : 0), gm.Position.Y + (minY < 0 ? -minY : 0));
                }
                if (minX < 0)
                    maxX -= minX;
                if (minY < 0)
                    maxY -= minY;
                //this.drawArea = new Rectangle(minX, minY, maxX - minX, maxY - minY);
                this.drawArea = new Rectangle(0, 0, maxX, maxY);
                Panel p = tp.Controls[0] as Panel;
                if (allowShrinking) {
                    p.Controls[0].Width = Math.Max(p.HorizontalScroll.Value + p.Width, this.drawArea.Width);
                    p.Controls[0].Height = Math.Max(p.VerticalScroll.Value + p.Height, this.drawArea.Height);
                } else {
                    p.Controls[0].Width = Math.Max(p.Controls[0].Width, this.drawArea.Width);
                    p.Controls[0].Height = Math.Max(p.Controls[0].Height, this.drawArea.Height);
                }
            }
        }

        private void tabViews_MouseMove(object sender, MouseEventArgs e) {
            if (movedModule != null) {
                movedModule.Position = new Point(
                    Math.Max(0, movedModule.Position.X + e.Location.X - lastMousePos.X),
                    Math.Max(0, movedModule.Position.Y + e.Location.Y - lastMousePos.Y));
                resizePanel();
                if (tabViews.SelectedTab != null) {
                    doTheScrollingShit(e.Location);
                }
                refreshCurrent();
            } else if (selectedCallee != null || selectedCaller != null) {
                Point tmp = new Point(mouseDownPos.X - e.Location.X, mouseDownPos.Y - e.Location.Y);
                int x = Math.Abs(tmp.X);
                x *= x;
                int y = Math.Abs(tmp.Y);
                y *= y;
                if (Math.Sqrt(x + y) > 4 && e.Button == System.Windows.Forms.MouseButtons.Left) {
                    drawConnection = true;
                    if (tabViews.SelectedTab != null) {
                        doTheScrollingShit(e.Location);
                    }
                    refreshCurrent();
                }
            } else {
                refreshCurrent();
            }
            lastMousePos = e.Location;
        }

        private void doTheScrollingShit(Point point) {
            Panel p = tabViews.SelectedTab.Controls[0] as Panel;

            if (point.X > p.HorizontalScroll.Value + p.Width + 10) {
                p.HorizontalScroll.Value += 10;
            } else if (point.X < p.HorizontalScroll.Value - 10) {
                p.HorizontalScroll.Value = Math.Max(0, p.HorizontalScroll.Value - 10);
            }
            if (point.Y > p.VerticalScroll.Value + p.Height + 10) {
                p.VerticalScroll.Value += 10;
            } else if (point.Y < p.VerticalScroll.Value - 10) {
                p.VerticalScroll.Value = Math.Max(0, p.VerticalScroll.Value - 10);
            }
        }

        private void tabViews_MouseUp(object sender, MouseEventArgs e) {
            movedModule = null;
            TabPage tp = tabViews.SelectedTab;
            Data.CallerSlot cr;
            Data.CalleeSlot ce;
            bool somethingSelected = false;
            if (tp != null) {
                foreach (GraphicalModule gm in tabModules[tp]) {
                    if (gm.IsSlotHit(e.Location, out ce, out cr, out connectingTip)) {
                        if (drawConnection && (selectedCallee != null || selectedCaller != null)) {
                            GraphicalModule src = null, dest = null;
                            Data.CallerSlot caller = null;
                            Data.CalleeSlot callee = null;
                            if (selectedCaller != null && ce != null) {
                                src = selectedModule;
                                dest = gm;
                                caller = selectedCaller;
                                callee = ce;
                            } else if (selectedCallee != null && cr != null) {
                                src = gm;
                                dest = selectedModule;
                                caller = cr;
                                callee = selectedCallee;
                            }
                            if (src != null) {
                                string[] compatibles = caller.CompatibleCalls.Intersect(callee.CompatibleCalls).ToArray();
                                if (compatibles.Count() > 0) {
                                    string theName = compatibles[0];
                                    if (compatibles.Count() > 1) {
                                        //MessageBox.Show("DEBUG-Info: multiple compatible calls found");
                                        using (CallSelector cs = new CallSelector(compatibles)) {
                                            cs.StartPosition = FormStartPosition.Manual;
                                            Point p = this.PointToScreen(e.Location);
                                            cs.Location = p;
                                            if (cs.ShowDialog(this) == System.Windows.Forms.DialogResult.Cancel) {
                                                SelectItem((GraphicalConnection)null);
                                                refreshCurrent();
                                                return;
                                            }
                                            theName = cs.SelectedItem;
                                        }
                                    }

                                    GraphicalConnection gc = new GraphicalConnection(src, dest, caller, callee, FindCallByName(theName));
                                    // does the callerslot already have a connection? erase it
                                    foreach (GraphicalConnection gtemp in tabConnections[tp]) {
                                        if (gtemp.src.Equals(src) && caller.Equals(gtemp.srcSlot)) {
                                            tabConnections[tp].Remove(gtemp);
                                            break;
                                        }
                                    }
                                    tabConnections[tp].Add(gc);
                                    somethingSelected = true;
                                    drawConnection = false;
                                    break;
                                }
                            }
                            //GraphicalConnection gc = new GraphicalConnection(
                        }
                    }
                }
                if (!somethingSelected) {
                    drawConnection = false;
                }
                resizePanel(true);
                refreshCurrent();
            }
        }

        private Data.PluginFile FindPluginByFileName(string p) {
            foreach (Data.PluginFile pf in plugins) {
                if (pf.Filename.Equals(p)) {
                    return pf;
                }
            }
            return null;
        }

        private Data.Call FindCallByName(string name) {
            foreach (Data.PluginFile pf in plugins) {
                foreach (Data.Call c in pf.Calls) {
                    if (c.Name.Equals(name)) {
                        return c;
                    }
                }
            }
            return null;
        }

        private Data.Module FindModuleByName(string name) {
            foreach (Data.PluginFile pf in plugins) {
                foreach (Data.Module m in pf.Modules) {
                    if (m.Name.Equals(name)) {
                        return m;
                    }
                }
            }
            return null;
        }

        /// <summary>
        /// Create a new project pane
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void btnNewProject_Click(object sender, EventArgs e) {
            this.newTabPage("Project " + ++tabCount);
        }

        /// <summary>
        /// Create a new project pane
        /// </summary>
        /// <param name="name">The name for the project pane</param>
        /// <returns>The new project page</returns>
        private TabPage newTabPage(string name) {
            tabViews.TabPages.Add(name);
            TabPage tp = tabViews.TabPages[tabViews.TabCount - 1];
            tp.ImageIndex = 1;
            tabModules[tp] = new List<GraphicalModule>();
            tabConnections[tp] = new List<GraphicalConnection>();
            tabMainViews[tp] = null;
            tabFileNames[tp] = null;
            tabStartParameters[tp] = null;
            NoStupidScrollingPanel p = new NoStupidScrollingPanel();
            p.AutoScroll = true;
            //p.HorizontalScroll.Visible = true;
            //p.VerticalScroll.Visible = true;
            tp.Controls.Add(p);
            p.Dock = DockStyle.Fill;
            p.BackColor = Color.Gainsboro;
            NoflickerPanel np = new NoflickerPanel();
            np.Tag = tp;
            p.Controls.Add(np);
            //np.Dock = DockStyle.Fill;
            np.Location = new Point(0, 0);
            //np.Size = new System.Drawing.Size(tabViews.GetTabRect(tabViews.SelectedIndex).Width,
            //    tabViews.GetTabRect(tabViews.SelectedIndex).Height);
            np.Paint += PaintTabpage;
            np.MouseDown += tabViews_MouseDown;
            np.MouseMove += tabViews_MouseMove;
            np.MouseUp += tabViews_MouseUp;
            np.PreviewKeyDown += tabViews_PreviewKeyDown;
            np.BackColor = Color.Gainsboro;
            tabViews.SelectedTab = tp;
            propertyGrid1.SelectedObject = new TabPageDescriptor(tp);
            SelectItem((GraphicalModule)null);
            updateFiltered();
            return tp;
        }

        void tabViews_PreviewKeyDown(object sender, PreviewKeyDownEventArgs e) {
            if (e.KeyCode == Keys.Delete) {
                btnDelete_Click(sender, null);
            }
        }

        /// <summary>
        /// Closes the current project pane
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void btnCloseProject_Click(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            this.CloseProjectTab(tp);
        }

        /// <summary>
        /// Closes the specified project tab page
        /// </summary>
        /// <param name="tp">The project tab page to close</param>
        private void CloseProjectTab(TabPage tp) {
            if (tp != null) {
                propertyGrid1.SelectedObject = null;
                tabViews.TabPages.Remove(tp);
                tabModules.Remove(tp);
                tabConnections.Remove(tp);
                tabModules.Remove(tp);
                tabConnections.Remove(tp);
                tabMainViews.Remove(tp);
                tabSelectedObjects.Remove(tp);
                tabFileNames.Remove(tp);
                tabStartParameters.Remove(tp);
                tp.Dispose();
            }
            if (tabViews.TabPages.Count <= 0) {
                this.btnNewProject_Click(null, null);
            }
        }

        /// <summary>
        /// Deletes the currently selected module or connection
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void btnDelete_Click(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            if (tp == null) return;

            if (selectedModule != null) {
                GraphicalModule gm = selectedModule;
                SelectItem((GraphicalModule)null);
                this.propertyGrid1.SelectedObject = null;
                List<GraphicalConnection> rgc = new List<GraphicalConnection>();
                foreach (GraphicalConnection gc in tabConnections[tp]) {
                    if ((gc.dest == gm) || (gc.src == gm)) {
                        rgc.Add(gc);
                    }
                }
                foreach (GraphicalConnection gc in rgc) {
                    tabConnections[tp].Remove(gc);
                }
                tabModules[tp].Remove(gm);
                refreshCurrent();

            } else if (selectedConnection != null) {
                tabConnections[tp].Remove(selectedConnection);
                SelectItem((GraphicalConnection)null);
                refreshCurrent();
            }
        }

        /// <summary>
        /// Makes a name safe for using in MegaMol
        /// </summary>
        /// <param name="name">The input name</param>
        /// <returns>The safe output name</returns>
        private static string safeName(string name) {
            return name.Replace(' ', '_');
        }

        /// <summary>
        /// Escapes string characters for xml serialization
        /// </summary>
        /// <param name="p">The input string</param>
        /// <returns>The output string</returns>
        private string safeString(string p) {
            return p.Replace("&", "&amp;").Replace("\"", "&quot;");
        }

        /// <summary>
        /// Escapes string characters for command line
        /// </summary>
        /// <param name="p">The input string</param>
        /// <returns>The output string</returns>
        private string safeCmdLineString(string p) {
            return "\"" + p.Replace("\"", "\"\"") + "\"";
        }

        /// <summary>
        /// Escapes string characters for command line
        /// </summary>
        /// <param name="p">The input string</param>
        /// <returns>The output string</returns>
        private string safePSLineString(string p) {
            return "\"" + p.Replace("\"", "`\"`\"") + "\"";
        }

        /// <summary>
        /// Creates the MegaMol command line
        /// </summary>
        /// <param name="tp">The tab page</param>
        /// <param name="filename">The file name</param>
        /// <param name="forPS">True for PowerShell and False for Cmd</param>
        /// <returns>The command line</returns>
        private string makeCmdLine(TabPage tp, string filename, bool forPS) {
            StringBuilder cmdLine = new StringBuilder();
            cmdLine.AppendFormat("-p \"{0}\" -i {1} inst", filename, safeName(tp.Text));
            foreach (GraphicalModule m in tabModules[tp]) {
                foreach (KeyValuePair<Data.ParamSlot, bool> p in m.ParameterCmdLineness) {
                    if (!p.Value) continue;
                    cmdLine.Append(" -v inst::");
                    cmdLine.Append(m.Name);
                    cmdLine.Append("::");
                    cmdLine.Append(p.Key.Name);
                    cmdLine.Append(" ");
                    if (m.ParameterValues.ContainsKey(p.Key)) {
                        if (forPS) {
                            cmdLine.Append(this.safePSLineString(m.ParameterValues[p.Key]));
                        } else {
                            cmdLine.Append(this.safeCmdLineString(m.ParameterValues[p.Key]));
                        }
                    } else {
                        cmdLine.Append("Click");
                    }
                }
            }
            return cmdLine.ToString();
        }

        /// <summary>
        /// The file name of teh dictionary
        /// </summary>
        private Dictionary<TabPage, string> tabFileNames = new Dictionary<TabPage,string>();

        /// <summary>
        /// Saves the currently selected pane as MegaMol Project file
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void btnSaveProject_Click(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            if (tp == null) return;
            
            try {
                if (String.IsNullOrEmpty(this.tabFileNames[tp])) {
                    saveShortcut = false;
                }
                this.saveFileDialog1.FileName = this.tabFileNames[tp];
                this.saveFileDialog1.InitialDirectory = System.IO.Path.GetDirectoryName(this.saveFileDialog1.FileName);
            } catch {
            }

            if (saveShortcut || this.saveFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK) {
                if (String.IsNullOrEmpty(System.IO.Path.GetExtension(this.saveFileDialog1.FileName))) {
                    this.saveFileDialog1.FileName = System.IO.Path.ChangeExtension(this.saveFileDialog1.FileName, this.saveFileDialog1.DefaultExt);
                }
                saveShortcut = false;

                Io.ProjectFile1 mmprj = new Io.ProjectFile1();
                mmprj.GeneratorComment = "generated by MegaMol™ Configurator " + System.Reflection.Assembly.GetExecutingAssembly().GetName().Version.ToString();
                mmprj.StartComment = @"

Use this command line arguments to start MegaMol™
in Cmd:
  " + this.makeCmdLine(tp, this.saveFileDialog1.FileName, false) + @"
in PowerShell:
  " + this.makeCmdLine(tp, this.saveFileDialog1.FileName, true) + @"

";
                Io.ProjectFile1.View view = new Io.ProjectFile1.View();
                mmprj.Views = new Io.ProjectFile1.View[1] { view };
                view.Name = safeName(tp.Text);
                List<Io.ProjectFile1.Module> mods = new List<Io.ProjectFile1.Module>();
                foreach (GraphicalModule m in tabModules[tp]) {
                    Io.ProjectFile1.Module mod = new Io.ProjectFile1.Module();
                    mods.Add(mod);
                    mod.Class = m.Module.Name;
                    mod.Name = safeName(m.Name);
                    mod.ConfPos = m.Position;
                    if (m == tabMainViews[tp]) view.ViewModule = mod;
                    List<Io.ProjectFile1.Param> param = new List<Io.ProjectFile1.Param>();
                    foreach (KeyValuePair<Data.ParamSlot, string> p in m.ParameterValues) {
                        if (m.ParameterCmdLineness[p.Key] || (p.Value == ((Data.ParamTypeValueBase)p.Key.Type).DefaultValueString())) continue;
                        Io.ProjectFile1.Param pm = new Io.ProjectFile1.Param();
                        pm.Name = p.Key.Name;
                        pm.Value = p.Value;
                        param.Add(pm);
                    }
                    mod.Params = (param.Count == 0) ? null : param.ToArray();
                }
                view.Modules = (mods.Count == 0) ? null : mods.ToArray();
                List<Io.ProjectFile1.Call> calls = new List<Io.ProjectFile1.Call>();
                foreach (GraphicalConnection c in tabConnections[tp]) {
                    Io.ProjectFile1.Call call = new Io.ProjectFile1.Call();
                    calls.Add(call);
                    call.Class = c.Call.Name;
                    call.FromSlot = c.srcSlot.Name;
                    call.ToSlot = c.destSlot.Name;
                    foreach (Io.ProjectFile1.Module mod in view.Modules) {
                        if (mod.Name == c.src.Name) call.FromModule = mod;
                        if (mod.Name == c.dest.Name) call.ToModule = mod;
                    }
                }
                view.Calls = (calls.Count == 0) ? null : calls.ToArray();

                mmprj.Save(this.saveFileDialog1.FileName);

                this.tabFileNames[tp] = this.saveFileDialog1.FileName;
                tp.ToolTipText = this.saveFileDialog1.FileName;
            }

        }

        /// <summary>
        /// Opens the project website
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void websiteToolStripMenuItem_Click(object sender, EventArgs e) {
            System.Diagnostics.Process.Start("http://megamol.org/");
        }

        /// <summary>
        /// Shows the about dialog
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void aboutToolStripMenuItem_Click(object sender, EventArgs e) {
            Util.AboutBox ab = new Util.AboutBox();
            ab.ShowDialog();
        }

        /// <summary>
        /// Loads a MegaMol project file
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void btnLoadProject_Click(object sender, EventArgs e) {
            try {
                TabPage tp = tabViews.SelectedTab;
                if (tp != null) {
                    this.openFileDialog1.FileName = this.tabFileNames[tp];
                }
                this.openFileDialog1.InitialDirectory = System.IO.Path.GetDirectoryName(this.openFileDialog1.FileName);
            } catch {
            }

            if (this.openFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK) {
                try {
                    Io.ProjectFile project = Io.ProjectFile.Load(this.openFileDialog1.FileName);
                    LoadProjectFile(project, this.openFileDialog1.FileName);
                    this.saveFileDialog1.FileName = this.openFileDialog1.FileName;
                    this.saveFileDialog1.InitialDirectory = this.openFileDialog1.InitialDirectory;
                    //TabPage tp = tabViews.SelectedTab;
                    //if (tp != null) {
                    //    this.tabFileNames[tp] = this.openFileDialog1.FileName;
                    //}

                } catch (Exception ex) {
                    MessageBox.Show("Failed to load: " + ex.ToString(), Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        [DllImport("User32.dll", SetLastError = true)]
        static extern void SwitchToThisWindow(IntPtr hWnd, bool fAltTab);

        public void LoadProjectFile(Io.ProjectFile project, string filename) {
            try {
                SwitchToThisWindow(Handle, true);
            } catch { }

            foreach (Io.ProjectFile.View view in project.Views) {
                TabPage tp = this.loadMMPrjView(view);

                tp.ToolTipText = filename;
                this.tabFileNames[tp] = filename;
            }

        }

        /// <summary>
        /// Loads a view description from a MegaMol project file
        /// </summary>
        /// <param name="view">The ProjectFile.View to load</param>
        private TabPage loadMMPrjView(Io.ProjectFile.View view) {
            string viewname = view.Name;

            List<Tuple<string, Data.Module, List<Tuple<Data.ParamSlot, string>>>> modules = new List<Tuple<string, Data.Module, List<Tuple<Data.ParamSlot, string>>>>();
            List<Tuple<Data.Call, string, string>> callWishes = new List<Tuple<Data.Call, string, string>>();
            List<Tuple<Data.Call, string, Data.CallerSlot, string, Data.CalleeSlot>> calls = new List<Tuple<Data.Call, string, Data.CallerSlot, string, Data.CalleeSlot>>();
            Dictionary<string, Point> modulePositions = new Dictionary<string, Point>();

            if (view.Modules != null) {
                foreach (Io.ProjectFile.Module m in view.Modules) {
                    Data.Module mod = this.FindModuleByName(m.Class);
                    if (mod == null) throw new Exception("Unknown Module \"" + m.Class + "\" encountered");

                    Tuple<string, Data.Module, List<Tuple<Data.ParamSlot, string>>> mt = new Tuple<string, Data.Module, List<Tuple<Data.ParamSlot, string>>>(
                        m.Name, mod, new List<Tuple<Data.ParamSlot, string>>());
                    modules.Add(mt);
                    if (!m.ConfPos.IsEmpty) modulePositions[m.Name] = m.ConfPos;

                    if (m.Params != null) {
                        foreach (Io.ProjectFile.Param prm in m.Params) {

                            Data.ParamSlot ps = null;
                            foreach (Data.ParamSlot p in mt.Item2.ParamSlots) {
                                if (p.Name == prm.Name) {
                                    ps = p;
                                    break;
                                }
                            }
                            if (ps == null) throw new Exception("Paramslot \"" + prm.Name + "\" not found");

                            if (ps.Type is Data.ParamType.Enum) {
                                prm.Value = ((Data.ParamType.Enum)ps.Type).ParseValue(prm.Value).ToString();
                            } else if (ps.Type is Data.ParamType.Float) {
                                float f;
                                if (float.TryParse(prm.Value,
                                        System.Globalization.NumberStyles.Float,
                                        System.Globalization.CultureInfo.InvariantCulture,
                                        out f)) {
                                    prm.Value = f.ToString(System.Globalization.CultureInfo.InvariantCulture);
                                } else if (float.TryParse(prm.Value, out f)) {
                                    prm.Value = f.ToString(System.Globalization.CultureInfo.InvariantCulture);
                                }
                            }

                            mt.Item3.Add(new Tuple<Data.ParamSlot, string>(ps, prm.Value));
                        }
                    }
                }
            }
            if (view.Calls != null) {
                foreach (Io.ProjectFile.Call c in view.Calls) {
                    Data.Call call = this.FindCallByName(c.Class);
                    if (call == null) throw new Exception("Unknown call encountered");
                    callWishes.Add(new Tuple<Data.Call, string, string>(call, c.FromModule.Name + "::" + c.FromSlot, c.ToModule.Name + "::" + c.ToSlot));
                }
            }
            if (view.Params != null) {
                throw new Exception("View-global parameters not supported");
                //foreach (Io.ProjectFile.Param prm in view.Params) {

                //    Data.ParamSlot ps = null;
                //    foreach (Data.ParamSlot p in mt.Item2.ParamSlots) {
                //        if (p.Name == prm.Name) {
                //            ps = p;
                //            break;
                //        }
                //    }
                //    if (ps == null) throw new Exception("Paramslot \"" + prm.Name + "\" not found");

                //    if (ps.Type is Data.ParamType.Enum) {
                //        prm.Value = ((Data.ParamType.Enum)ps.Type).ParseValue(prm.Value).ToString();
                //    } else if (ps.Type is Data.ParamType.Float) {
                //        float f;
                //        if (float.TryParse(prm.Value,
                //                System.Globalization.NumberStyles.Float,
                //                System.Globalization.CultureInfo.InvariantCulture,
                //                out f)) {
                //            prm.Value = f.ToString(System.Globalization.CultureInfo.InvariantCulture);
                //        } else if (float.TryParse(prm.Value, out f)) {
                //            prm.Value = f.ToString(System.Globalization.CultureInfo.InvariantCulture);
                //        }
                //    }

                //    mt.Item3.Add(new Tuple<Data.ParamSlot, string>(ps, prm.Value));
                //}
            }

            foreach (Tuple<Data.Call, string, string> cw in callWishes) {
                string callfromname = cw.Item2;
                string calltoname = cw.Item3;

                Data.CallerSlot fromslot = null;
                Data.CalleeSlot toslot = null;
                foreach (Tuple<string, Data.Module, List<Tuple<Data.ParamSlot, string>>> mt in modules) {
                    if (callfromname.StartsWith(mt.Item1)) {
                        string slotname = callfromname.Substring(mt.Item1.Length);
                        if (slotname.StartsWith("::")) slotname = slotname.Substring(2);
                        if (slotname.StartsWith(".")) slotname = slotname.Substring(1);
                        foreach (Data.CallerSlot crs in mt.Item2.CallerSlots) {
                            if (crs.Name.Equals(slotname, StringComparison.InvariantCultureIgnoreCase)) {
                                fromslot = crs;
                                callfromname = mt.Item1;
                                break;
                            }
                        }
                    }
                    if (calltoname.StartsWith(mt.Item1)) {
                        string slotname = calltoname.Substring(mt.Item1.Length);
                        if (slotname.StartsWith("::")) slotname = slotname.Substring(2);
                        if (slotname.StartsWith(".")) slotname = slotname.Substring(1);
                        foreach (Data.CalleeSlot ces in mt.Item2.CalleeSlots) {
                            if (ces.Name.Equals(slotname, StringComparison.InvariantCultureIgnoreCase)) {
                                toslot = ces;
                                calltoname = mt.Item1;
                                break;
                            }
                        }
                    }
                }
                if (fromslot == null) throw new Exception("CallerSlot \"" + callfromname + "\" not found");
                if (toslot == null) throw new Exception("CalleeSlot \"" + calltoname + "\" not found");

                if (!fromslot.CompatibleCalls.Contains(cw.Item1.Name)) throw new Exception("CallerSlot \"" + callfromname + "\" is not compatible with call \"" + cw.Item1.Name + "\"");
                if (!toslot.CompatibleCalls.Contains(cw.Item1.Name)) throw new Exception("CalleeSlot \"" + calltoname + "\" is not compatible with call \"" + cw.Item1.Name + "\"");

                calls.Add(new Tuple<Data.Call, string, Data.CallerSlot, string, Data.CalleeSlot>(cw.Item1, callfromname, fromslot, calltoname, toslot));
            }

            // now instantiate the data
            TabPage tp = this.newTabPage(viewname);
            bool modPosMissing = false;

            foreach (Tuple<string, Data.Module, List<Tuple<Data.ParamSlot, string>>> miw in modules) {
                GraphicalModule gm = new GraphicalModule(miw.Item2, tabModules[tp]);
                gm.Name = miw.Item1;
                if (modulePositions.Keys.Contains(gm.Name)) {
                    gm.Position = modulePositions[gm.Name];
                } else {
                    modPosMissing = true;
                }
                tabModules[tp].Add(gm);
                foreach (Tuple<Data.ParamSlot, string> pw in miw.Item3) {
                    gm.ParameterValues[pw.Item1] = pw.Item2;
                }
            }
            if (view.ViewModule != null) {
                foreach (GraphicalModule gm in tabModules[tp]) {
                    if (gm.Name == view.ViewModule.Name) {
                        tabMainViews[tp] = gm;
                        break;
                    }
                }
            }
            if (tabMainViews[tp] == null) {
                foreach (GraphicalModule gm in tabModules[tp]) {
                    if (gm.Module.IsViewModule) {
                        tabMainViews[tp] = gm;
                        break;
                    }
                }
            }

            foreach (Tuple<Data.Call, string, Data.CallerSlot, string, Data.CalleeSlot> ciw in calls) {
                GraphicalModule src = null;
                GraphicalModule dst = null;
                foreach (GraphicalModule gm in tabModules[tp]) {
                    if (gm.Name == ciw.Item2) src = gm;
                    if (gm.Name == ciw.Item4) dst = gm;
                }
                Debug.Assert(src != null);
                Debug.Assert(dst != null);
                GraphicalConnection gc = new GraphicalConnection(src, dst, ciw.Item3, ciw.Item5, ciw.Item1);
                tabConnections[tp].Add(gc);
            }

            refreshCurrent();
            if (modPosMissing) ForceDirectedLayout(tp);
            updateFiltered();
            refreshCurrent();
            this.resizePanel();

            return tp;
        }

        /// <summary>
        /// Performs force-directed layout on tp
        /// </summary>
        /// <param name="tp">The TabPage to layout</param>
        private void ForceDirectedLayout(TabPage tp) {
            Random rnd = new Random();
            foreach (GraphicalModule gm in tabModules[tp]) {
                gm.Position = new Point(rnd.Next(2500), rnd.Next(1000));
            }
            if (tabMainViews[tp] != null) {
                tabMainViews[tp].Position = new Point(20, this.Height / 3);
            } else if (tabModules[tp].Count > 0) {
                tabModules[tp][0].Position = new Point(20, this.Height / 3);
            }

            foreach (GraphicalModule gm in tabModules[tp]) {
                gm.Pos = gm.Position;
                gm.Speed = PointF.Empty;
            }

            for (int iter = 0; iter < 10000; iter++) {
                float fullspeed = layoutStep(tp);
                if (fullspeed < 3.0f) break;
            }
        }

        /// <summary>
        /// Performs one step of force-directed layout on tp
        /// </summary>
        /// <param name="tp">The TabPage to layout</param>
        /// <returns>The speed of the system</returns>
        private float layoutStep(TabPage tp) {
            // no force
            foreach (GraphicalModule gm in tabModules[tp]) {
                gm.Force = PointF.Empty;
            }

            // repulsive force
            const float k = 75.0f;
            foreach (GraphicalModule gm1 in tabModules[tp]) {
                foreach (GraphicalModule gm2 in tabModules[tp]) {
                    if (gm2 == gm1) continue;

                    PointF vec = new PointF(gm2.Pos.X - gm1.Pos.X, gm2.Pos.Y - gm1.Pos.Y); // vec from gm1 to gm2
                    float dist = (float)Math.Sqrt(vec.X * vec.X + vec.Y * vec.Y); // distance

                    gm2.Force.X += k * vec.X / (dist * dist);
                    gm2.Force.Y += k * vec.Y / (dist * dist);
                }
            }

            // attractive force
            const float springConst = 0.01f;
            foreach (GraphicalConnection gc in tabConnections[tp]) {
                float dist = Math.Max(gc.Bounds.Width, gc.Bounds.Height) * 1.25f;

                PointF fromPoint = new PointF(
                    gc.src.GetTipLocation(gc.srcSlot).X + dist * 0.33f,
                    gc.src.GetTipLocation(gc.srcSlot).Y);

                PointF toPoint = new PointF(
                    gc.dest.GetTipLocation(gc.destSlot).X - dist * 0.33f,
                    gc.dest.GetTipLocation(gc.destSlot).Y);

                PointF vec = new PointF(toPoint.X - fromPoint.X, toPoint.Y - fromPoint.Y); // vec from fromPoint to toPoint
                float vecLen = (float)Math.Sqrt(vec.X * vec.X + vec.Y * vec.Y); // distance
                vec.X /= vecLen;
                vec.Y /= vecLen;
                vecLen -= dist * 0.34f;

                gc.src.Force.X += springConst * vec.X * vecLen;
                gc.src.Force.Y += springConst * vec.Y * vecLen;
                gc.dest.Force.X -= springConst * vec.X * vecLen;
                gc.dest.Force.Y -= springConst * vec.Y * vecLen;
            }

            float fullSpeed = 0.0f;

            // integration
            const float damping = 0.95f;
            const float timestep = 0.75f;
            foreach (GraphicalModule gm in tabModules[tp]) {
                gm.Speed.X = gm.Speed.X * damping + gm.Force.X * timestep;
                gm.Speed.Y = gm.Speed.Y * damping + gm.Force.Y * timestep;

                if (tabMainViews[tp] != null) {
                    if (tabMainViews[tp] == gm) continue;
                } else if (tabModules[tp].Count > 0) {
                    if (tabModules[tp][0] == gm) continue;
                }

                fullSpeed += (float)Math.Sqrt(gm.Speed.X * gm.Speed.X + gm.Speed.Y * gm.Speed.Y);

                gm.Pos.X += gm.Speed.X * timestep;
                gm.Pos.Y += gm.Speed.Y * timestep;

                if (gm.Pos.X < 0.0f) {
                    gm.Pos.X = 0.0f;
                    gm.Speed.X = -gm.Speed.X;
                }
                if (gm.Pos.Y < 0.0f) {
                    gm.Pos.Y = 0.0f;
                    gm.Speed.Y = -gm.Speed.Y;
                }

                gm.Repos();
            }

            return fullSpeed;
        }

        /// <summary>
        /// Refresh drawing if property changed in property grid
        /// </summary>
        /// <param name="s">not used</param>
        /// <param name="e">not used</param>
        private void propertyGrid1_PropertyValueChanged(object s, PropertyValueChangedEventArgs e) {
            this.refreshCurrent();
            MegaMolInstanceInfo mmii = this.tabViews.SelectedTab.Tag as MegaMolInstanceInfo;
            if (mmii != null) {
                mmii.SendUpdate(e.ChangedItem.Label, e.ChangedItem.Value.ToString());
            }
        }

        /// <summary>
        /// Update menu items on menu opening
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void propertyGridContextMenuStrip_Opening(object sender, CancelEventArgs e) {
            GridItem item = this.propertyGrid1.SelectedGridItem;
            if (item.PropertyDescriptor == null) item = null;
            this.resetValueToolStripMenuItem.Enabled = (item != null) && (item.PropertyDescriptor.CanResetValue(((GraphicalModuleDescriptor)this.propertyGrid1.SelectedObject).Module));
            GraphicalModuleParameterDescriptor gmpd = (item != null) ? item.PropertyDescriptor as GraphicalModuleParameterDescriptor : null;
            this.storeInProjectFileToolStripMenuItem.Enabled = (gmpd != null);
            this.storeInProjectFileToolStripMenuItem.Checked = (gmpd != null) && !((GraphicalModuleDescriptor)this.propertyGrid1.SelectedObject).Module.ParameterCmdLineness[gmpd.Parameter];
            this.specifyInCommandLineToolStripMenuItem.Enabled = (gmpd != null);
            this.specifyInCommandLineToolStripMenuItem.Checked = !this.storeInProjectFileToolStripMenuItem.Checked;
        }

        /// <summary>
        /// Resetting a property value
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void resetValueToolStripMenuItem_Click(object sender, EventArgs e) {
            this.propertyGrid1.ResetSelectedProperty();
        }

        /// <summary>
        /// Marks a property to be stored in the MegaMol™ Project file
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void storeInProjectFileToolStripMenuItem_Click(object sender, EventArgs e) {
            GridItem item = this.propertyGrid1.SelectedGridItem;
            if ((item == null) || (item.PropertyDescriptor == null) || !(item.PropertyDescriptor is GraphicalModuleParameterDescriptor)) return;
            if ((this.propertyGrid1.SelectedObject == null) || !(this.propertyGrid1.SelectedObject is GraphicalModuleDescriptor)) return;
            ((GraphicalModuleDescriptor)this.propertyGrid1.SelectedObject).Module.ParameterCmdLineness[((GraphicalModuleParameterDescriptor)item.PropertyDescriptor).Parameter] = false;
            ((GraphicalModuleParameterDescriptor)item.PropertyDescriptor).UseInCmdLine = false;
            item.Select();
        }

        /// <summary>
        /// Marks a property to be specified in the command line
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void specifyInCommandLineToolStripMenuItem_Click(object sender, EventArgs e) {
            GridItem item = this.propertyGrid1.SelectedGridItem;
            if ((item == null) || (item.PropertyDescriptor == null) || !(item.PropertyDescriptor is GraphicalModuleParameterDescriptor)) return;
            if ((this.propertyGrid1.SelectedObject == null) || !(this.propertyGrid1.SelectedObject is GraphicalModuleDescriptor)) return;
            ((GraphicalModuleDescriptor)this.propertyGrid1.SelectedObject).Module.ParameterCmdLineness[((GraphicalModuleParameterDescriptor)item.PropertyDescriptor).Parameter] = true;
            ((GraphicalModuleParameterDescriptor)item.PropertyDescriptor).UseInCmdLine = true;
            item.Select();
        }

        /// <summary>
        /// Default-wise collapse the "Button Parameters 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void propertyGrid1_SelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e) {
            GridItem i = this.propertyGrid1.SelectedGridItem;
            while (i.Parent != null) i = i.Parent;
            foreach (GridItem j in i.GridItems) {
                if (j.GridItemType != GridItemType.Category) continue;
                if (j.Label == "Button Parameters") {
                    j.Expanded = false;
                    this.propertyGrid1.SelectedGridItemChanged -= propertyGrid1_SelectedGridItemChanged;
                    break;
                }
            }
        }

        private void tabViews_SelectedIndexChanged(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            if (tp != null) {
                if (tabSelectedObjects.ContainsKey(tp)) {
                    this.propertyGrid1.SelectedObject = tabSelectedObjects[tp];
                    GraphicalModuleDescriptor gmd = tabSelectedObjects[tp] as GraphicalModuleDescriptor;
                    if (gmd != null) {
                        SelectItem(gmd.Module);
                    }
                } else {
                    this.propertyGrid1.SelectedObject = null;
                }
                resizePanel();
                refreshCurrent();
            }
        }

        private void propertyGrid1_SelectedObjectsChanged(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            if (tp != null) {
                tabSelectedObjects[tp] = propertyGrid1.SelectedObject;
            }
        }

        private void tabViews_Click(object sender, EventArgs e) {
            if (tabViews.SelectedTab != null) {
                propertyGrid1.SelectedObject = new TabPageDescriptor(tabViews.SelectedTab);
            }
        }

        /// <summary>
        /// Copy the command line (for cmd) of the last saved file to the clipboard
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void forCmdToolStripMenuItem_Click(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            if (tp == null) return;
            Clipboard.SetText(this.makeCmdLine(tp, string.IsNullOrWhiteSpace(this.tabFileNames[tp]) ? "<filename.mmprj>" : this.tabFileNames[tp], false));
        }

        /// <summary>
        /// Copy the command line (for Powershell) of the last saved file to the clipboard
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void forPowershellToolStripMenuItem_Click(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            if (tp == null) return;
            Clipboard.SetText(this.makeCmdLine(tp, string.IsNullOrWhiteSpace(this.tabFileNames[tp]) ? "<filename.mmprj>" : this.tabFileNames[tp], true));
        }

        /// <summary>
        /// Trigger re-layouting of graph
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void toolStripButton1_Click(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            if (tp == null) return;
            ForceDirectedLayout(tp);
            updateFiltered();
            refreshCurrent();
            this.resizePanel();
        }

        /// <summary>
        /// Close tab if clicked by middle mouse button
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">The mouse event args identifying the clicking mouse button</param>
        private void tabViews_MouseClick(object sender, MouseEventArgs e) {
            if (e.Button == System.Windows.Forms.MouseButtons.Middle) {
                for (int i = 0; i < this.tabViews.TabCount; i++) {
                    Rectangle tabRect = this.tabViews.GetTabRect(i);
                    if (tabRect.Contains(e.Location)) {
                        // this tab hit
                        this.CloseProjectTab(this.tabViews.TabPages[i]);
                        return;
                    }
                }
            }
        }

        /// <summary>
        /// Display info about plugins state
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void MenuItem_StateInfo_Click(object sender, EventArgs e) {
            StringBuilder sb = new StringBuilder();
            if ((this.plugins == null) || (this.plugins.Count <= 0)) {
                sb.Append("No MegaMolConf State loaded");
            } else {
                int mc = 0;
                int cc = 0;
                foreach (Data.PluginFile p in this.plugins) {
                    sb.Append(p.Filename);
                    sb.Append("\n");
                    sb.AppendFormat("\t{0} Modules, {1} Calls\n", p.Modules.Length, p.Calls.Length);
                    mc += p.Modules.Length;
                    cc += p.Calls.Length;
                }
                sb.Append("Summary:\n");
                sb.AppendFormat("\t{0} Modules, {1} Calls", mc, cc);
            }
            MessageBox.Show(sb.ToString(), Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        /// <summary>
        /// Display the simple settings dialog
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void settingsToolStripMenuItem_Click(object sender, EventArgs e) {
            Util.SettingsDialog d = new Util.SettingsDialog();
            d.Settings = new Properties.Settings();
            if (d.ShowDialog() == System.Windows.Forms.DialogResult.OK) {
                d.Settings.Save();
                Properties.Settings.Default.Reload();
            }
        }

        /// <summary>
        /// Update menu entry states when the "start" menu is opening
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void toolStripDropDownButton4_DropDownOpening(object sender, EventArgs e) {
            bool enable = false;
            TabPage tp = tabViews.SelectedTab;
            if (tp != null) {
                enable = File.Exists(this.tabFileNames[tp]);
            }

            this.startMegaMolToolStripMenuItem.Enabled = enable;
            this.megaMolStartArgumentsToolStripMenuItem.Enabled = enable;
            this.forCmdToolStripMenuItem.Enabled = enable;
            this.forPowershellToolStripMenuItem.Enabled = enable;
            registerProjectToViewFileTypesToolStripMenuItem.Enabled = (Environment.OSVersion.Platform == PlatformID.Win32NT) && enable;
        }

        /// <summary>
        /// Shows the dialog to edit the MegaMol command line arguments
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void megaMolStartArgumentsToolStripMenuItem_Click(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            if (tp == null) return;

            StartParamDialog dlg = new StartParamDialog();
            dlg.ShellType = (StartParamDialog.StartShellType)Properties.Settings.Default.StartShellType;
            dlg.KeepShellOpen = Properties.Settings.Default.StartShellKeepOpen;
            dlg.LiveConnection = Properties.Settings.Default.LiveConnection;
            dlg.StdCmdArgs = this.makeCmdLine(tp, this.tabFileNames[tp], false);
            dlg.StdPSArgs = this.makeCmdLine(tp, this.tabFileNames[tp], true);
            dlg.ArgsHistory = Properties.Settings.Default.StartParamHistory;
            dlg.StartArgs = this.EnsureCmdLineArguments(this.tabStartParameters[tp], dlg.StdCmdArgs, dlg.StdPSArgs, dlg.ShellType);
            dlg.Application = Properties.Settings.Default.MegaMolBin;
            dlg.Application = this.EnsureMegaMolFrontendApplication(dlg.Application);
            dlg.WorkingDir = Properties.Settings.Default.WorkingDirectory;
            dlg.UseApplicationWorkingDir = Properties.Settings.Default.UseApplicationDirectoryAsWorkingDirectory;

            DialogResult dr = dlg.ShowDialog();

            if ((dr == System.Windows.Forms.DialogResult.OK) || (dr == System.Windows.Forms.DialogResult.Yes)) {
                Properties.Settings.Default.StartShellType = (int)dlg.ShellType;
                Properties.Settings.Default.StartShellKeepOpen = dlg.KeepShellOpen;
                Properties.Settings.Default.LiveConnection = dlg.LiveConnection;
                this.tabStartParameters[tp] = dlg.StartArgs;
                if (Properties.Settings.Default.StartParamHistory == null) Properties.Settings.Default.StartParamHistory = new System.Collections.Specialized.StringCollection();
                if (!string.IsNullOrWhiteSpace(dlg.StartArgs)) {
                    Properties.Settings.Default.StartParamHistory.Remove(dlg.StartArgs);
                    Properties.Settings.Default.StartParamHistory.Insert(0, dlg.StartArgs);
                    const int MAX_HIST_LEN = 25;
                    while (Properties.Settings.Default.StartParamHistory.Count > MAX_HIST_LEN) {
                        Properties.Settings.Default.StartParamHistory.RemoveAt(MAX_HIST_LEN);
                    }
                }
                Properties.Settings.Default.MegaMolBin = dlg.Application;
                Properties.Settings.Default.WorkingDirectory = dlg.WorkingDir;
                Properties.Settings.Default.UseApplicationDirectoryAsWorkingDirectory = dlg.UseApplicationWorkingDir;
                Properties.Settings.Default.Save();
            }

            if (dr == System.Windows.Forms.DialogResult.Yes) {
                this.startMegaMolToolStripMenuItem_Click(null, null);
            }
        }

        /// <summary>
        /// Makes sure a valid MegaMol Frontend Application is selected.
        /// </summary>
        /// <param name="p">The application path</param>
        /// <returns>The ensured application path</returns>
        private string EnsureMegaMolFrontendApplication(string p) {

            if (string.IsNullOrWhiteSpace(p) && plugins != null) {
                if (Environment.OSVersion.Platform == PlatformID.Win32NT) {
                    // file not specified, so assume console based on the core
                    foreach (Data.PluginFile pf in this.plugins) {
                        if (!pf.IsCore) continue;
                        Match m = Regex.Match(pf.Filename, @"^(.*)MegaMolCore([^\.]*)\.dll$");
                        if (m.Success) {
                            p = String.Format("{0}MegaMolCon{1}.exe", m.Groups[1].Value, m.Groups[2].Value);
                            break;
                        }
                    }
                } else {
                    // file not specified, so assume console based on the core
                    foreach (Data.PluginFile pf in this.plugins) {
                        if (!pf.IsCore) continue;
                        Match m = Regex.Match(pf.Filename, @"^(.*)/[^/]+/MegaMolCore([^\.]*)\.so$");
                        if (m.Success) {
                            p = String.Format("{0}megamol.sh", m.Groups[1].Value);
                            break;
                        }
                    }
                }
            }

            // file must exist
            if (!File.Exists(p)) p = null;
            return p;
        }

        /// <summary>
        /// Makes sure the command line arguments are meaningful
        /// </summary>
        /// <param name="arg">The command line arguments</param>
        /// <param name="defCmdArgs">The default command line arguments</param>
        /// <param name="defPSArgs">The default Powershell command line arguments</param>
        /// <param name="shell">The shell to start with</param>
        /// <returns>The ensured command line arguments</returns>
        private string EnsureCmdLineArguments(string arg, string defCmdArgs, string defPSArgs, StartParamDialog.StartShellType shell) {
            if (string.IsNullOrWhiteSpace(arg)) {
                arg = (shell == StartParamDialog.StartShellType.Powershell) ? defPSArgs : defCmdArgs;
            }
            return arg;
        }

        /// <summary>
        /// Starts MegaMol
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void startMegaMolToolStripMenuItem_Click(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            if (tp == null) return;

            try {

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = this.EnsureMegaMolFrontendApplication(Properties.Settings.Default.MegaMolBin);
                psi.WorkingDirectory = Properties.Settings.Default.UseApplicationDirectoryAsWorkingDirectory
                    ? Path.GetDirectoryName(psi.FileName)
                    : Properties.Settings.Default.WorkingDirectory;
                psi.UseShellExecute = false;

                StartParamDialog.StartShellType shell = (StartParamDialog.StartShellType)Properties.Settings.Default.StartShellType;
                bool keepOpen = Properties.Settings.Default.StartShellKeepOpen;
                bool live = Properties.Settings.Default.LiveConnection;
                string cmdLine = this.EnsureCmdLineArguments(this.tabStartParameters[tp],
                    this.makeCmdLine(tp, this.tabFileNames[tp], false),
                    this.makeCmdLine(tp, this.tabFileNames[tp], true),
                    shell);

                MegaMolInstanceInfo mmii = new MegaMolInstanceInfo();
                mmii.Port = ReservePort();
                mmii.ParentForm = this;
                if (live) {
                    cmdLine += " -o LRHostAddress tcp://*:" + mmii.Port;
                }
                switch (shell) {
                    case StartParamDialog.StartShellType.Direct:
                        psi.Arguments = cmdLine;
                        break;
                    case StartParamDialog.StartShellType.Cmd:
                        psi.Arguments = string.Format("{0} \"{2} {1}\"", keepOpen ? "/K" : "/C", cmdLine, psi.FileName);
                        psi.FileName = "CMD";
                        break;
                    case StartParamDialog.StartShellType.Powershell:
                        psi.Arguments = string.Format("{0}-Command \"{2}\" {1}", keepOpen ? "-NoExit " : "", cmdLine, psi.FileName);
                        psi.FileName = "powershell";
                        break;
                    default:
                        goto case StartParamDialog.StartShellType.Direct;
                }

                mmii.Process = Process.Start(psi);
                mmii.Process.EnableRaisingEvents = true;
                tp.ImageIndex = 0;
                mmii.Process.Exited += new EventHandler(delegate (Object o, EventArgs a) {
                    SetTabPageIcon(tp, 1);
                    listBoxLog.Log(Util.Level.Info, string.Format("Tab '{0}' exited", tp.Text));
                    mmii.Process = null;
                });
                mmii.TabPage = tp;
                tp.Tag = mmii;
                if (live) {
                    mmii.Thread = new Thread(new ThreadStart(mmii.Observe));
                    mmii.Thread.Start();
                } else {
                    mmii.Connection = null;
                }
            } catch (Exception ex) {
                MessageBox.Show("Failed to Start MegaMol™: " + ex.ToString(),
                    Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        /// <summary>
        /// Fancy printing the graph
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">not used</param>
        private void printGraphToolStripMenuItem_Click(object sender, EventArgs e) {
            try {
                Rectangle bounds = Rectangle.Empty;
                foreach (GraphicalModule gm in this.tabModules[this.tabViews.SelectedTab]) {
                    Rectangle gmb = new Rectangle(gm.Position, gm.Bounds);
                    bounds = bounds.IsEmpty ? gmb : Rectangle.Union(bounds, gmb);
                }
                if (bounds.IsEmpty) throw new Exception("Graph is empty.");

                this.printDialog1.Document = this.printDocument1;
                if (this.printDialog1.ShowDialog() != System.Windows.Forms.DialogResult.OK) return;
                this.printDocument1.Print();

            } catch (Exception ex) {
                MessageBox.Show("Failed: " + ex);
            }

        }

        /// <summary>
        /// Prints the graph
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">The print page arguments</param>
        private void printDocument1_PrintPage(object sender, System.Drawing.Printing.PrintPageEventArgs e) {
            Rectangle bounds = Rectangle.Empty;
            foreach (GraphicalModule gm in this.tabModules[this.tabViews.SelectedTab]) {
                Rectangle gmb = gm.DrawBounds;
                bounds = bounds.IsEmpty ? gmb : Rectangle.Union(bounds, gmb);
            }
            if (bounds.IsEmpty) throw new Exception("Graph is empty.");
            bounds.Inflate(10, 10);

            e.Graphics.ResetTransform();
            float scale = Math.Min((float)e.PageBounds.Width / (float)bounds.Width,
                (float)e.PageBounds.Height / (float)bounds.Height);
            e.Graphics.ScaleTransform(scale, scale);
            e.Graphics.TranslateTransform(-bounds.Left, -bounds.Top);

            object sel = this.tabSelectedObjects[this.tabViews.SelectedTab];
            this.tabSelectedObjects[this.tabViews.SelectedTab] = null;
            foreach (GraphicalModule gm in this.tabModules[this.tabViews.SelectedTab]) {
                gm.Draw(e.Graphics);
            }
            foreach (GraphicalConnection gc in this.tabConnections[this.tabViews.SelectedTab]) {
                gc.Draw(e.Graphics);
            }
            this.tabSelectedObjects[this.tabViews.SelectedTab] = sel;
        }

        private void moduleFilterBox_TextChanged(object sender, EventArgs e) {
            if (!string.IsNullOrEmpty(moduleFilterBox.Text)) {
                cmdResetFilter.Show();
            } else {
                cmdResetFilter.Hide();
            }
            this.updateFiltered();
        }

        private void lbModules_SelectedIndexChanged(object sender, EventArgs e) {
            Data.Module m = (lbModules.SelectedItem as Data.Module);
            if (m != null) {
                moduleText.Text = m.Description + System.Environment.NewLine + "[" + m.PluginName + "]";
            }
        }

        private void cmdResetFilter_Click(object sender, EventArgs e) {
            moduleFilterBox.Text = string.Empty;
            cmdResetFilter.Hide();
        }

        private void Form1_Shown(object sender, EventArgs e) {
            updateFiltered();
            refreshCurrent();
            this.resizePanel();

            if (Properties.Settings.Default.StartupChecks) {
                Util.StartupCheckForm scform = new Util.StartupCheckForm();
                scform.ShowDialog(this);
            }
            Assembly asm = Assembly.GetExecutingAssembly();
            FileVersionInfo fvi = FileVersionInfo.GetVersionInfo(asm.Location);
            listBoxLog.Log(Util.Level.Info, string.Format("MegaMolConf {0}.{1} build {2}",
                fvi.ProductMajorPart, fvi.ProductMinorPart, fvi.ProductBuildPart));
        }

        private void importParamfileToolStripMenuItem_Click(object sender, EventArgs e) {
            TabPage tp = tabViews.SelectedTab;
            if (tp == null) return;

            if (String.IsNullOrWhiteSpace(openParamfileDialog.InitialDirectory)) {
                try {
                    openParamfileDialog.InitialDirectory = Properties.Settings.Default.UseApplicationDirectoryAsWorkingDirectory
                        ? System.IO.Path.GetDirectoryName(Properties.Settings.Default.MegaMolBin)
                        : Properties.Settings.Default.WorkingDirectory;
                } catch { }
            }
            if (openParamfileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK) {
                Io.Paramfile paramFile = new Io.Paramfile();
                try {
                    paramFile.Load(openParamfileDialog.FileName);
                } catch (Exception ex) {
                    MessageBox.Show("Failed to load parameter file: " + ex.ToString(),
                        Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                ImportParamfileForm ipff = new ImportParamfileForm();
                ipff.ProjectName = tp.Text;
                ipff.Modules = tabModules[tp];
                ipff.ParamFile = paramFile;
                if (ipff.ShowDialog() == System.Windows.Forms.DialogResult.OK) {
                    object oo = propertyGrid1.SelectedObject;
                    propertyGrid1.SelectedObject = null;
                    propertyGrid1.SelectedObject = oo;
                }

            }
        }

        private void registerProjectToViewFileTypesToolStripMenuItem_Click(object sender, EventArgs e) {
            if (Environment.OSVersion.Platform != PlatformID.Win32NT) {
                MessageBox.Show(this, "This Feature is only available on the Windows Platform at the moment.", Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Hand);
                return;
            }

            TabPage tp = tabViews.SelectedTab;
            if (tp == null) return;

            string app = Properties.Settings.Default.MegaMolBin;
            app = this.EnsureMegaMolFrontendApplication(app);
            if (!File.Exists(app)) app = null;

            string core = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(app), "MegaMolCore.dll");
            if (!File.Exists(core)) core = null;

            string workDir = Properties.Settings.Default.WorkingDirectory;
            if (Properties.Settings.Default.UseApplicationDirectoryAsWorkingDirectory) {
                workDir = System.IO.Path.GetDirectoryName(app);
            }

            string args = this.makeCmdLine(tp, this.tabFileNames[tp], false);

            string ext = ".mmpld";
            string name = "MegaMol™ Particle List Data";
            string defaultIcon = this.safeCmdLineString(core) + ",-1001";
            string openCommand = this.safeCmdLineString(app) + " " + args.Replace("::filename \"\"", "::filename \"%1\"");

            Util.RegisterDataFileTypeDialog regDlg = new Util.RegisterDataFileTypeDialog();

            regDlg.FileExtension = ext;
            regDlg.FileDescription = name;
            regDlg.FileIconPath = defaultIcon;
            regDlg.FileOpenCommand = openCommand;

            regDlg.ShowDialog();

        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e) {
            foreach (TabPage tp in this.tabViews.TabPages) {
                MegaMolInstanceInfo mmii = tp.Tag as MegaMolInstanceInfo;
                if (mmii != null && mmii.Process != null) {
                    mmii.StopObserving();
                    mmii.Process.EnableRaisingEvents = false;
                    mmii.Process.Kill();
                }
            }
#if false
            foreach (IntPtr i in iconsToDestroy) {
                DestroyIcon(i);
            }
#endif
            e.Cancel = false;
        }

        private void toolStripCopy_Click(object sender, EventArgs e) {
            copiedModule = selectedModule;
        }

        private void toolStripPaste_Click(object sender, EventArgs e) {
            GraphicalModule m = pasteModule(tabViews.SelectedTab, copiedModule);
            selectedModule = m;
            this.propertyGrid1.SelectedObject = new GraphicalModuleDescriptor(m);
            this.refreshCurrent();
            resizePanel();
        }

        private void btnEyeDrop_Click(object sender, EventArgs e) {
            eyedropperTarget = selectedModule;
            if (eyedropperTarget != null && tabViews.SelectedTab != null) {
                tabViews.Cursor = eyeDropperCursor;
                isEyedropping = true;
            }
        }

        private void tabViews_Selected(object sender, TabControlEventArgs e) {
            Form1.selectedTab = this.tabViews.SelectedTab;
        }
    }
}
