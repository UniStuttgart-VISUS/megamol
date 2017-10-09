using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace MegaMolConf
{
    static class Program
    {

        /// <summary>
        /// Pipe object to keep a single instance
        /// </summary>
        private static Util.IPCNamedPipe appServerPipe;

        /// <summary>
        /// The application's main form
        /// </summary>
        private static Form1 mainForm = null;

        /// <summary>
        /// Access to the application's main form
        /// </summary>
        public static Form1 MainForm { get { return mainForm; } }

        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static int Main(string[] args) {
            Util.DisplayDPI.TryEnableDPIAware();

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;

            int retval;
            if (!handleSpecialArguments(args, out retval)) return retval;

            if (Environment.OSVersion.Platform == PlatformID.Win32NT) {
                // Now check that only a single instance is running
                if (Properties.Settings.Default.SingleInstance) {
                    appServerPipe = new Util.IPCNamedPipe("MegaMolConf");
                    if (!appServerPipe.ServerPipeOpen) {
                        // I am a secondary instance
                        if (args.Length > 0) {
                            try {
                                appServerPipe.SendStrings(args);
                            } catch (Exception ex) {
                                MessageBox.Show("Failed to communicate parameters: " + ex.ToString(),
                                    Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
                            }
                        }
                        return 0;
                    } else {
                        // Create main form
                        if (mainForm == null) mainForm = new Form1();
                        appServerPipe.StringReceived += AppServerPipe_StringReceived;
                    }
                }
            }

            // Create main form
            if (mainForm == null) mainForm = new Form1();

            // parse normal command line arguments
            if (args != null) {
                foreach (string arg in args) {
                    Debug.Assert(!arg.StartsWith("?#"));
                    string error = TryLoadRunFile(arg);
                    if (error != null) {
                        MessageBox.Show(mainForm, error, Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }

            Application.Run(mainForm);
            return 0;   
        }

        private static bool handleSpecialArguments(string[] args, out int retval) {
            List<string> aa = new List<string>();
            int argc = args.Length;
            for (int i = 0; i < argc; ++i) {
                if ((args[i] == "?#REG") || (args[i] == "?#UNREG")) {
                    bool doReg = (args[i] == "?#REG");
                    try {
                        if (i + 2 >= argc) throw new Exception("syntax error. Too few arguments");
                        string appPath = args[i + 1];
                        IWin32Window wnd = NativeWindow.FromHandle((IntPtr)ulong.Parse(args[i + 2]));
                        if (doReg) {
                            Util.FileTypeRegistration.Register(wnd, appPath);
                        } else {
                            Util.FileTypeRegistration.Unregister(wnd, appPath);
                        }
                        retval = 0;
                        return false;
                    } catch (Exception ex) {
                        MessageBox.Show("Failed to " + (doReg ? "" : "un") + "register file type: " + ex.ToString());
                        retval = int.MinValue;
                        return false;
                    }
                } else if (args[i] == "?#FILEREG") {
                    try {
                        MemoryStream mem = new MemoryStream(Convert.FromBase64String(args[i + 1]));

                        string ext, desc, iconPath, openCmd;

                        byte[] lenData = new byte[4];
                        mem.Read(lenData, 0, 4);
                        int len = BitConverter.ToInt32(lenData, 0);
                        byte[] strData = new byte[len];
                        mem.Read(strData, 0, len);
                        ext = System.Text.Encoding.UTF8.GetString(strData);

                        mem.Read(lenData, 0, 4);
                        len = BitConverter.ToInt32(lenData, 0);
                        strData = new byte[len];
                        mem.Read(strData, 0, len);
                        desc = System.Text.Encoding.UTF8.GetString(strData);

                        mem.Read(lenData, 0, 4);
                        len = BitConverter.ToInt32(lenData, 0);
                        strData = new byte[len];
                        mem.Read(strData, 0, len);
                        iconPath = System.Text.Encoding.UTF8.GetString(strData);

                        mem.Read(lenData, 0, 4);
                        len = BitConverter.ToInt32(lenData, 0);
                        strData = new byte[len];
                        mem.Read(strData, 0, len);
                        openCmd = System.Text.Encoding.UTF8.GetString(strData);

                        IWin32Window wnd = NativeWindow.FromHandle((IntPtr)ulong.Parse(args[i + 2]));

                        Util.FileTypeRegistration.RegisterDataFileType(wnd, ext, desc, iconPath, openCmd);

                        retval = 0;
                        return false;
                    } catch (Exception ex) {
                        MessageBox.Show("Failed to register file type: " + ex.ToString());
                        retval = int.MinValue;
                        return false;
                    }

                } else if (args[i] == "?#FILEUNREG") {
                    try {
                        string ext = args[i + 1];
                        IWin32Window wnd = NativeWindow.FromHandle((IntPtr)ulong.Parse(args[i + 2]));
                        Util.FileTypeRegistration.UnregisterDataFileType(wnd, ext);
                        retval = 0;
                        return false;
                    } catch (Exception ex) {
                        MessageBox.Show("Failed to unregister file type: " + ex.ToString());
                        retval = int.MinValue;
                        return false;
                    }
                }
            }
            retval = 0;
            return true;
        }

        private static void AppServerPipe_StringReceived(object sender, string e) {
            string error = TryLoadRunFile(e);
            if (error != null) {
                MessageBox.Show(mainForm, error, Application.ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
        private static string TryLoadRunFile(string path) {
            if (mainForm.InvokeRequired) {
                return (string)mainForm.Invoke(new Func<string, string>(TryLoadRunFile), new object[] { path });
            }

            if (System.IO.File.Exists(path)) {
                try {
                    Io.ProjectFile pf = Io.ProjectFile.Load(path);
                    mainForm.LoadProjectFile(pf, path);
                } catch(Exception ex) {
                    return ex.ToString();
                }
            }

            return null;
        }

    }
}
