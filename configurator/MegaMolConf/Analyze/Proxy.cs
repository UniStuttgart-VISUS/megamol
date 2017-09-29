using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace MegaMolConf.Analyze {

    /// <summary>
    /// Utility class to contact the proxy process
    /// </summary>
    internal class Proxy {

        /// <summary>
        /// The GUID to identify itself against the analyzer exe
        /// </summary>
        public const string analyzerGUID = "A328E88A-4BD4-479A-963F-00A46F93D4BD";

        /// <summary>
        /// The process start information
        /// </summary>
        private ProcessStartInfo psi = new ProcessStartInfo();

        /// <summary>
        /// The process
        /// </summary>
        private Process proc = null;

        /// <summary>
        /// The stdout data
        /// </summary>
        private StringBuilder stdout = new StringBuilder();

        /// <summary>
        /// The stderr data
        /// </summary>
        private StringBuilder stderr = new StringBuilder();

        /// <summary>
        /// Answer if a process is running in Wow64
        /// </summary>
        /// <param name="hProcess">The process handle</param>
        /// <param name="wow64Process">Set to ture if the process is executed in Wow64</param>
        /// <returns>Error codes</returns>
        [DllImport("kernel32.dll", SetLastError = true, CallingConvention = CallingConvention.Winapi)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool IsWow64Process([In] IntPtr hProcess, [Out] out bool wow64Process);

        /// <summary>
        /// Checks if the locally running process is 32bit but running in Wow64
        /// </summary>
        /// <returns>True if the currently running process is 32bit but running in Wow64</returns>
        private static bool InternalCheckIsWow64() {
            if ((Environment.OSVersion.Version.Major == 5 && Environment.OSVersion.Version.Minor >= 1) ||
                Environment.OSVersion.Version.Major >= 6) {
                using (Process p = Process.GetCurrentProcess()) {
                    bool retVal;
                    if (!IsWow64Process(p.Handle, out retVal)) {
                        return false;
                    }
                    return retVal;
                }
            } else {
                return false;
            }
        }

        /// <summary>
        /// Ctor
        /// </summary>
        /// <param name="bit">The bit for the proxy</param>
        public Proxy(int bit) {
            this.psi.CreateNoWindow = true;
            this.psi.ErrorDialog = false;
            this.psi.RedirectStandardError = true;
            this.psi.RedirectStandardInput = true;
            this.psi.RedirectStandardOutput = true;
            this.psi.UseShellExecute = false;
            this.psi.WorkingDirectory = Path.GetDirectoryName(Application.ExecutablePath);
            this.psi.FileName = string.Format("mmaproxy{0}.exe", bit);

            if (bit == 32) {
                if (!String.IsNullOrWhiteSpace(Properties.Settings.Default.AnalyzeProxy32)) {
                    if (File.Exists(Properties.Settings.Default.AnalyzeProxy32)) {
                        this.psi.FileName = Properties.Settings.Default.AnalyzeProxy32;
                    }
                }
            } else if (bit == 64) {
                bool is64BitOperatingSystem = (IntPtr.Size == 8) || InternalCheckIsWow64();
                if (is64BitOperatingSystem) {

                    if (!String.IsNullOrWhiteSpace(Properties.Settings.Default.AnalyzeProxy64)) {
                        if (File.Exists(Properties.Settings.Default.AnalyzeProxy64)) {
                            this.psi.FileName = Properties.Settings.Default.AnalyzeProxy64;
                        }
                    }

                } else {
                    // do not try to start 
                    this.psi = null;
                }

            } else throw new ArgumentException("bit must be 32 or 64");

            if (!File.Exists(psi.FileName)) {
                this.psi = null;
            }
        }

        /// <summary>
        /// Answer whether or not the proxy can be started
        /// </summary>
        public bool Enabled {
            get { return this.psi != null; }
        }

        /// <summary>
        /// Starts the proxy
        /// </summary>
        /// <param name="cmdline">The command line to talk to the proxy</param>
        public void Start(string cmdline) {
            this.stdout.Clear();
            this.stderr.Clear();
            this.psi.Arguments = cmdline;
            this.proc = Process.Start(this.psi);
            this.proc.ErrorDataReceived += proc_ErrorDataReceived;
            this.proc.OutputDataReceived += proc_OutputDataReceived;
            this.proc.BeginOutputReadLine();
            this.proc.BeginErrorReadLine();
        }

        /// <summary>
        /// Ansynchronously receive data from stdout of the proxy process
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">The read data</param>
        void proc_OutputDataReceived(object sender, DataReceivedEventArgs e) {
            this.stdout.Append(e.Data);
            this.stdout.Append('\n');
        }

        /// <summary>
        /// Ansynchronously receive data from stderr of the proxy process
        /// </summary>
        /// <param name="sender">not used</param>
        /// <param name="e">The read data</param>
        void proc_ErrorDataReceived(object sender, DataReceivedEventArgs e) {
            this.stderr.Append(e.Data);
            this.stderr.Append('\n');
        }

        /// <summary>
        /// Waits for the proxy process to exit
        /// </summary>
        /// <param name="sout">The lines wrote to stdout by the proxy process</param>
        /// <param name="serr">The lines wrote to stderr by the proxy process</param>
        internal void WaitForExit(out string[] sout, out string[] serr) {
            this.proc.WaitForExit();
            sout = this.stdout.ToString().Split(new char[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            serr = this.stderr.ToString().Split(new char[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        }

    }

}
