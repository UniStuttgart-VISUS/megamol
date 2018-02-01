using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace MegaMolConf.Util {

    /// <summary>
    /// Utility class managing DPI settings
    /// </summary>
    /// <remarks>
    /// Copyright 2016, by SGrottel
    /// Implementation copied from ReWeb-Project
    /// </remarks>
    internal static class DisplayDPI {

        [DllImport("gdi32.dll")]
        static extern int GetDeviceCaps(IntPtr hdc, int nIndex);
        public enum DeviceCap {
            VERTRES = 10,
            DESKTOPVERTRES = 117,
            LOGPIXELSX = 88,

            // http://pinvoke.net/default.aspx/gdi32/GetDeviceCaps.html
        }

        internal static int DPI {
            get {
                Graphics g = Graphics.FromHwnd(IntPtr.Zero);
                IntPtr desktop = g.GetHdc();
                int LogicalScreenHeight = GetDeviceCaps(desktop, (int)DeviceCap.VERTRES);
                int PhysicalScreenHeight = GetDeviceCaps(desktop, (int)DeviceCap.DESKTOPVERTRES);

                float ScreenScalingFactor = (float)PhysicalScreenHeight / (float)LogicalScreenHeight;

                // dpi1 answers correctly if application is "dpiaware=false"
                int dpi1 = (int)(96.0 * ScreenScalingFactor);
                // dpi2 answers correctly if application is "dpiaware=true"
                int dpi2 = GetDeviceCaps(desktop, (int)DeviceCap.LOGPIXELSX);

                return Math.Max(dpi1, dpi2);
            }
        }

        [DllImport("SHCore.dll")]
        private static extern bool SetProcessDpiAwareness(PROCESS_DPI_AWARENESS awareness);

        //[DllImport("SHCore.dll")]
        //private static extern void GetProcessDpiAwareness(IntPtr hprocess, out PROCESS_DPI_AWARENESS awareness);

        private enum PROCESS_DPI_AWARENESS {
            Process_DPI_Unaware = 0,
            Process_System_DPI_Aware = 1,
            Process_Per_Monitor_DPI_Aware = 2
        }

        [DllImport("user32.dll", SetLastError = true)]
        static extern bool SetProcessDPIAware();

        internal static void TryEnableDPIAware() {
            try {
                SetProcessDpiAwareness(PROCESS_DPI_AWARENESS.Process_Per_Monitor_DPI_Aware);
            } catch {
                try { // fallback, use (simpler) internal function
                    SetProcessDPIAware();
                } catch { }
            }
        }

    }

}
