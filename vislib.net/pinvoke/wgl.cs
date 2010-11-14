using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace vislib.pinvoke {

    /// <summary>
    /// class holding wgl functions
    /// </summary>
    public static class wgl {

        #region extensions

        private delegate int wglGetSwapIntervalExtDelegate();
        private static wglGetSwapIntervalExtDelegate wglGetSwapIntervalExt = null;
        private delegate int wglSwapIntervalExtDelegate(int interval);
        private static wglSwapIntervalExtDelegate wglSwapIntervalExt = null;

        #endregion

        #region private methods

        [DllImport("Opengl32.dll", CharSet = CharSet.Ansi)]
        private extern static IntPtr wglGetProcAddress(string procName);

        private static void assertSwapIntervalExt() {
            if (wglGetSwapIntervalExt == null) {
                try {
                    wglGetSwapIntervalExt = (wglGetSwapIntervalExtDelegate)Marshal.GetDelegateForFunctionPointer(
                        wglGetProcAddress("wglGetSwapIntervalEXT"), typeof(wglGetSwapIntervalExtDelegate));
                } catch {
                }
            }
            if (wglSwapIntervalExt == null) {
                try {
                    wglSwapIntervalExt = (wglSwapIntervalExtDelegate)Marshal.GetDelegateForFunctionPointer(
                        wglGetProcAddress("wglSwapIntervalEXT"), typeof(wglSwapIntervalExtDelegate));
                } catch {
                }
            }
        }

        #endregion

        #region public members

        static public bool VSync {
            get {
                assertSwapIntervalExt();
                if (wglGetSwapIntervalExt != null) {
                    return (wglGetSwapIntervalExt() == 1);
                }
                return false; /* we don't know any better */
            }
            set {
                assertSwapIntervalExt();
                if (wglSwapIntervalExt != null) {
                    wglSwapIntervalExt(value ? 1 : 0);
                }
            }
        }

        #endregion

    }

}
