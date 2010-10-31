using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace vislib.pinvoke {

    /// <summary>
    /// class holding user32 p/invoke definitions
    /// </summary>
    public static class user32 {

        /// <summary>
        /// The library file name
        /// </summary>
        public const string LIBNAME = "user32.dll";

        #region structs

        #endregion

        #region functions

        [DllImport(LIBNAME)]
        public static extern IntPtr GetDC(IntPtr hWnd);

        [DllImport(LIBNAME)]
        public static extern int ReleaseDC(IntPtr hWnd, IntPtr hDC);

        #endregion

    }

}
