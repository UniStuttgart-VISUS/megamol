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

        public const UInt32 WM_PAINT = 0x000F;

        #endregion

        #region functions

        [DllImport(LIBNAME)]
        public static extern IntPtr GetDC(IntPtr hWnd);

        [DllImport(LIBNAME)]
        public static extern int ReleaseDC(IntPtr hWnd, IntPtr hDC);

        [DllImport(LIBNAME, CharSet = CharSet.Auto)]
        public static extern IntPtr SendMessage(IntPtr hWnd, UInt32 Msg, IntPtr wParam, IntPtr lParam);

        [return: MarshalAs(UnmanagedType.Bool)]
        [DllImport(LIBNAME, SetLastError = true)]
        public static extern bool PostMessage(IntPtr hWnd, UInt32 Msg, IntPtr wParam, IntPtr lParam);

        [DllImport("user32.dll")]
        public static extern bool InvalidateRect(IntPtr hWnd, IntPtr lpRect, bool bErase);

        #endregion

    }

}
