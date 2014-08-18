using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace vislib.pinvoke {

    /// <summary>
    /// class holding kernel32 p/invoke definitions
    /// </summary>
    public static class kernel32 {

        /// <summary>
        /// The library file name
        /// </summary>
        public const string LIBNAME = "kernel32.dll";

        #region structs

        #endregion

        #region functions

        //[DllImport(LIBNAME)]
        //public static extern IntPtr LoadLibrary(string lpFileName);

        #endregion

    }

}
