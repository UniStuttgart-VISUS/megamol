using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace vislib.pinvoke {

    /// <summary>
    /// class holding glu32 p/invoke definitions
    /// </summary>
    public static class glu32 {

        /// <summary>
        /// The library file name
        /// </summary>
        public const string LIBNAME = "Glu32.dll";

        #region structs

        #endregion

        #region functions

        [DllImport(LIBNAME)]
        public static extern void gluPerspective(double fovy, double aspect, double zNear, double zFar);

        #endregion

    }

}
