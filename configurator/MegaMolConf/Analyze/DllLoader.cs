using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace MegaMolConf.Analyze {

    public class DllLoader {
        const int LOAD_LIBRARY_AS_DATAFILE = 0x00000002;
        const int DONT_RESOLVE_DLL_REFERENCES = 0x00000001;
        const int LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008;
        const int LOAD_IGNORE_CODE_AUTHZ_LEVEL = 0x00000010;

        public delegate int VoidDelegateInt();
        public delegate IntPtr IntDelegateIntPtr(int idx);

        /// <summary>
        /// To load the dll - dllFilePath dosen't have to be const - so I can read path from registry
        /// </summary>
        /// <param name="dllFilePath">file path with file name</param>
        /// <param name="hFile">use IntPtr.Zero</param>
        /// <param name="dwFlags">What will happend during loading dll
        /// <para>LOAD_LIBRARY_AS_DATAFILE</para>
        /// <para>DONT_RESOLVE_DLL_REFERENCES</para>
        /// <para>LOAD_WITH_ALTERED_SEARCH_PATH</para>
        /// <para>LOAD_IGNORE_CODE_AUTHZ_LEVEL</para>
        /// </param>
        /// <returns>Pointer to loaded Dll</returns>
        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibraryEx(string dllFilePath, IntPtr hFile, uint dwFlags);

        /// <summary>
        /// To unload library 
        /// </summary>
        /// <param name="dllPointer">Pointer to Dll witch was returned from LoadLibraryEx</param>
        /// <returns>If unloaded library was correct then true, else false</returns>
        [DllImport("kernel32.dll")]
        public extern static bool FreeLibrary(IntPtr dllPointer);

        /// <summary>
        /// To get function pointer from loaded dll 
        /// </summary>
        /// <param name="dllPointer">Pointer to Dll witch was returned from LoadLibraryEx</param>
        /// <param name="functionName">Function name with you want to call</param>
        /// <returns>Pointer to function</returns>
        [DllImport("kernel32.dll", CharSet = CharSet.Ansi)]
        public extern static IntPtr GetProcAddress(IntPtr dllPointer, string functionName);

        /// <summary>
        /// This will to load concret dll file
        /// </summary>
        /// <param name="dllFilePath">Dll file path</param>
        /// <returns>Pointer to loaded dll</returns>
        /// <exception cref="ApplicationException">
        /// when loading dll will failure
        /// </exception>
        public static IntPtr LoadWin32Library(string dllFilePath) {
            System.IntPtr moduleHandle = LoadLibraryEx(dllFilePath, IntPtr.Zero, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (moduleHandle == IntPtr.Zero) {
                // I'm gettin last dll error
                int errorCode = Marshal.GetLastWin32Error();
                throw new ApplicationException(
                    string.Format("There was an error during dll loading : {0}, error - {1}", dllFilePath, errorCode)
                    );
            }
            return moduleHandle;
        }
    }
}
