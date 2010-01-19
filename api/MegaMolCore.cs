///
/// MegaMolCore.cs
///
/// Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
/// Alle Rechte vorbehalten.
///
using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.IO;
using System.Windows.Forms;

namespace MegaMol.Core {

    internal partial class MegaMolCore {

        #region pInvoke

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr LoadLibrary(String dllname);

        [DllImport("kernel32.dll", CharSet = CharSet.Ansi, ExactSpelling = true, SetLastError = true)]
        private static extern IntPtr GetProcAddress(IntPtr hModule, String procname);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool FreeLibrary(IntPtr hModule);

        #endregion

        #region Data types

        /// <summary>
        /// Possible error codes
        /// </summary>
        public enum mmcErrorCode : int {
            MMC_ERR_NO_ERROR = 0, // No Error. This denotes success.
            MMC_ERR_MEMORY, // Generic memory error.
            MMC_ERR_HANDLE, // Generic handle error.
            MMC_ERR_INVALID_HANDLE, // The handle specified was invalid.
            MMC_ERR_NOT_INITIALISED, // The object was not initialised.
            MMC_ERR_STATE, // The object was in a incompatible state.
            MMC_ERR_TYPE, // Generic type error (normally incompatible type or cast 
                          // failed).
            MMC_ERR_NOT_IMPLEMENTED, // Function not implemented.
            MMC_ERR_LICENSING, // Requested action not possible due to licensing
            MMC_ERR_UNKNOWN // Unknown error.
        }

        /// <summary>
        /// Possinle operating systems
        /// </summary>
        public enum mmcOSys {
            MMC_OSYSTEM_WINDOWS,
            MMC_OSYSTEM_LINUX,
            MMC_OSYSTEM_UNKNOWN
        }

        /// <summary>
        /// Possible architectures
        /// </summary>
        public enum mmcHArch {
            MMC_HARCH_I86,
            MMC_HARCH_X64,
            MMC_HARCH_UNKNOWN
        }

        /// <summary>
        /// Possible initialisation values
        /// </summary>
        public enum mmcInitValue {
            MMC_INITVAL_CFGFILE, // The configuration file to load.
            MMC_INITVAL_CFGSET, // A configuration set to be added.
            MMC_INITVAL_LOGFILE, // The log file to use.
            MMC_INITVAL_LOGLEVEL, // The log level to use.
            MMC_INITVAL_LOGECHOLEVEL, // The log echo level to use.
            MMC_INITVAL_INCOMINGLOG, // Connects an incoming log object to the one of 
                                     // the core instance
            MMC_INITVAL_LOGECHOFUNC, // The log echo function to use.
            MMC_INITVAL_VISLIB_STACKTRACEMANAGER // The vislib StackTrace manager object
        }

        /// <summary>
        /// Possible value types
        /// </summary>
        public enum mmcValueType {
            MMC_TYPE_INT32, // 32 bit signed integer.(Pointer to!)
            MMC_TYPE_UINT32, // 32 bit unsigned integer.(Pointer to!)
            MMC_TYPE_INT64, // 64 bit signed integer.(Pointer to!)
            MMC_TYPE_UINT64, // 64 bit unsigned integer.(Pointer to!)
            MMC_TYPE_BYTE, // 8 bit unsigned integer.(Pointer to!)
            MMC_TYPE_BOOL, // bool (platform specific integer size) (Pointer to!)
            MMC_TYPE_FLOAT, // 32 bit float (Pointer to!)
            MMC_TYPE_CSTR, // Ansi string (Pointer or Array of ansi characters).
            MMC_TYPE_WSTR, // Unicode string (Pointer or Array of wide characters).
            MMC_TYPE_VOIDP
        }

        /// <summary>
        /// Library building flags
        /// </summary>
        [Flags]
        public enum mmcBFlag : uint {
            MMC_BFLAG_DEBUG = 0x00000001, // debug build
            MMC_BFLAG_DIRTY = 0x00000002  // dirty build (DO NOT RELEASE!)
        }

        #endregion

        #region Management

        /// <summary>
        /// The library handle
        /// </summary>
        private IntPtr lib = IntPtr.Zero;

        /// <summary>
        /// The full file name of the library
        /// </summary>
        private string filename;

        /// <summary>
        /// Gets the full file name of the library
        /// </summary>
        public string FileName {
            get { return this.filename; }
        }

        /// <summary>
        /// Answer whether this lib is valid or not
        /// </summary>
        public bool IsValid {
            get { return this.lib != IntPtr.Zero; }
        }

        /// <summary>
        /// Ctor.
        /// </summary>
        public MegaMolCore() {
            // intentionally empty. Use LoadCore!
        }

        /// <summary>
        /// Loads the core from default file names "MegaMolCore$(BitsD).dll"
        /// </summary>
        /// <returns>'true' on success</returns>
        public bool LoadCore() {
            // We are not loading debug dlls because they do not work properly!
            if (IntPtr.Size == 4) {
                if (this.LoadCoreDirect("MegaMolCore32.Dll")) return true;
                if (this.LoadCoreDirect("MegaMolCore64.Dll")) return true;
            } else {
                if (this.LoadCoreDirect("MegaMolCore64.Dll")) return true;
                if (this.LoadCoreDirect("MegaMolCore32.Dll")) return true;
            }
            return false;
        }

        /// <summary>
        /// Loads the core from default file names "MegaMolCore$(BitsD).dll"
        /// </summary>
        /// <param name="path">The path to load the core from</param>
        /// <returns>'true' on success</returns>
        public bool LoadCore(string path) {
            // We are not loading debug dlls because they do not work properly!
            if (IntPtr.Size == 4) {
                if (this.LoadCoreDirect(Path.Combine(path, "MegaMolCore32.Dll"))) return true;
                if (this.LoadCoreDirect(Path.Combine(path, "MegaMolCore64.Dll"))) return true;
            } else {
                if (this.LoadCoreDirect(Path.Combine(path, "MegaMolCore64.Dll"))) return true;
                if (this.LoadCoreDirect(Path.Combine(path, "MegaMolCore32.Dll"))) return true;
            }
            return false;
        }

        /// <summary>
        /// Loads the core from specified file name
        /// </summary>
        /// <returns>'true' on success</returns>
        public bool LoadCoreDirect(string filename) {
            if (this.lib != IntPtr.Zero) return false; // library already loaded

            try {
                this.lib = LoadLibrary(filename);
                if (this.lib == IntPtr.Zero) return false; // failed to load

                foreach (FieldInfo fi in this.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public)) {
                    fi.SetValue(this, Marshal.GetDelegateForFunctionPointer(GetProcAddress(this.lib, fi.Name), fi.FieldType));
                    if (fi.GetValue(this) == null) throw new Exception();
                }

                // For some reason working with DEBUG-Dlls does not work!!! So warn the user.
                mmcOSys os = mmcOSys.MMC_OSYSTEM_UNKNOWN;
                mmcHArch arch = mmcHArch.MMC_HARCH_UNKNOWN;
                int debug = 0;

                this.mmcGetCoreTypeInfo(ref os, ref arch, ref debug);
                if (debug != 0) {
                    DialogResult dr =
                        MessageBox.Show(
                        "The loaded library is a debug version. For some very strange reason debug dlls are not working properly sometimes.\n\nDo you which to continue using this library (NOT RECOMMENDED)?",
                        Application.ProductName, MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2);

                    if (dr != DialogResult.Yes) {
                        throw new Exception();
                    }
                }

            } catch {
                if (this.lib != IntPtr.Zero) {
                    try {
                        FreeLibrary(this.lib);
                        this.lib = IntPtr.Zero;
                    } catch {
                    }
                }
                return false;
            }

            this.filename = filename;
            return true;
        }

        #endregion

    }

}
