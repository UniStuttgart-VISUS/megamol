using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MegaMolConf.Data {
    [DebuggerDisplay("PluginFile {Filename}")]
    public sealed class PluginFile {

        /// <summary>
        /// Flag marking the core object
        /// </summary>
        public bool IsCore { get; set; }

        public string Filename { get; set; }
        public Module[] Modules { get; set; }
        public Call[] Calls { get; set; }
    }
}
