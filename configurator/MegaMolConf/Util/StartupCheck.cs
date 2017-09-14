using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace MegaMolConf.Util {
    internal class StartupCheck {
        public string Title { get; set; }
        public string Description { get; set; }
        public string EvaluateComment { get; set; }
        public bool Value { get; set; }
        public Func<StartupCheck, bool> DoEvaluate { get; set; }
        public Action<StartupCheck, IWin32Window> DoFix { get; set; }
        public bool NeedsElevation { get; set; }
    }
}
