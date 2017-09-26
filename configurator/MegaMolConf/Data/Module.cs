using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MegaMolConf.Data {
    [DebuggerDisplay("Module {Name}")]
    public sealed class Module {
        public string Name { get; set; }
        public string Description { get; set; }
        public ParamSlot[] ParamSlots { get; set; }
        public CallerSlot[] CallerSlots { get; set; }
        public CalleeSlot[] CalleeSlots { get; set; }
        public override string ToString() {
            return Name;
        }
        public string PluginName { get; set; }
        public bool IsViewModule {
            get {
                return this.Name.Equals("SplitView")
                    || this.Name.Equals("ColStereoDisplay")
                    || this.Name.Equals("View2D")
                    || this.Name.Equals("View3D")
                    || this.Name.Equals("SimpleClusterView")
                    || this.Name.Equals("TileView")
                    || this.Name.Equals("PowerwallView")
                    || this.Name.Equals("QuadBufferStereoView")
                    || this.Name.Equals("AnaglyphStereoView")
                    || this.Name.Equals("TileView3D")
                    || this.Name.Equals("RemoteTileView");
            }
        }
    }
}
