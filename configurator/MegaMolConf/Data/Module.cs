using System.Diagnostics;

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
                return Name.Equals("SplitView")
                    || Name.Equals("ColStereoDisplay")
                    || Name.Equals("View2D")
                    || Name.Equals("View3D")
                    || Name.Equals("GUIView")
                    || Name.Equals("SimpleClusterView")
                    || Name.Equals("TileView")
                    || Name.Equals("PowerwallView")
                    || Name.Equals("QuadBufferStereoView")
                    || Name.Equals("AnaglyphStereoView")
                    || Name.Equals("TileView3D")
                    || Name.Equals("RemoteTileView")
                    || Name.Equals("HeadView")
                    || Name.Equals("View3D_2");
            }
        }
    }
}
