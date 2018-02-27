using System.ComponentModel;
using System.Windows.Forms;

namespace MegaMolConf {
    class TabPageDescriptor {

        private TabPage tp;

        public TabPageDescriptor(TabPage tp) {
            this.tp = tp;
        }

        [Category("General"), Description("the name of the ViewDescription")]
        public string Name {
            get {
                return tp.Text;
            }
            set {
                tp.Text = value;
            }
        }
    }
}
