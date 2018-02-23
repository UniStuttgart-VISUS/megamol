using System.Windows.Forms;

namespace MegaMolConf {
    class NoStupidScrollingPanel : Panel {
        protected override System.Drawing.Point ScrollToControl(Control activeControl) {
            // Returning the current location prevents the panel from
            // scrolling to the active control when the panel loses and regains focus
            return DisplayRectangle.Location;
        }
    }
}
