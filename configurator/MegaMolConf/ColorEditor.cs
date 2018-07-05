using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Globalization;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Forms.Design;

namespace MegaMolConf {

    internal class ColorEditor : UITypeEditor {
        private Data.ParamType.Color e;

        private IWindowsFormsEditorService _editorService;

        public ColorEditor(Data.ParamType.Color e) {
            this.e = e;
        }

        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context) {
            return UITypeEditorEditStyle.DropDown;
        }

        public override object EditValue(ITypeDescriptorContext context, IServiceProvider provider, object value) {
            _editorService = (IWindowsFormsEditorService)provider.GetService(typeof(IWindowsFormsEditorService));

            // show this model stuff
            ColorDialog colorDialog = new ColorDialog((string)value);
            _editorService.ShowDialog(colorDialog);

            return colorDialog.Format();
        }

        private void OnListBoxSelectedValueChanged(object sender, EventArgs e) {
            // close the drop down as soon as something is clicked
            _editorService.CloseDropDown();
        }
    }

}