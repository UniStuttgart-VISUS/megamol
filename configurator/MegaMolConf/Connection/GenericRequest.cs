using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MegaMolConf.Communication {

    /// <summary>
    /// Abstract implementation helper for generic requests
    /// </summary>
    public abstract class AbstractGenericRequest : Request {
        protected string cmd;

        /// <summary>
        /// The command string to be sent
        /// </summary>
        public override string Command { get { return cmd; } }
    }

    /// <summary>
    /// A generic request allowing to send anything
    /// </summary>
    public class GenericRequest : AbstractGenericRequest {

        /// <summary>
        /// Gets or sets the command string to be sent
        /// </summary>
        public new string Command { get { return cmd; } set { cmd = value; } }

    }

}
