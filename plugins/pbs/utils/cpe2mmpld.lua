mmCreateJob("converter", "DataWriterJob", "j")
mmCreateModule("CPERAWDataSource", "::converter::data")
mmCreateModule("MMPLDWriter", "::converter::writer")
mmSetParamValue("::converter::writer::version", "1.3")

mmCreateCall("DataWriterCtrlCall", "::converter::j::writer", "::converter::writer::control")
mmCreateCall("MultiParticleDataCall", "::converter::writer::data", "::converter::data::getdata")

inpath = "\\\\etude\\Archiv\\Daten\\Partikel\\FARO"
outpath = "\\\\paganmetal\\u$\\FARO\\"
--filename = "dischingen_011_000_110_100_100"
filename = mmGetConfigValue("file")

mmSetParamValue("::converter::data::filename", inpath .. filename .. ".raw")
mmSetParamValue("::converter::writer::filename", outpath .. filename .. ".mmpld")