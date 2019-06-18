
packinglistfile = "T:\\temp\\graph_mol\\plugins\\pbs\\utils\\pkd_pack.lua"

local content = mmReadTextFile(packinglistfile, nil)

--print(content)

code = load(content)
pack = code()

--print(table.concat(pack.nodeFiles[1], ";"))

-- mmCreateJob("pkd_concat", "DataWriterJob", "::pkd_concat::dwj")

-- mmCreateModule("StaticMMPLDProvider", "::pkd_concat::reader")

-- mmCreateModule("MMPLDWriter", "::pkd_concat::writer")
-- mmSetParamValue("::pkd_concat::writer::version", "103")

-- mmCreateCall("MultiParticleDataCall", "::pkd_concat::writer::data", "::pkd_concat::reader::outData")
-- mmCreateCall("DataWriterCtrlCall", "::pkd_concat::dwj::writer", "::pkd_concat::writer::control")

for el = 1, #pack.nodeFiles do
    filenames = table.concat(pack.nodeFiles[el], ";")
    print(filenames)
    -- mmSetParamValue("::pkd_concat::reader::filenames", "hurz")
    -- mmSetParamValue("::pkd_concat::writer::filename", "bla")
    -- mmFlush()
end

