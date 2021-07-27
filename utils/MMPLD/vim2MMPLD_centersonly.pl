# convert new imd MPIIO binaries to MMPLD
# this only works for very specific files since the observable description is textual and not regulated (it seems)
# you need to look at the header and then decide how to parse it (line 43 onwards)

use strict;
use warnings qw(FATAL);
use File::stat;

use MMPLD;

$#ARGV == 1 or die "usage: $0 <inputfile> <outputfile>";

my $inname = $ARGV[0];
my $outname = $ARGV[1];

print "in: $inname\tout: $outname\n";

open my $infile, '<', $inname or die "could not open input file $!";

my $frames = 0;
while($_ = <$infile>) {
    if (/^#/) {
        $frames++;
    }
}
$frames = $frames - 1;
print("found " . $frames . " frames\n");

my $m = MMPLD->new({filename=>$outname, numframes=>$frames});
seek $infile, 0, 0;
my %lists;
my $frametime = 0;
while($_ = <$infile>) {
    /^\~/ and next;
    if (/^#/) {
        my $numlists = (keys %lists);
        print "new frame, " . $numlists . " lists\n";
        $numlists == 0 and next;
        $m->StartFrame({frametime=>$frametime++, numlists=>$numlists});
        foreach my $key (sort keys %lists) {
            my $numatoms = scalar(@{$lists{$key}});
            $m->StartList({
                        vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_NONE, globalradius=>1.0,
                        globalcolor=>"255 255 255 255",
                        particlecount=>$numatoms
                        });
            print "key: $key, items: " . $numatoms . "\n";
            foreach my $p (@{$lists{$key}}) {
                $p =~ /(\d+)\s+(\d+)\s+(\d+)/;
                #print("x: $1, y: $2, z: $3\n");
                $m->AddParticle({x=>$1, y=>$2, z=>$3});
            }
        }
        %lists = ();
    } elsif (/^!\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)/) {
        ($1 == 0) && next;
        push @{$lists{$1}},"$2 $3 $4";
    }
}
$m->Close();