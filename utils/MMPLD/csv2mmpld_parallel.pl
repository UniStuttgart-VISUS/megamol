use MCE::Loop;
use Class::Struct;
use Data::Dumper;
use lib "Y:/SSD_Cache/src/megamol/utils/MMPLD";
use MMPLD;

MCE::Loop->init(chunk_size  => 2000000, max_workers => 12, use_slurpio => 1);

my $inputfile = 'region_34_27_10_atoms.csv';
#my $inputfile = 'region_8_23_25_atoms.csv';
#my $inputfile = 'hurz.txt';

struct(SphereType => {
    rad => '$',
    r => '$',
    g => '$',
    b => '$'
});

sub type_index {
    my ($rad, $r, $g, $b, $tps) = @_;
    my $found = -1;
    @types = @$tps;
    for my $i (0 .. $#types) {
        $t = $types[$i];
        if ($t->rad eq $rad and $t->r eq $r and $t->g eq $g and $t->b eq $b) {
            # found
            $found = $i;
            last;
        }
    }
    return $found;
}

sub type_exists {
    return (type_index(@_) ne -1) ? 1 : 0;
}

my @result = mce_loop_f {
    my ($mce, $slurp_ref, $chunk_id) = @_;
    #print $chunk_id . "\n";
    my @types;
    foreach my $line ( split /\n/, $$slurp_ref ) {
        chop $line;
        my ($x, $y, $z, $rad, $r, $g, $b) = split /,/, $line;
        $f = type_exists($rad, $r, $g, $b, \@types);
        if ($f == 0) {
            my $t = SphereType->new(rad => $rad, r => $r, g => $g, b => $b);
            $t->rad eq "radius" and next;
            #print "type is new: " . (Dumper $t) . "\n";
            push @types, $t;
        }
    }
    MCE->gather(@types);
} $inputfile;

#print join('', @result);
print "got " . ($#result + 1) . " types\n";

my @realtypes;
for my $t (@result) {
    my $f = type_exists($t->rad, $t->r, $t->g, $t->b, \@realtypes);
    if ($f == 0) {
        my $rt = SphereType->new(rad => $t->rad, r => $t->r, g => $t->g, b => $t->b);
        #print "type is new: " . (Dumper $rt) . "\n";
        #print "exists returns: " . $f . " (" . ($f ? "true" : "false") .  ")\n";
        #print "eval exists returns: " . type_exists($t->rad, $t->r, $t->g, $t->b, @realtypes) . " (" . (type_exists($t->rad, $t->r, $t->g, $t->b, @realtypes) ? "true" : "false") . ")\n";
        push @realtypes, $rt;
    }
}

print "got " . ($#realtypes + 1) . " filtered types\n";

my $m = MMPLD->new({filename=>"out.mmpld", numframes=>1});
$m->StartFrame({frametime=>0.0, numlists=>($#realtypes + 1)});

my $totalcount = 0;
for my $i (0 .. $#realtypes) {
    my $rt = $realtypes[$i];
    print "gathering type $i: " . (Dumper $rt) . "\n";

    my @result = mce_loop_f {
        my ($mce, $slurp_ref, $chunk_id) = @_;
        #print $chunk_id . "\n";
        my @parts;
        foreach my $line ( split /\n/, $$slurp_ref ) {
            chop $line;
            my ($x, $y, $z, $rad, $r, $g, $b) = split /,/, $line;
            my $t_idx = type_index($rad, $r, $g, $b, \@realtypes);
            if ($t_idx eq $i) {
                push @parts, "$x,$y,$z";
            }
        }
        MCE->gather(@parts);
    } $inputfile;
    my $num = $#result + 1;
    print "got $num particles.\n";
    $totalcount += $num;

    my $red = int($rt->r * 255.0);
    my $green = int($rt->g * 255.0);
    my $blue = int($rt->b * 255.0);
    $m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_NONE, globalradius=>$rt->rad, globalcolor=>"$red $green $blue 255",
            minintensity=>0.0, maxintensity=>255, particlecount=>$num
            });
    for my $p (@result) {
        my ($x, $y, $z) = split /,/, $p;
        $m->AddParticle({x=>$x, y=>$y, z=>$z});
    }
}

$m->Close();

print "$totalcount particles in all.";
