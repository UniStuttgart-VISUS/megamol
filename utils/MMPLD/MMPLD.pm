# MMPLD.inc
use strict;
use warnings;

package MMPLD;

use Readonly;
use POSIX;
require Exporter;
our @ISA = qw(Exporter);
our @EXPORT = qw( $VERTEX_XYZ_FLOAT $VERTEX_XYZR_FLOAT $VERTEX_XYZ_DOUBLE $COLOR_NONE $COLOR_INTENSITY_FLOAT $COLOR_RGBA_BYTE $COLOR_RGB_FLOAT $COLOR_RGBA_FLOAT $COLOR_RGBA_USHORT $COLOR_INTENSITY_DOUBLE @VERTEX_FORMAT_NAMES @COLOR_FORMAT_NAMES );

Readonly my $NOTHINGGOES => 0;
Readonly my $CANADDFRAME => 1;
Readonly my $CANADDLIST => 2;
Readonly my $CANADDPARTICLE => 3;
Readonly my $ALLDONE => 4;
Readonly my $CLOSED => 5;

Readonly our $VERTEX_XYZ_FLOAT => 0;
Readonly our $VERTEX_XYZR_FLOAT => 1;
Readonly our $VERTEX_XYZ_DOUBLE => 2;

Readonly our @VERTEX_FORMAT_NAMES => ("xyz_float", "xyzr_float", "xyz_double");

Readonly our $COLOR_NONE => 0;
Readonly our $COLOR_INTENSITY_FLOAT => 1;
Readonly our $COLOR_RGBA_BYTE => 2;
Readonly our $COLOR_RGB_FLOAT => 3;
Readonly our $COLOR_RGBA_FLOAT => 4;
Readonly our $COLOR_RGBA_USHORT => 5;
Readonly our $COLOR_INTENSITY_DOUBLE => 6;

Readonly our @COLOR_FORMAT_NAMES => ("none", "int_float", "rgba_byte", "rgb_float", "rgba_float", "rgba_ushort", "int_double");

sub _closeAndDie {
    my $self = shift;
    close($self->{filehandle});
    die "ack. I'm dead.'";
}

sub new {
    my $class = shift;
    my ($args) = @_;
    my $filename = $args->{filename};
    my $numframes = $args->{numframes};
    
    defined $filename or die "you need to specify a filename";
    defined $numframes or die "you need to specify the number of frames";
    
    my $self = {};
    bless $self, $class;
    $self->_state($NOTHINGGOES);
    $self->_initialize($filename, $numframes);
    return $self;    
}

sub _initialize {
    my $self = shift;
    $self->FileName(shift);
    $self->NumFrames(shift);
    #die "private method _initialize called" unless caller[0]->isa(ref($self));
    $self->NumFrames() > 0 or die "you need at least one frame";
    $self->_state() == $NOTHINGGOES or die "trying to initialize a file that has already been written to";
    
    open my $fh, ">", $self->FileName() or die "cannot open $self->{filename}";
    $self->{filehandle} = $fh;
    binmode $self->{filehandle};
    #flock($self->{filehandle}, 1);
    
    print {$self->{filehandle}} "MMPLD\x00";
    $self->AppendUShorts(102); # version
    $self->AppendUInt32s($self->NumFrames());   # number of frames

    $self->{bboxposition} = $self->_doTell();
    $self->AppendFloats((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)); # (for now) illegal bbox
    $self->AppendFloats((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)); # (for now) illegal clipbox
    
    #$self->closeAndDie();
    #seek table #frames + 1 (last = end of file)
    $self->{seektable} = $self->_doTell();
    for (my $x = 0; $x < $self->NumFrames() + 1; $x++) {
        $self->AppendUInt64s($x); # (for now) illegal seek table entry
    }
    $self->{currframe} = 0;
    
    $self->{maxradius} = 0;
    #$self->{minx} = $self->{miny} = $self->{minz} = POSIX::FLT_MAX;
    #$self->{maxx} = $self->{maxy} = $self->{maxz} = 0;
    $self->{firstparticle} = 1;
    $self->_state($CANADDFRAME);
}

sub DESTROY {
    my $self = shift;
    if (${^GLOBAL_PHASE} eq 'DESTRUCT') {
        #if ($self and $self->{state} and $self->{state} != $CLOSED) {
        #    print "not closing a file explicitly will result in frame table corruption!\n";
        #}
        return;
    }
    #$self->Close();
}

sub _finishup {
    my $self = shift;
    my $end = $self->_doTell();
    $self->_state() == $ALLDONE or die "you cannot finalize a file when in state: " . $self->_stringState();
    #print "end = $end\n";
    # set bounding boxes
    $self->_doSeek($self->{bboxposition});

    if (defined $self->{doOverrideBBox} && $self->{doOverrideBBox} == 1) {
        $self->AppendFloats(($self->{forceMinX}, $self->{forceMinY}, $self->{forceMinZ}, $self->{forceMaxX}, $self->{forceMaxY}, $self->{forceMaxZ}));
        $self->AppendFloats(($self->{forceMinX} - $self->{maxradius}, $self->{forceMinY} - $self->{maxradius}, $self->{forceMinZ} - $self->{maxradius},
                             $self->{forceMaxX} + $self->{maxradius}, $self->{forceMaxY} + $self->{maxradius}, $self->{forceMaxZ} + $self->{maxradius}));
    } else {
        $self->AppendFloats(($self->{minx}, $self->{miny}, $self->{minz}, $self->{maxx}, $self->{maxy}, $self->{maxz}));
        $self->AppendFloats(($self->{minx} - $self->{maxradius}, $self->{miny} - $self->{maxradius}, $self->{minz} - $self->{maxradius},
                             $self->{maxx} + $self->{maxradius}, $self->{maxy} + $self->{maxradius}, $self->{maxz} + $self->{maxradius}));
    }
    
    # append end pointer
    $self->_doSeek($self->{seektable} + ($self->{currframe}) * 8);
    $self->AppendUInt64s($end);
    close($self->{filehandle});
    $self->_state($CLOSED);
}

sub StartFrame {
    my $self = shift;
    my ($args) = @_;
    my $frametime = $args->{frametime};
    my $numlists = $args->{numlists};
    defined $frametime or die "you need to set the frame time";
    $numlists > 0 or die "you need to write at least one list per frame";
    $self->_state() == $CANADDFRAME or die "you cannot add a frame when in state: " . $self->_stringState();
    # TODO check whether the last frame had all lists written!
    
    $self->{currframe} < $self->NumFrames() or die "tried to write more frames than available";
    
    # update frame seek table
    my $pos = $self->_doTell();
    $self->_doSeek($self->{seektable} + ($self->{currframe}) * 8);
    $self->AppendUInt64s($pos);
    $self->_doSeek($pos);
    
    #write frame
    $self->AppendFloats($frametime);
    $self->AppendUInt32s($numlists);
    $self->{numlists} = $numlists;
    $self->{currlist} = 0;
    $self->_state($CANADDLIST);

    $self->{currframe} = $self->{currframe} + 1;
}

sub StartList() {
    my $self = shift;
    my ($args) = @_;
    my $vertextype = $args->{vertextype};
    my $colortype = $args->{colortype};
    my $particlecount = $args->{particlecount};
    my $globalradius = $args->{globalradius};
    my $globalcolor = $args->{globalcolor};
    my $minintensity = $args->{minintensity};
    my $maxintensity = $args->{maxintensity};
    
    $self->_state() == $CANADDLIST or die "you cannot add a list when in state: " . $self->_stringState();
    $self->{currlist} < $self->{numlists} or die "tried to write more lists than available in frame $self->{currframe}";
    
    $self->{currvertextype} = $vertextype;
    if ($vertextype == $VERTEX_XYZ_FLOAT) {
        $self->AppendUBytes(1);
    } elsif ($vertextype == $VERTEX_XYZR_FLOAT) {
        $self->AppendUBytes(2);
    } elsif ($vertextype == $VERTEX_XYZ_DOUBLE) {
        $self->AppendUBytes(4);
    } else {
        die "illegal vertex type $vertextype";
    }
    
    $self->{currcolortype} = $colortype;
    if ($colortype == $COLOR_NONE) {
        $self->AppendUBytes(0);
    } elsif ($colortype == $COLOR_INTENSITY_FLOAT) {
        $self->AppendUBytes(3);        
    } elsif ($colortype == $COLOR_RGBA_BYTE) {
        $self->AppendUBytes(2);
    } elsif ($colortype == $COLOR_RGB_FLOAT) {
        $self->AppendUBytes(4);
    } elsif ($colortype == $COLOR_RGBA_FLOAT) {
        $self->AppendUBytes(5);
    } elsif ($colortype == $COLOR_RGBA_USHORT) {
        $self->AppendUBytes(6);
    } elsif ($colortype == $COLOR_INTENSITY_DOUBLE) {
        $self->AppendUBytes(7);
    } else {
        die "illegal color type $colortype";
    }
    
    if ($vertextype == $VERTEX_XYZ_FLOAT || $vertextype == $VERTEX_XYZ_DOUBLE) {
        if (!defined $globalradius ||  (0 + $globalradius) != $globalradius || $globalradius == 0) {
            die qq{global radius "$globalradius" is weird.};
        }
        $self->AppendFloats($globalradius);
        if ($globalradius > $self->{maxradius}) {
            $self->{maxradius} = $globalradius;
        }
    }
    if ($colortype == $COLOR_NONE) {
        my @gc = split /\s+/, $globalcolor;
        my $ok = _checkColor(@gc);
        if ($ok == 0) {
            die qq{global color "$globalcolor" is weird.};
        }
        $self->AppendUBytes(@gc);
    } elsif ($colortype == $COLOR_INTENSITY_FLOAT || $colortype == $COLOR_INTENSITY_DOUBLE) {
        if (!defined $minintensity || !defined $maxintensity || (0 + $minintensity) != $minintensity || (0 + $maxintensity) != $maxintensity) {
            die qq{minintensity "$minintensity" or maxintensity "$maxintensity" is weird.};
        }
        $self->AppendFloats($minintensity);
        $self->AppendFloats($maxintensity);
        $self->{currminintensity} = $minintensity;
        $self->{currmaxintensity} = $maxintensity;
    }
    
    if (!defined $particlecount || $particlecount == 0) {
        die qq{please specify a particlecount};
    }
    $self->AppendUInt64s($particlecount);
    
    $self->{numparticles} = $particlecount;
    $self->{currparticle} = 0;
    $self->_state($CANADDPARTICLE);
    $self->{currlist} = $self->{currlist} + 1;
}

sub AddParticle {
    my $self = shift;
    my ($args) = @_;
    my $x = $args->{x};
    my $y = $args->{y};
    my $z = $args->{z};
    my $rad = $args->{rad};
    my $r = $args->{r};
    my $g = $args->{g};
    my $b = $args->{b};
    my $a = $args->{a};
    my $i = $args->{i};
    
    $self->_state() == $CANADDPARTICLE or die "you cannot add a particle when in state: " . $self->_stringState();
    $self->{currparticle} < $self->{numparticles} or die "tried to write more particles than available in frame $self->{currframe}, list $self->{currlist}";
    
    if (!defined $x || !defined $y || !defined $z) {
        die "need xyz coordinates in frame $self->{currframe}, list $self->{currlist}, particle $self->{currparticle}";
    }
    if ($self->{currvertextype} == $VERTEX_XYZ_DOUBLE) {
        $self->AppendDoubles($x, $y, $z);
    } else {
        $self->AppendFloats($x, $y, $z);
    }
    $self->_adjustBounds($x, $y, $z);
    if ($self->{currvertextype} == $VERTEX_XYZR_FLOAT) {
        if (!defined $rad) {
            die "need radius in frame $self->{currframe}, list $self->{currlist}, particle $self->{currparticle}";
        }
        $self->AppendFloats($rad);
        if ($rad > $self->{maxradius}) {
            $self->{maxradius} = $rad;
        }
    }
    if ($self->{currcolortype} == $COLOR_INTENSITY_FLOAT) {
        if (!defined $i || $i < $self->{currminintensity} || $i > $self->{currmaxintensity}) {
            die qq{invalid intensity "$i" in frame $self->{currframe}, list $self->{currlist}, particle $self->{currparticle}};
        }
        $self->AppendFloats($i);
    } elsif ($self->{currcolortype} == $COLOR_INTENSITY_DOUBLE) {
        if (!defined $i || $i < $self->{currminintensity} || $i > $self->{currmaxintensity}) {
            die qq{invalid intensity "$i" in frame $self->{currframe}, list $self->{currlist}, particle $self->{currparticle}};
        }
        $self->AppendDoubles($i);
    } elsif ($self->{currcolortype} == $COLOR_RGBA_BYTE) {
        my @gc = ($r, $g, $b, $a);
        my $ok = _checkColor(@gc);
        if ($ok == 0) {
            die qq{color "$r, $g, $b, $a" is weird in frame $self->{currframe}, list $self->{currlist}, particle $self->{currparticle}.};
        }
        $self->AppendUBytes($r, $g, $b, $a);
    } elsif ($self->{currcolortype} == $COLOR_RGB_FLOAT) {
        my @gc = ($r, $g, $b);
        my $ok = _checkFloatColor(@gc);
        if ($ok == 0) {
            die qq{color "$r, $g, $b" is weird in frame $self->{currframe}, list $self->{currlist}, particle $self->{currparticle}.};
        }
        $self->AppendFloats($r, $g, $b);
    } elsif ($self->{currcolortype} == $COLOR_RGBA_FLOAT) {
        my @gc = ($r, $g, $b, $a);
        my $ok = _checkFloatColor(@gc);
        if ($ok == 0) {
            die qq{color "$r, $g, $b, $a" is weird in frame $self->{currframe}, list $self->{currlist}, particle $self->{currparticle}.};
        }
        $self->AppendFloats($r, $g, $b, $a);
    } elsif ($self->{currcolortype} == $COLOR_RGBA_USHORT) {
        my @gc = ($r, $g, $b, $a);
        my $ok = _checkUShortColor(@gc);
        if ($ok == 0) {
            die qq{color "$r, $g, $b, $a" is weird in frame $self->{currframe}, list $self->{currlist}, particle $self->{currparticle}.};
        }
        $self->AppendUShorts($r, $g, $b, $a);
    }
    
    $self->{currparticle} = $self->{currparticle} + 1;
    if ($self->{currparticle} == $self->{numparticles}) {
        if ($self->{currlist} == $self->{numlists}) {
            if ($self->{currframe} == $self->{numframes}) {
                # done
                #$self->_finishup();
                $self->_state($ALLDONE);
            } else {
                $self->_state($CANADDFRAME);
            }
        } else {
            $self->_state($CANADDLIST);
        }
    }
}

sub Close {
    my $self = shift;
    $self->_state() == $ALLDONE or die "you cannot close the MMPLD when in state: " . $self->_stringState();
    $self->_finishup();
}

sub OverrideBBox {
    my $self = shift;
    my ($minx, $miny, $minz, $maxx, $maxy, $maxz) = @_;
    $self->{forceMinX} = $minx;
    $self->{forceMinY} = $miny;
    $self->{forceMinZ} = $minz;
    $self->{forceMaxX} = $maxx;
    $self->{forceMaxY} = $maxy;
    $self->{forceMaxZ} = $maxz;
    $self->{doOverrideBBox} = 1;
    #print "overriding bounding box\n";
}

sub _doTell {
    my $self = shift;
    #die "private method _doTell called" unless caller[0]->isa(ref($self));
    return tell $self->{filehandle};
}

sub _doSeek {
    my $self = shift;
    #die "private method _doSeek called" unless caller[0]->isa(ref($self));
    my $pos = shift;
    seek($self->{filehandle}, $pos, 0); 
}

sub FileName {
    my $self = shift;
    if (@_) {
        $self->{filename} = shift;
    }
    return $self->{filename};
}

sub _state {
    my $self = shift;
    #die "private property _state called" unless caller[0]->isa("MMPLD");
    if (@_) {
        my $val = shift;
        if (   $val == $NOTHINGGOES
            or $val == $CANADDFRAME
            or $val == $CANADDLIST
            or $val == $CANADDPARTICLE
            or $val == $ALLDONE
            or $val == $CLOSED) {
            $self->{state} = $val;
        } else {
            die "trying to set illegal state $val";
        }
    }
    return $self->{state};
}

sub _stringState {
    my $self = shift;
    #die "private method _stringState called" unless caller[0]->isa(ref($self));
    if ($self->{state} == $NOTHINGGOES) {
        return "uninitialized";
    }
    if ($self->{state} == $CANADDFRAME) {
        return "waiting for a frame";
    }
    if ($self->{state} == $CANADDLIST) {
        return "waiting for a list";
    }
    if ($self->{state} == $CANADDPARTICLE) {
        return "waiting for a particle";
    }
    if ($self->{state} == $ALLDONE) {
        return "all done";
    }
    if ($self->{state} == $CLOSED) {
        return "closed"
    }
    return "illegal state: " . $self->{state};
}

sub NumFrames {
    my $self = shift;
    if (@_) {
        $self->{numframes} = shift;
    }
    return $self->{numframes};
}

sub AppendStuff {
    my $self = shift;
    my $type = shift;
    my @rest = @_;
    if ($self->{filehandle}) {
        print {$self->{filehandle}} pack($type x ($#rest + 1), @rest);
    }
}

sub AppendUBytes {
    my $self = shift;
    $self->AppendStuff('C', @_);
}

sub AppendBytes {
    my $self = shift;
    $self->AppendStuff('c', @_);
}

sub AppendUShorts {
    my $self = shift;
    $self->AppendStuff('S', @_);
}

sub AppendShorts {
    my $self = shift;
    $self->AppendStuff('s', @_);
}

sub AppendUInt32s {
    my $self = shift;
    $self->AppendStuff('L', @_);
}

sub AppendInt32s {
    my $self = shift;
    $self->AppendStuff('l', @_);
}

sub AppendUInt64s {
    my $self = shift;
    $self->AppendStuff('Q', @_);
}

sub AppendInt64s {
    my $self = shift;
    $self->AppendStuff('q', @_);
}

sub AppendFloats {
    my $self = shift;
    $self->AppendStuff('f', @_);
}

sub AppendDoubles {
    my $self = shift;
    $self->AppendStuff('d', @_);
}

sub _checkColor {
    my @gc = @_;
    my $ok = 0;
    if ($#gc == 3) {
        $ok = 1;
        for (my $x = 0; $x < 4; $x++) {
            if (int($gc[$x]) != $gc[$x] || $gc[$x] < 0 || $gc[$x] > 255) {
                $ok = 0;
            }
        }
    }
    return $ok;
}

sub _checkUShortColor {
    my @gc = @_;
    my $ok = 0;
    if ($#gc == 3) {
        $ok = 1;
        for (my $x = 0; $x < 4; $x++) {
            if (int($gc[$x]) != $gc[$x] || $gc[$x] < 0 || $gc[$x] > 65535) {
                $ok = 0;
            }
        }
    }
    return $ok;
}

sub _checkFloatColor {
    my @gc = @_;
    my $ok = 0;
    if ($#gc == 3 || $#gc == 2) {
        $ok = 1;
        for (my $x = 0; $x <= $#gc; $x++) {
            if ($gc[$x] < 0 || $gc[$x] > 1.0) {
                $ok = 0;
            }
        }
    }
    return $ok;
}

sub _adjustBounds {
    my $self = shift;
    my ($x, $y, $z) = @_;
    
    if ($self->{firstparticle} == 1) {
        $self->{minx} = $self->{maxx} = $x;
        $self->{miny} = $self->{maxy} = $y;
        $self->{minz} = $self->{maxz} = $z;
        $self->{firstparticle} = 0;
    } else {
        if ($x < $self->{minx}) {
            $self->{minx} = $x;
        }
        if ($y < $self->{miny}) {
            $self->{miny} = $y;
        }
        if ($z < $self->{minz}) {
            $self->{minz} = $z;
        }
        if ($self->{maxx} < $x) {
            $self->{maxx} = $x;
        }
        if ($self->{maxy} < $y) {
            $self->{maxy} = $y;
        }
        if ($self->{maxz} < $z) {
            $self->{maxz} = $z;
        }
    }
}

1;
