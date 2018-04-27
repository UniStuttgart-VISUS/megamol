function [ colCnt, rowCnt, col_info, data ] = mmft2matlab( filename )
% reads a mmft file into matlab workspace

file = fopen(filename);

magic = fread(file, 6, '*char');
version = fread(file, 1, '*uint16');
colCnt = fread(file, 1, '*uint32');

field1 = 'name';
field2 = 'type';
field3 = 'minimum';
field4 = 'maximum';

for ci = 1:colCnt
    name_len = fread(file, 1, '*uint16');
    name = fread(file, name_len, '*char');
    type = fread(file, 1, '*uint8');
    min_val = fread(file, 1, '*float');
    max_val = fread(file, 1, '*float');
    
    col_info(ci) = struct(field1, name, field2, type, field3, min_val, field4, max_val);
end

rowCnt = fread(file, 1, '*uint64');

data = fread(file, [colCnt rowCnt], '*float');    

end

