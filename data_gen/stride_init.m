function [ stride1, stride2 ] = stride_init( hei, wid, a, b, size_input)
%STRIDE_INIT Summary of this function goes here
%   Detailed explanation goes here
bn1 = 1;
for ii = 1:1000
    bn1 = bn1+1;
    
    if hei < a+bn1*b
        if hei - bn1*b < size_input/3
            bn1 = bn1-1;
        end
        break;
    end
    
end
bn2 = 1;
for ii = 1:1000
    bn2 = bn2+1;
    if wid < a+bn2*b
        if wid - bn2*b < size_input/3
            bn2 = bn2-1;
        end
        break;
    end
end
    if hei < 150
        stride1 = 60;
    else
        stride1 = floor((hei-size_input)/(bn1-1));
    end
    if wid < 150
        stride2 = 60;
    else
        stride2 = floor((wid-size_input)/(bn2-1));
    end
    if hei == size_input
        stride1 = size_input;
    end
    if wid == size_input
        stride2 = size_input;
    end

end
