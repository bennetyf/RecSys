data = test;

for i=1:size(data,1)
    if size(data(data(i,:) ~= 0),2) == 0
        fprintf("%d Alert\n",i)
    end
end

