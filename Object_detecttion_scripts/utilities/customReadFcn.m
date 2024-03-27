function data = customReadFcn(datastore)
    % Read the next batch of data from the datastore
    tmp = readall(datastore);
    
    % Initialize the output cell array
    data = cell(size(tmp, 2)/3, 3);
    
    % Loop through each set of image data, bounding boxes, and labels
    for i = 1:size(data, 1)
        % Image data is in the first three columns
        data{i, 1} = cat(3, tmp{1, (i-1)*3+1}, tmp{1, (i-1)*3+2}, tmp{1, (i-1)*3+3});
        
        % Bounding boxes and labels are in pairs after the image data
        data{i, 2} = tmp{1, size(tmp, 2)/3 + (i-1)*2 + 1}; % Bounding box coordinates
        data{i, 3} = tmp{1, size(tmp, 2)/3 + (i-1)*2 + 2}; % Bounding box labels
    end
end