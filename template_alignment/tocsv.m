for i = 1 : 6
    
    file = sprintf('object_template_%d.pcd', i-1);
    tofile = sprintf('object_template_%d.csv', i-1);
    pc = pcread(file);
    loc = pc.Location;
    
    csvwrite(tofile, loc);
    
end
    