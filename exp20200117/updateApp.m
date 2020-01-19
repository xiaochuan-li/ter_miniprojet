function varargout = updateApp(varargin)

persistent app_plot 
%create a run time object that can return the value of the gain block's
%output and then put the value in a string.
rto1 = get_param('exampleModell/Out1','RuntimeObject');
str1 = num2str(rto1.InputPort(1).Data);

rto2 = get_param('exampleModell/Out2','RuntimeObject');
str2 = num2str(rto2.InputPort(1).Data);

%get a handle to the GUI's 'current state' window
all_tag_objects = findall(0, '-property', 'tag');
all_tags = get(all_tag_objects, 'tag');
[tf, idx] = ismember('Out1', all_tags);
if tf  
  st1 = all_tag_objects(idx);
end
[tf, idx] = ismember('Out2', all_tags);
if tf  
  st2 = all_tag_objects(idx);
end
[tf, idx] = ismember('UIAxes', all_tags);
if tf  
  app_plot = all_tag_objects(idx);
end

XData = get_param('exampleModell','SimulationTime');
YData = rto1.InputPort(1).Data; 
%update the gui
set(st2,'Value',str2double(str2));
set(st1,'Value',str2double(str1));

plot(app_plot,XData,YData,'.');
hold(app_plot, 'on' );
drawnow;

