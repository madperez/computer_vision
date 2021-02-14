function [list_reduced,value]=pop(list)
  value=list(end,:);
  list_reduced=list(1:end-1,:);
endfunction
