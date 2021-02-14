function neighborhood=get_neighborhood(seed)
  neighborhood=[seed(1)-1,seed(2)-1;seed(1)-1,seed(2);seed(1)-1,seed(2)+1;seed(1),seed(2)-1;seed(1),seed(2)+1;seed(1)+1,seed(2)-1;seed(1)+1,seed(2);seed(1)+1,seed(2)+1];
endfunction
