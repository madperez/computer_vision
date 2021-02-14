# segmentation by seed 
seeds=[]
imagen=imread('semantica.png');
imshow(imagen);
seed=[20,30]
seeds=push(seeds,seed)
for i=1:10
  [seeds,seed]=pop(seeds)
  val_pixel=imagen(seed(1),seed(2))
  neighborhood=get_neighborhood(seed);
  [r,c]=size(neighborhood)  
  for j=1:r
    val_pixel_neighborhood=imagen(neighborhood(j,1),neighborhood(j,2))
    if val_pixel==val_pixel_neighborhood
      seeds=push(seeds,neighborhood(j,:))
    endif
  endfor
endfor
  
