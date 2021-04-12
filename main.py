# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import time
from PIL import Image
import multiprocessing
from joblib import Parallel, delayed
class Procesamiento_imagenes():
    def __init__(self,ancho=10):
        # imagen monocromatica o binaria
        self.ancho=ancho
        self.imagen_pil=Image.open('imagen.jpg')
        self.imagen_mono=np.array(self.imagen_pil.convert("L"))
        self.imagen_color = np.array(self.imagen_pil)

    def mostrar(self):
        print(self.imagen_mono.shape)
        print(self.imagen_color.shape)
        print(self.imagen_pil.getpixel((20,20)))
        print(self.ancho)
    def conversion_color2monocromatico(self):
        # pixel_mono=(R+G+B)/3
        renglon,columna=self.imagen_pil.size
        imagemonocromatica=Image.new('L',(renglon,columna))
        pixels=imagemonocromatica.load()
        for i in range(renglon):
            for j in range(columna):
                red,green,blue=self.imagen_pil.getpixel((i,j))
                #imagemonocromatica.putpixel((columna,renglon),int((red+green+blue)/3))
                pixels[i,j]=int((red+green+blue)/3)
        #imagemonocromatica.show()
        return imagemonocromatica
    def convolucion_secuencial(self):
        immono=self.conversion_color2monocromatico()
        renglon,columna=self.imagen_pil.size
        imagen_convolucion = Image.new('L', (renglon, columna))
        pixels = imagen_convolucion.load()
        time_start=time.time()
        for i in range(renglon):
            for j in range(1,columna-1):
                pixels[i,j]=abs(immono.getpixel((i,j+1))-immono.getpixel((i,j-1)))
        elapsed=time.time()-time_start
        print(elapsed)
        imagen_convolucion.show()
    def convolucion_secuencial_ventana(self):
        immono=self.conversion_color2monocromatico()
        renglon,columna=self.imagen_pil.size
        imagen_convolucion = Image.new('L', (renglon, columna))
        pixels = imagen_convolucion.load()
        time_start=time.time()
        for i in range(1,renglon-1):
            for j in range(1,columna-1):
                pixels[i,j]=int((immono.getpixel((i,j+1))+immono.getpixel((i,j-1))+immono.getpixel((i-1,j-1))+immono.getpixel((i-1,j))+immono.getpixel((i-1,j+1))+immono.getpixel((i+1,j-1))+immono.getpixel((i+1,j))+immono.getpixel((i+1,j+1)))/8)
        elapsed=time.time()-time_start
        print(elapsed)
        imagen_convolucion.show()
    def convolucion(self,lista_pixeles):
        resultado=abs(lista_pixeles[0]-lista_pixeles[1])
        return resultado
    def convolucion_parallel(self):
        immono=self.conversion_color2monocromatico()
        renglon,columna=self.imagen_pil.size
        imagen_convolucion = Image.new('L', (renglon, columna))
        pixels = imagen_convolucion.load()
        num_cores=multiprocessing.cpu_count()
        print('número de procesadores',num_cores)
        # preparo los datos en columnas x 2 renglones
        # [[1,3],[2,4],[3,5],...]
        datos=[]
        for i in range(renglon):
            for j in range(1,columna-1):
                datos.append([immono.getpixel((i,j+1)),immono.getpixel((i,j-1))])
        time_start=time.time()
        # se ejecutan en paralelo
        result=Parallel(n_jobs=num_cores)(delayed(self.convolucion)(i)for i in datos)
        elapsed=time.time()-time_start
        print('tiempo de ejecución',elapsed)
        #reacomodo la lista para que tenga la forma de una imagen, de 1xn_pixeles a mxn pixeles
        x=np.array(result)
        y=np.transpose(np.reshape(x,(renglon,columna-2)))
        im_result=Image.fromarray(np.uint8(y))
        im_result.show()
    def binarizacion(self):
        imagen_mono=Image.open('pastillas_simple.jpeg').convert("L")
        renglon,columna=imagen_mono.size
        imagen_binaria = Image.new('L', (renglon, columna))
        valor_referencia=180
        for i in range(renglon):
            for j in range(columna):
                if imagen_mono.getpixel((i,j))>valor_referencia:
                    imagen_binaria.putpixel((i, j), (255))
                else:
                    imagen_binaria.putpixel((i, j), (0))
        imagen_binaria.show()
    def crea_mosaico(self):
        imagen=Image.open('lena.jpeg')
        renglon,columna=imagen.size
        print(renglon,columna)
        imc1=imagen.crop((0,0,55,55))
        imc2=imagen.crop((56,0,110,55))
        imc3=imagen.crop((111,0,165,55))
        imc4=imagen.crop((166,0,224,55))
        imc1.save("lena1.jpg")
        imc2.save("lena2.jpg")
        imc3.save("lena3.jpg")
        imc4.save("lena4.jpg")
        imc5=imagen.crop((0,56,55,111))
        imc6=imagen.crop((56,56,110,111))
        imc7=imagen.crop((111,56,165,111))
        imc8=imagen.crop((166,56,224,111))
        imc5.save("lena5.jpg")
        imc6.save("lena6.jpg")
        imc7.save("lena7.jpg")
        imc8.save("lena8.jpg")
        imc1.show()
    def traslacion(self):
        p=np.array([5,8,1])
        p_t=np.transpose(p)
        tx=4
        ty=3
        nuevo_punto=(5+4,8+3)
        mt=np.array([[1,0,0],[0,1,0],[tx,ty,1]])
        n_p=mt*p_t
        print(n_p)

# aqui se crea el objeto
miimagen=Procesamiento_imagenes()
#miimagen.crea_mosaico()
#miimagen.binarizacion()
#miimagen.convolucion_secuencial()
#miimagen.convolucion_secuencial_ventana()
#miimagen.convolucion_parallel()
#miimagen.mostrar()
#miimagen.conversion_color2monocromatico()
# transformaciones geometricas
miimagen.traslacion()
# operaciones morfologicas
# imagenes en color
# operaciones morfológicas
# transformada de Hough
# segmentación
# estereovisión