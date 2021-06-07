# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
from matplotlib import pyplot as plt
import numpy as np
import time
from PIL import Image, ImageFilter
import multiprocessing
from joblib import Parallel, delayed
class Medio_nivel():
    def __init__(self):
        self.imagen=np.array([[1,0,1,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]])
    def transformada_hough_academico(self):
        angulos=4
        valor_angulo=[0,45*2*math.pi/360,90*2*math.pi/360,135*2*math.pi/360]
        renglon,columna=self.imagen.shape
        distancia=renglon
        acumuladores=np.zeros([distancia,angulos])
        for r in range(renglon):
            for c in range(columna):
                if self.imagen[r,c]==1:
                    for j in range(angulos):
                        distance=int(r*math.sin(valor_angulo[j])+c*math.cos(valor_angulo[j]))
                        print(distance)
                        acumuladores[distance,j]+=1
                    print(self.imagen,acumuladores)
        valor_maximo=np.max(acumuladores)
        posicion=np.argmax(acumuladores)
        print(valor_maximo,posicion)
        # posicion 3 corresponde a una distancia 0 y un angulo 135 xsin(135)+ycos(135)=0
        # y=(-xsin(135))/cos(135)
        for x in range(3):
            y=(x*math.sin(valor_angulo[x]))/math.cos(valor_angulo[x])
            print(y)


    def transformada_hough(self):
        imagen_color=Image.open('pasillo.jpeg')
        imagen_mono=imagen_color.convert('L')
        imagen_bordes=imagen_mono.filter(ImageFilter.FIND_EDGES)
        imagen_bordes.show()
        renglon,columna=imagen_bordes.size
        print(renglon,columna)
        angulos = 4
        valor_angulo = np.arange(0,3.14,.1)
        print(valor_angulo)
        #renglon, columna = self.imagen.shape
        distancia = renglon
        acumuladores = np.zeros([distancia, valor_angulo.size])
        for r in range(renglon):
            for c in range(columna):
                if imagen_bordes.getpixel((r, c)) > 125:
                    for j in range(angulos):
                        distance = int(r * math.sin(valor_angulo[j]) + c * math.cos(valor_angulo[j]))
                        #print(distance)
                        acumuladores[distance, j] += 1
        print(acumuladores)
        valor_maximo = np.max(acumuladores)
        posicion = np.argmax(acumuladores)
        print(valor_maximo, posicion)
        # posicion 3 corresponde a una distancia 0 y un angulo 135 xsin(135)+ycos(135)=0
        # y=(-xsin(135))/cos(135)
        for x in range(3):
            y = (x * math.sin(valor_angulo[x])) / math.cos(valor_angulo[x])
            print(y)


hough=Medio_nivel()
hough.transformada_hough()
class Profundidad():
    def __init__(self, imagen=Image.open('leverkusen_depth.png')):
        self.image_depth = imagen.resize((1024, 512))
        #self.image_depth = imagen
        renglon,columna=imagen.size
        self.image_result = Image.new('L', (renglon, columna))
    def show_region(self,list_region):
        # muestra los pixeles que pertenecen a una region y devuelve una lista con las coordenadas de cada pixel
        pixels_region=list_region
        depth_region=[]
        for r,c in pixels_region:
            depth_region.append(self.image_depth.getpixel((c,r)))
            self.image_result.putpixel((c,r),self.image_depth.getpixel((c,r)))
        self.image_result.show()
        return depth_region


class Segmentacion():
    def __init__(self,imagen=Image.open('leverkusen_semantica.png')):
        self.clasificados=[]
        self.lut_colors=[(0,0,255),(0,125,255),(0,255,255),(0,255,125),(0,255,0),(125,255,0),(255,255,0),(255,125,0),(255,0,0)]
        self.pointer_class=0
        self.segmentos=[]
        self.area=[]
        self.imagen=imagen
        renglon,columna=imagen.size
        self.imagen_segmentada = Image.new('RGB', (renglon, columna))
        self.imagen_result = Image.new('RGB', (renglon, columna))

    def show_region(self,id_region):
        # muestra los pixeles que pertenecen a una region y devuelve una lista con las coordenadas de cada pixel
        pixels_region=self.segmentos[id_region]
        for r,c in pixels_region:
            self.imagen_result.putpixel((c,r),self.get_color_class())
        self.imagen_result.show()
        return pixels_region
    def get_color_class(self):
        # obtiene el color con que se pinta la clase actual marcada por pointer_class
        return self.lut_colors[self.pointer_class]
    def set_new_class(self):
        # asigna un valor para la nueva clase
        self.pointer_class+=1
        if self.pointer_class==len(self.lut_colors):
            self.pointer_class=0
    def get_seccion(self, threshold=0, ren_seed=100, col_seed=100, imagen=None):
        # segmenta una seccion por crecimiento de regiones
        max_size_region=8000
        self.semillas = []
        height, width = imagen.size
        self.semillas.append([col_seed, ren_seed])
        val_pixel_ref = imagen.getpixel((ren_seed,col_seed))
        #print('pixel de referencia',val_pixel_ref)
        ciclos = 0
        self.segmento = []
        clases = []
        while len(self.semillas) > 0 and ciclos < max_size_region:
            # obtiene una semilla
            col_seed, ren_seed = self.semillas.pop()
            # valida si no está en las esquinas
            if col_seed > 0 and ren_seed > 0 and col_seed < width - 1 and ren_seed < height - 1 and [col_seed,ren_seed] not in self.clasificados:
                cuenta = 0
                # agrega los vecinos suprimiendo el central
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if cuenta != 4:
                            val_pixel = imagen.getpixel((ren_seed + i, col_seed + j))
                            # se discriminan aquellos que tienen colores diferentes
                            if val_pixel == val_pixel_ref:
                            #if abs(val_pixel - val_pixel_ref) <= threshold:
                                # se evitan los duplicados
                                if [col_seed + i, ren_seed + j] not in self.segmento:
                                    self.semillas.append(
                                        [col_seed + i, ren_seed + j])  # guarda las semillas de esta region
                                    self.segmento.append(
                                        [col_seed + i, ren_seed + j])  # guarda los clasificados en esta region
                                    self.imagen_segmentada.putpixel((ren_seed,col_seed),self.get_color_class())
                        cuenta += 1
                ciclos += 1
        # detecta la etiqueta mayoritaria
        # clases=[]
        contador_clase = []
        for pixel in self.segmento:
            # print(pixel[0],pixel[1])
            clase_pixel = imagen.getpixel((pixel[1], pixel[0]))
            if clase_pixel not in clases:
                clases.append(clase_pixel)
                contador_clase.append(0)
            else:
                indice = clases.index(clase_pixel)
                contador_clase[indice] += 1
        try:
            max_clases = max(contador_clase)
            indice_max = contador_clase.index(max_clases)
            clase_mayoritaria = clases[indice_max]
            for i in self.segmento:
                self.img_out.putpixel((i[0], i[1]), clase_mayoritaria)
        except:
            pass
        return self.segmento
    def segmenta_imagen(self):
        imagen_semantica = self.imagen
        renglon, columna = imagen_semantica.size
        imagen_autos = Image.new('L', (renglon, columna))

        flag = 0
        cuenta = 0
        for r in range(renglon):
            for c in range(columna):
                red, green, blue = imagen_semantica.getpixel((r, c))
                if blue == 142 and [r,c] not in self.clasificados:
                    segmento = self.get_seccion(0,r, c, imagen_semantica)
                    self.clasificados=self.clasificados+segmento
                    cuenta+=1
                    if len(segmento)>0:
                        self.segmentos.append(segmento)
                        self.set_new_class()
                        self.area.append(len(segmento))
        self.imagen_segmentada.show()

#misegmento=Segmentacion(Image.open('leverkusen_semantica.png'))
#miprofundidad=Profundidad()
#misegmento.segmenta_imagen()
#print(len(misegmento.segmentos))
#print(misegmento.area)
#list_segmento=misegmento.show_region(1)
#val_depth=miprofundidad.show_region(list_segmento)
#print(val_depth)
class Procesamiento_imagenes():
    def __init__(self, ancho=10):
        # imagen monocromatica o binaria
        self.ancho = ancho
        self.imagen_pil = Image.open('imagen.jpg')
        self.imagen_pastillas = Image.open('pastillas.jpeg')
        self.imagen_mono = np.array(self.imagen_pil.convert("L"))
        self.imagen_color = np.array(self.imagen_pil)
        self.clasificados=[]
        self.lut_colors=[(0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0)]
        self.pointer_class=0
        self.segmentos=[]

    def erosion(self):
        imagen_binaria = self.binarizacion()
        elemento_estructurante = np.array([1, 1, 1])
        renglon, columna = imagen_binaria.size
        imagen_erosionada = Image.new('L', (renglon, columna))
        for i in range(renglon):
            for j in range(1, columna - 1):
                ventana = [imagen_binaria.getpixel((i, j - 1)), imagen_binaria.getpixel((i, j)),
                           imagen_binaria.getpixel((i, j + 1))]
                valor_minimo = 255
                for k in range(3):
                    valor_multiplicacion = elemento_estructurante[k] * ventana[k]
                    if valor_multiplicacion < valor_minimo:
                        valor_minimo = valor_multiplicacion
                    # print(elemento_estructurante[i],ventana[i],valor_minimo)
                imagen_erosionada.putpixel((i, j), (int(valor_minimo)))
        Image.fromarray(np.hstack((np.array(imagen_binaria), np.array(np.uint8(imagen_erosionada))))).show()

    def estereo(self):
        imagen_izquierda = Image.open('tsukubal.png').convert('L')
        imagen_derecha = Image.open('tsukubar.png').convert('L')
        renglon, columna = imagen_derecha.size
        imagen_profundidad = Image.new('L', (renglon, columna))
        window_size = 5
        dmax = 15
        error = np.zeros(dmax)
        # for r in range(70,71):
        #   for c in range(80,81):
        for r in range(window_size, renglon - window_size):
            for c in range(window_size, columna - window_size - dmax):
                dmin = 0
                emin = 255 * 4 * window_size
                for d in range(dmax):
                    suma_error = 0
                    for j in range(-window_size, window_size + 1):
                        for k in range(-window_size, window_size + 1):
                            suma_error += abs(
                                imagen_izquierda.getpixel((r + j, c + k)) - imagen_derecha.getpixel((r + j, c + k + d)))
                    error[d] = suma_error
                    if suma_error < emin:
                        emin = suma_error
                        dmin = d
                # plt.plot(error)
                # plt.show()
                imagen_profundidad.putpixel((r, c), int(dmin * 15))
        imagen_profundidad.show()

        # Image.fromarray(np.hstack((np.array(imagen_izquierda), np.array(np.uint8(imagen_derecha))))).show()

    def mostrar(self):
        print(self.imagen_mono.shape)
        print(self.imagen_color.shape)
        print(self.imagen_pil.getpixel((20, 20)))
        print(self.ancho)

    def manejo_color(self):
        imagen_rgb = Image.new('RGB', (20, 20))
        imagen_rgb.putpixel((5, 5), (255, 0, 0))
        imagen_rgb.putpixel((6, 5), (0, 255, 0))
        imagen_rgb.show()
        return

    def conversion_color2monocromatico(self):
        # pixel_mono=(R+G+B)/3
        renglon, columna = self.imagen_pil.size
        imagemonocromatica = Image.new('L', (renglon, columna))
        pixels = imagemonocromatica.load()
        for i in range(renglon):
            for j in range(columna):
                red, green, blue = self.imagen_pil.getpixel((i, j))
                # imagemonocromatica.putpixel((columna,renglon),int((red+green+blue)/3))
                pixels[i, j] = int((red + green + blue) / 3)
        # imagemonocromatica.show()
        return imagemonocromatica

    def convolucion_secuencial(self):
        immono = self.conversion_color2monocromatico()
        renglon, columna = self.imagen_pil.size
        imagen_convolucion = Image.new('L', (renglon, columna))
        pixels = imagen_convolucion.load()
        time_start = time.time()
        for i in range(renglon):
            for j in range(1, columna - 1):
                pixels[i, j] = abs(immono.getpixel((i, j + 1)) - immono.getpixel((i, j - 1)))
        elapsed = time.time() - time_start
        print(elapsed)
        imagen_convolucion.show()

    def convolucion_secuencial_ventana(self):
        immono = self.conversion_color2monocromatico()
        renglon, columna = self.imagen_pil.size
        imagen_convolucion = Image.new('L', (renglon, columna))
        pixels = imagen_convolucion.load()
        time_start = time.time()
        for i in range(1, renglon - 1):
            for j in range(1, columna - 1):
                pixels[i, j] = int((immono.getpixel((i, j + 1)) + immono.getpixel((i, j - 1)) + immono.getpixel(
                    (i - 1, j - 1)) + immono.getpixel((i - 1, j)) + immono.getpixel((i - 1, j + 1)) + immono.getpixel(
                    (i + 1, j - 1)) + immono.getpixel((i + 1, j)) + immono.getpixel((i + 1, j + 1))) / 8)
        elapsed = time.time() - time_start
        print(elapsed)
        imagen_convolucion.show()

    def convolucion(self, lista_pixeles):
        resultado = abs(lista_pixeles[0] - lista_pixeles[1])
        return resultado

    def convolucion_ventana(self, lista_pixeles):
        resultado = abs(lista_pixeles[0] + lista_pixeles[1] + lista_pixeles[2] + lista_pixeles[3] + lista_pixeles[4] +
                        lista_pixeles[5] + lista_pixeles[6] + lista_pixeles[7]) / 8
        return resultado

    def convolucion_parallel(self):
        immono = self.conversion_color2monocromatico()
        renglon, columna = immono.size
        imagen_convolucion = Image.new('L', (renglon, columna))
        # pixels = imagen_convolucion.load()
        num_cores = multiprocessing.cpu_count()
        print('número de procesadores', num_cores)
        # preparo los datos en columnas x 2 renglones
        # [[1,3],[2,4],[3,5],...]
        datos = []
        for i in range(renglon):
            for j in range(1, columna - 1):
                datos.append([immono.getpixel((i, j + 1)), immono.getpixel((i, j - 1))])
        time_start = time.time()  # estimamos el tiempo de inicio
        # se ejecutan en paralelo
        result = Parallel(n_jobs=num_cores)(delayed(self.convolucion)(i) for i in datos)
        elapsed = time.time() - time_start
        print('tiempo de ejecución', elapsed)
        # reacomodo la lista para que tenga la forma de una imagen, de 1xn_pixeles a mxn pixeles
        x = np.array(result)
        y = np.transpose(np.reshape(x, (renglon, columna - 2)))
        im_result = Image.fromarray(np.uint8(y))
        im_result.show()

    def convolucion_parallel_ventana(self):
        immono = self.conversion_color2monocromatico()
        renglon, columna = immono.size
        imagen_convolucion = Image.new('L', (renglon, columna))
        # pixels = imagen_convolucion.load()
        num_cores = multiprocessing.cpu_count()
        print('número de procesadores', num_cores)
        # preparo los datos en columnas x 2 renglones
        # [[1,3],[2,4],[3,5],...]
        datos = []
        for i in range(1, renglon - 1):
            for j in range(1, columna - 1):
                datos.append(
                    [immono.getpixel((i - 1, j - 1)), immono.getpixel((i - 1, j)), immono.getpixel((i - 1, j + 1)),
                     immono.getpixel((i, j - 1)), immono.getpixel((i, j + 1)), immono.getpixel((i + 1, j - 1)),
                     immono.getpixel((i + 1, j)), immono.getpixel((i + 1, j + 1))])
        time_start = time.time()
        # se ejecutan en paralelo
        result = Parallel(n_jobs=num_cores)(delayed(self.convolucion_ventana)(i) for i in datos)
        # result=Parallel(n_jobs=num_cores)(delayed(self.convolucion_ventana(i)for i in datos)
        elapsed = time.time() - time_start
        print('tiempo de ejecución', elapsed)
        # reacomodo la lista para que tenga la forma de una imagen, de 1xn_pixeles a mxn pixeles
        x = np.array(result)
        y = np.transpose(np.reshape(x, (renglon - 2, columna - 2)))
        im_result = Image.fromarray(np.uint8(y))
        im_result.show()

    def binarizacion(self):
        imagen_mono = Image.open('pastillas_simple.jpeg').convert("L")
        renglon, columna = imagen_mono.size
        imagen_binaria = Image.new('L', (renglon, columna))
        valor_referencia = 180
        for i in range(renglon):
            for j in range(columna):
                if imagen_mono.getpixel((i, j)) > valor_referencia:
                    imagen_binaria.putpixel((i, j), (255))
                else:
                    imagen_binaria.putpixel((i, j), (0))
        # imagen_binaria.show()
        return imagen_binaria

    def crea_mosaico(self):
        imagen = Image.open('lena.jpeg')
        renglon, columna = imagen.size
        print(renglon, columna)
        imc1 = imagen.crop((0, 0, 55, 55))
        imc2 = imagen.crop((56, 0, 110, 55))
        imc3 = imagen.crop((111, 0, 165, 55))
        imc4 = imagen.crop((166, 0, 224, 55))
        imc1.save("lena1.jpg")
        imc2.save("lena2.jpg")
        imc3.save("lena3.jpg")
        imc4.save("lena4.jpg")
        imc5 = imagen.crop((0, 56, 55, 111))
        imc6 = imagen.crop((56, 56, 110, 111))
        imc7 = imagen.crop((111, 56, 165, 111))
        imc8 = imagen.crop((166, 56, 224, 111))
        imc5.save("lena5.jpg")
        imc6.save("lena6.jpg")
        imc7.save("lena7.jpg")
        imc8.save("lena8.jpg")
        imc1.show()

    def traslacion(self):
        p = np.array([5, 8, 1])
        p_t = np.transpose(p)
        tx = 4
        ty = 3
        nuevo_punto = (5 + 4, 8 + 3)
        mt = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        nc = np.transpose(np.array([0, 0, 0]))
        for i in range(3):
            for j in range(3):
                print(mt[i][j], p_t[j])
                nc[i] += mt[i][j] * p_t[j]
            print(nc)
        n_p = mt * p_t
        print(n_p)

    def escalamiento(self, sx, sy):
        matriz_escalamiento = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        renglon, columna = self.imagen_pil.size
        imagen_escalada = Image.new('RGB', (renglon * 2, columna * 2))
        for r in range(renglon):
            for c in range(columna):
                nuevo_p = np.transpose(np.array([0, 0, 0]))
                valor_pixel = self.imagen_pil.getpixel((r, c))
                p = np.transpose(np.array([r, c, 1]))
                for i in range(3):
                    for j in range(3):
                        nuevo_p[i] += matriz_escalamiento[i][j] * p[j]
                imagen_escalada.putpixel((nuevo_p[0], nuevo_p[1]), valor_pixel)
        imagen_escalada.show()
        imagen_escalada.save('imescalada.jpg')

    def interpolacion_bilineal(self):
        imagen = Image.open('imescalada.jpg').convert('L')
        renglon, columna = imagen.size
        for i in range(1, renglon - 1, 2):
            for j in range(1, columna - 1, 2):
                valor_pixel = int((imagen.getpixel((i - 1, j - 1)) + imagen.getpixel((i - 1, j + 1)) + imagen.getpixel(
                    (i + 1, j - 1)) + imagen.getpixel((i + 1, j + 1))) / 4)
                imagen.putpixel((i, j), valor_pixel)
        imagen.show()

    def multiplicacion(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([4, 5, 6])
        print(a * b)

    def proyeccion(self, focal_distance=3, punto_3d=np.array([2, 3, 5])):
        renglones, columnas = punto_3d.shape
        nuevo_punto = np.zeros((renglones, columnas))
        for i in range(renglones):
            matriz_proyeccion = np.array(
                [[focal_distance / punto_3d[i][2], 0, 0], [0, focal_distance / punto_3d[i][2], 0], [0, 0, 1]])
            nuevo_punto[i] = np.matmul(matriz_proyeccion, punto_3d[i])
        return nuevo_punto

    def test_proyeccion(self):
        punto_3d = np.array(
            [[1, 1, 1], [3, 1, 1], [3, 2, 1], [1, 2, 1], [1, 1, 1], [1, 1, 4], [3, 1, 4], [3, 2, 4], [1, 2, 4],
             [1, 1, 4]])
        rx, ry, rz = self.rotacion(10, punto_3d)
        print(punto_3d)
        print(ry)
        nuevo_punto = self.proyeccion(3, ry)
        print(nuevo_punto)
        plt.title("Proyecciones")
        plt.xlabel("x axis caption")
        plt.ylabel("y axis caption")
        plt.plot(nuevo_punto[:, 0], nuevo_punto[:, 1])
        plt.show()

    def rotacion(self, angle_deg=10, punto=np.array([1, 1, 1])):
        renglones, columnas = punto.shape
        px = np.zeros((renglones, columnas))
        py = np.zeros((renglones, columnas))
        pz = np.zeros((renglones, columnas))
        angle_rad = angle_deg * 2 * 3.14 / 360
        mrx = np.array(
            [[1, 0, 0], [0, math.cos(angle_rad), -math.sin(angle_rad)], [0, math.sin(angle_rad), math.cos(angle_rad)]])
        mry = np.array(
            [[math.cos(angle_rad), 0, math.sin(angle_rad)], [0, 1, 0], [-math.sin(angle_rad), 0, math.cos(angle_rad)]])
        mrz = np.array(
            [[math.cos(angle_rad), -math.sin(angle_rad), 0], [math.sin(angle_rad), math.cos(angle_rad), 0], [0, 0, 1]])
        for i in range(renglones):
            px[i] = np.matmul(mrx, punto[i])
            py[i] = np.matmul(mry, punto[i])
            pz[i] = np.matmul(mrz, punto[i])
        return px, py, pz

    def test_rotacion(self):
        punto_3d = np.array([[1, 1, 1], [3, 1, 1], [3, 2, 1], [1, 2, 1], [1, 1, 1]])
        rx, ry, rz = self.rotacion(10, punto_3d)
        print(punto_3d)
        print(rx, ry, rz)
        plt.figure(1)
        ax1 = plt.subplot(311)
        plt.plot(rx[:, 0], rx[:, 1])
        plt.setp(ax1.get_xticklabels(), fontsize=6)
        ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
        plt.plot(ry[:, 0], ry[:, 1])
        ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
        plt.plot(rz[:, 0], rz[:, 1])
        plt.show()

    def usa_proyeccion(self):
        img_semantica = Image.open('leverkusen_semantica.png')
        img_depth = Image.open('leverkusen_depth.png').convert('L')
        renglon, columna = img_semantica.size
        print(renglon, columna)
        imagen_proyeccion = Image.new('RGB', (renglon, columna))
        for r in range(renglon):
            for c in range(columna):
                value_red, value_green, value_blue = img_semantica.getpixel((r, c))
                value_pixel = img_semantica.getpixel((r, c))
                value_depth = img_depth.getpixel((r, c))
                coordenada_punto = np.array([r, c, value_depth])
                puntox, puntoy, puntoz = self.rotacion(0, coordenada_punto)
                if value_blue == 142:
                    # print(img_semantica.getpixel((r, c)))
                    x, y, z = self.proyeccion(1, puntoy)
                    print(x, y, z)
                    if x > 0 and y > 0:
                        imagen_proyeccion.putpixel((int(x), int(y)), value_pixel)
        imagen_proyeccion.show()


    def clustering_depth(self, threshold=20):
        # le asigna un solo valor de disparidad a pixeles con disparidad similar
        size_semantica = (1024, 768)
        imagen_profundidad = Image.open("leverkusen_depth.png")
        imagen_depth = imagen_profundidad.resize((1024, 512))
        renglon, columna = imagen_depth.size
        imagen_cluster = Image.new('L', (renglon, columna))
        valor_escala = 255 // threshold
        for r in range(renglon):
            for c in range(columna):
                imagen_cluster.putpixel((r, c), int(imagen_depth.getpixel((r, c)) // threshold * valor_escala))
        return imagen_cluster



# aqui se crea el objeto
#miimagen = Procesamiento_imagenes()
# miimagen.estereo()
# miimagen.manejo_color()
# miimagen.clustering_depth()
#miimagen.separa_objetos_por_profundidad()
# miimagen.test_rotacion()
# miimagen.usa_proyeccion()
# miimagen.rotacion()
# miimagen.erosion()
# miimagen.traslacion()
# miimagen.crea_mosaico()
# miimagen.binarizacion()
# miimagen.convolucion_secuencial()
# miimagen.convolucion_secuencial_ventana()
# miimagen.convolucion_parallel()
# miimagen.convolucion_parallel_ventana()
# miimagen.mostrar()
# miimagen.conversion_color2monocromatico()
# transformaciones geometricas
# miimagen.escalamiento(2,2)
# miimagen.interpolacion_bilineal()
# operaciones morfologicas
# imagenes en color
# operaciones morfológicas
# transformada de Hough
# segmentación
# estereovisión
